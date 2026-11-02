# A tile algorithm on exchangeable tiles (src/tilenames.jl, tiledir.jl, tilestore.jl): the tiled Cholesky factorization in the style of HPX.
#
# Every rank creates tasks only for the tiles it owns. A task produces a new version of its tile and needs, besides its own tile (which stays on the rank and
# is updated in place from task to task), the *final* versions of some tiles of earlier columns, which it asks for as pull futures: nobody has a plan that all
# ranks share, nobody has to know which rank runs which other task, and the set of ranks enters only through `owner(i, j)` and the directory. The requests go out
# when the tasks are created, not when they start: the owner holds a request until it has published the version, so the answer follows the publication at
# once (one hop on the critical path, like a push) instead of waiting for a request first. Only the final
# version of a tile is published (the intermediate versions never leave the rank); a rank asks for a final version once however many of its tasks use it.
#
# Tile (m, n), m >= n, is updated at steps k = 1..n-1 (syrk if m == n, gemm otherwise, using the final versions of (n, k) and, for gemm, (m, k)) and then finished
# (potrf if m == n, trsm with the final (n, n) otherwise); its final version is `n`.

"""
    tile_cholesky!(d, nt, init, owner; space=0x1) -> Dict{Tuple{Int,Int},Matrix}

Factorize (lower, `A = L L'`) the symmetric positive definite matrix of `nt` x `nt` tiles on the ranks of the directory `d`. `init(i, j)` returns the tile `(i, j)`
of the lower triangle (called only on its owner, and the result is changed in place), `owner(i, j)` is the identifier of the rank that owns it. Returns the final tiles
this rank owns after all its tasks have finished. Every rank must call it with the same `nt`, `owner` and `space`.
"""
function tile_cholesky!(d::TileDirectory, nt::Int, init, owner; space::UInt64=UInt64(1), prefetch::Bool=true, probe=nothing)
    me = d.me
    kern(kind, f, args...) = probe === nothing ? f(args...) : probe(kind, f, args...)     # `probe(kind, f, args...)` may time the kernel (benchmarks)
    name(i, j) = TileName(space, (j - 1) * nt + i)
    cache = Dict{Tuple{Int,Int},DataSlot}()               # final versions asked for by this rank: once, however many tasks use them
    clock = ReentrantLock()
    early(i, j) = prefetch ? final_tile(i, j) : nothing       # ask when the task is created ...
    late(s, i, j) = s === nothing ? final_tile(i, j) : s      # ... or only when it runs
    function final_tile(i, j)                              # the final version of (i, j) is version j
        lock(clock) do
            get!(cache, (i, j)) do
                tile_fetch(d, name(i, j), j; owner=owner(i, j))      # the layout says who owns it: no directory lookup
            end
        end
    end
    chains = Pair{Tuple{Int,Int},Task}[]
    for j in 1:nt, i in j:nt
        owner(i, j) == me || continue
        published = Threads.Atomic{Bool}(false)             # has the final version (or its failure) been published?
        guard(f) = try f() catch e
            if !Threads.atomic_xchg!(published, true)           # a failed task must not leave the ranks that wait for this tile waiting for ever
                tile_publish_failure!(d, name(i, j), j, CapturedException(e, catch_backtrace()))
            end
            rethrow()
        end
        cur = Threads.@spawn guard(() -> init(i, j))
        # N.B. each task gets its predecessor through a `let`: a captured variable that the loop reassigns would make the task wait for itself
        for k in 1:j-1
            if i == j
                cur = let prev = cur, sa = early(j, k)
                    Threads.@spawn guard() do
                        C = fetch(prev); A = fetch(late(sa, j, k))
                        kern(3, BLAS.syrk!, 'L', 'N', -1.0, A, 1.0, C)
                        C
                    end
                end
            else
                cur = let prev = cur, sa = early(i, k), sb = early(j, k)
                    Threads.@spawn guard() do
                        C = fetch(prev); A = fetch(late(sa, i, k)); B = fetch(late(sb, j, k))
                        kern(4, BLAS.gemm!, 'N', 'T', -1.0, A, B, 1.0, C)
                        C
                    end
                end
            end
        end
        if i == j
            cur = let prev = cur
                Threads.@spawn guard() do
                    C = fetch(prev)
                    _, info = kern(1, LAPACK.potrf!, 'L', C)
                    info == 0 || throw(LinearAlgebra.PosDefException(Int(info)))
                    Threads.atomic_xchg!(published, true)
                    tile_publish!(d, name(i, j), j, C; readers=-1)     # the final version: published, last
                    C
                end
            end
        else
            cur = let prev = cur, sl = early(j, j)
                Threads.@spawn guard() do
                    C = fetch(prev); L = fetch(late(sl, j, j))
                    kern(2, BLAS.trsm!, 'R', 'L', 'T', 'N', 1.0, L, C)
                    Threads.atomic_xchg!(published, true)
                    tile_publish!(d, name(i, j), j, C; readers=-1)
                    C
                end
            end
        end
        push!(chains, (i, j) => cur)
    end
    return Dict{Tuple{Int,Int},Matrix{Float64}}(ij => fetch(t) for (ij, t) in chains)
end

# --- more algorithms on the same blocks -----------------------------------------------------------------------------------------------------

# One step of the chain of a tile: wait for the tile's previous value and for the slots of the final tiles it needs, then run `f(C, values...)` on it in place.
# The previous value and the slots are arguments, not captured variables, so a loop that reassigns them cannot make the task wait for itself.
function _chain_step(guard, prev, slots, f)
    return Threads.@spawn guard() do
        C = fetch(prev); vals = map(fetch, slots)
        f(C, vals...)
        C
    end
end
# The last step of the chain: the same, and then the final version of the tile is published (a failure is published instead by `guard`).
function _chain_final(guard, prev, slots, f, d, nm, ver, published)
    return Threads.@spawn guard() do
        C = fetch(prev); vals = map(fetch, slots)
        f(C, vals...)
        Threads.atomic_xchg!(published, true)
        tile_publish!(d, nm, ver, C; readers=-1)
        C
    end
end
# the guard of one tile: the first failure of its chain publishes the failure of the tile's final version
function _chain_guard(d, nm, ver, published)
    return function (f)
        try
            f()
        catch e
            Threads.atomic_xchg!(published, true) || tile_publish_failure!(d, nm, ver, CapturedException(e, catch_backtrace()))
            rethrow()
        end
    end
end

"""
    tile_lu_nopiv!(d, nt, init, owner; space=UInt64(1), probe=nothing) -> Dict{Tuple{Int,Int},Matrix}

LU factorization without pivoting of a matrix of `nt` x `nt` tiles (`A = L U`, `L` unit lower), in the style of [`tile_cholesky!`](@ref): each rank makes the tasks of its own
tiles and pulls the final versions it needs. `init(i, j)` gives the tile (called on its owner, changed in place), `owner(i, j)` its rank. Returns the final tiles of this rank:
the strictly lower and diagonal-unit part of `L` and the upper part of `U` in the usual packed form. Tile `(i, j)` has the final version `min(i, j)`.
"""
function tile_lu_nopiv!(d::TileDirectory, nt::Int, init, owner; space::UInt64=UInt64(1), probe=nothing)
    me = d.me
    kern(kind, f, args...) = probe === nothing ? f(args...) : probe(kind, f, args...)
    name(i, j) = TileName(space, (j - 1) * nt + i)
    cache = Dict{Tuple{Int,Int},DataSlot}(); clock = ReentrantLock()
    final_tile(i, j) = lock(clock) do
        get!(cache, (i, j)) do
            tile_fetch(d, name(i, j), min(i, j); owner=owner(i, j))
        end
    end
    chains = Pair{Tuple{Int,Int},Task}[]
    for j in 1:nt, i in 1:nt
        owner(i, j) == me || continue
        m = min(i, j)
        published = Threads.Atomic{Bool}(false)
        guard = _chain_guard(d, name(i, j), m, published)
        cur = Threads.@spawn guard(() -> init(i, j))
        for k in 1:m-1
            cur = _chain_step(guard, cur, (final_tile(i, k), final_tile(k, j)), (C, A, B) -> kern(4, BLAS.gemm!, 'N', 'N', -1.0, A, B, 1.0, C))
        end
        if i == j
            cur = _chain_final(guard, cur, (), C -> kern(1, LinearAlgebra.lu!, C, LinearAlgebra.NoPivot()), d, name(i, j), m, published)
        elseif i > j
            cur = _chain_final(guard, cur, (final_tile(j, j),), (C, U) -> kern(2, BLAS.trsm!, 'R', 'U', 'N', 'N', 1.0, U, C), d, name(i, j), m, published)
        else
            cur = _chain_final(guard, cur, (final_tile(i, i),), (C, L) -> kern(2, BLAS.trsm!, 'L', 'L', 'N', 'U', 1.0, L, C), d, name(i, j), m, published)
        end
        push!(chains, (i, j) => cur)
    end
    return Dict{Tuple{Int,Int},Matrix{Float64}}(ij => fetch(t) for (ij, t) in chains)
end

"""
    tile_gemm!(d, mt, nt, kt, initA, initB, initC, ownerA, ownerB, ownerC; space=UInt64(1), probe=nothing) -> Dict{Tuple{Int,Int},Matrix}

`C += A * B` for `A` of `mt` x `kt` tiles, `B` of `kt` x `nt` tiles and `C` of `mt` x `nt` tiles. Each rank publishes the tiles of `A` and `B` it owns (version 0) and makes the chain of
`kt` updates for each tile of `C` it owns, pulling the tiles of `A` and `B` it needs, once each. `initX(i, j)` gives a tile on its owner; `C` is updated in place. Returns the tiles of `C`
of this rank (their final version is `kt`; `A` and `B` use `space` and `space + 1`).
"""
function tile_gemm!(d::TileDirectory, mt::Int, nt::Int, kt::Int, initA, initB, initC, ownerA, ownerB, ownerC; space::UInt64=UInt64(1), probe=nothing)
    me = d.me
    kern(kind, f, args...) = probe === nothing ? f(args...) : probe(kind, f, args...)
    nameA(i, k) = TileName(space, (k - 1) * mt + i)
    nameB(k, j) = TileName(space + 1, (j - 1) * kt + k)
    nameC(i, j) = TileName(space + 2, (j - 1) * mt + i)
    for k in 1:kt, i in 1:mt
        ownerA(i, k) == me && tile_publish!(d, nameA(i, k), 0, initA(i, k); readers=-1)
    end
    for j in 1:nt, k in 1:kt
        ownerB(k, j) == me && tile_publish!(d, nameB(k, j), 0, initB(k, j); readers=-1)
    end
    cacheA = Dict{Tuple{Int,Int},DataSlot}(); cacheB = Dict{Tuple{Int,Int},DataSlot}(); clock = ReentrantLock()
    slotA(i, k) = lock(clock) do; get!(cacheA, (i, k)) do; tile_fetch(d, nameA(i, k), 0; owner=ownerA(i, k)); end; end
    slotB(k, j) = lock(clock) do; get!(cacheB, (k, j)) do; tile_fetch(d, nameB(k, j), 0; owner=ownerB(k, j)); end; end
    chains = Pair{Tuple{Int,Int},Task}[]
    for j in 1:nt, i in 1:mt
        ownerC(i, j) == me || continue
        published = Threads.Atomic{Bool}(false)
        guard = _chain_guard(d, nameC(i, j), kt, published)
        cur = Threads.@spawn guard(() -> initC(i, j))
        gemm_step = (C, A, B) -> kern(4, BLAS.gemm!, 'N', 'N', 1.0, A, B, 1.0, C)
        for k in 1:kt-1
            cur = _chain_step(guard, cur, (slotA(i, k), slotB(k, j)), gemm_step)
        end
        cur = _chain_final(guard, cur, (slotA(i, kt), slotB(kt, j)), gemm_step, d, nameC(i, j), kt, published)
        push!(chains, (i, j) => cur)
    end
    return Dict{Tuple{Int,Int},Matrix{Float64}}(ij => fetch(t) for (ij, t) in chains)
end
