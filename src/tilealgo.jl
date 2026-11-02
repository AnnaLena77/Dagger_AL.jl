# Tile algorithms on exchangeable tiles (src/tilenames.jl, tiledir.jl, tilestore.jl): the tiled Cholesky factorization, LU without pivoting and the matrix product, in the style of HPX.
#
# Every rank creates tasks only for the tiles it owns. A task produces a new version of its tile and needs, besides its own tile (which stays on the rank and is updated in
# place from task to task), the *final* versions of some tiles of earlier columns, which it asks for as pull futures: nobody has a plan that all ranks share, nobody has to know
# which rank runs which other task, and the set of ranks enters only through `owner(i, j)` and the directory. The requests go out when the tasks are created, not when they start:
# the owner holds a request until it has published the version, so the answer follows the publication at once (one hop on the critical path, like a push). Only the final version
# of a tile is published (the intermediate versions never leave the rank); a rank asks for a final version once however many of its tasks use it. A received array goes back to the
# pool of the directory when the last of those tasks is done with it (`slot_ref!` / `slot_unref!`), so the next receive finds memory that has been used before.

# One step of the chain of a tile: wait for the tile's previous value and for the slots of the final tiles it needs, then run `f(C, values...)` on it in place. The previous value and the
# slots are arguments, not captured variables, so a loop that reassigns them cannot make the task wait for itself. Every slot gets one more user before the task starts and the task
# gives it up when it is done (also when it fails).
function _chain_step(guard, d, prev, slots, f)
    foreach(slot_ref!, slots)
    return Threads.@spawn guard() do
        try
            C = fetch(prev); vals = map(fetch, slots)
            f(C, vals...)
            C
        finally
            foreach(s -> slot_unref!(d, s), slots)
        end
    end
end
# The last step of the chain: the same, and then the final version of the tile is published (a failure is published instead by `guard`).
function _chain_final(guard, d, prev, slots, f, nm, ver, published)
    foreach(slot_ref!, slots)
    return Threads.@spawn guard() do
        try
            C = fetch(prev); vals = map(fetch, slots)
            f(C, vals...)
            Threads.atomic_xchg!(published, true)
            tile_publish!(d, nm, ver, C; readers=-1)
            C
        finally
            foreach(s -> slot_unref!(d, s), slots)
        end
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

# The final tiles other ranks (or this one) publish and this rank asks for, once each: `_get_slot!` makes the slot; the creator holds one reference until it has handed it to
# every task (`finish_slots!`), so a task that finishes early cannot give the array back while later tasks still have to be created.
struct _SlotCache
    d::TileDirectory
    slots::Dict{Tuple{Int,Int},DataSlot}
    lock::ReentrantLock
end
_SlotCache(d::TileDirectory) = _SlotCache(d, Dict{Tuple{Int,Int},DataSlot}(), ReentrantLock())
_get_slot!(f, c::_SlotCache, ij) = lock(c.lock) do; get!(f, c.slots, ij); end
function finish_slots!(c::_SlotCache)
    for s in values(c.slots); slot_unref!(c.d, s); end
    return nothing
end

"""
    tile_cholesky!(d, nt, init, owner; space=UInt64(1), probe=nothing) -> Dict{Tuple{Int,Int},Matrix}

Factorize (lower, `A = L L'`) the symmetric positive definite matrix of `nt` x `nt` tiles on the ranks of the directory `d`. `init(i, j)` returns the tile `(i, j)`
of the lower triangle (called only on its owner, and the result is changed in place), `owner(i, j)` is the identifier of the rank that owns it. Returns the final tiles
this rank owns after all its tasks have finished. Every rank must call it with the same `nt`, `owner` and `space`. Tile `(m, n)`, `m >= n`, is updated at the steps `k = 1..n-1`
(syrk if `m == n`, gemm otherwise) and then finished (potrf, trsm); its final version is `n`.
"""
function tile_cholesky!(d::TileDirectory, nt::Int, init, owner; space::UInt64=UInt64(1), prefetch::Bool=true, probe=nothing)
    me = d.me
    kern(kind, f, args...) = probe === nothing ? f(args...) : probe(kind, f, args...)     # `probe(kind, f, args...)` may time the kernel (benchmarks)
    name(i, j) = TileName(space, (j - 1) * nt + i)
    cache = _SlotCache(d)
    final_tile(i, j) = _get_slot!(cache, (i, j)) do
        tile_fetch(d, name(i, j), j; owner=owner(i, j))          # the layout says who owns it: no directory lookup
    end
    potrf_step = C -> (kern(1, LAPACK.potrf!, 'L', C)[2] == 0 || throw(LinearAlgebra.PosDefException(1)); nothing)
    chains = Pair{Tuple{Int,Int},Task}[]
    for j in 1:nt, i in j:nt
        owner(i, j) == me || continue
        published = Threads.Atomic{Bool}(false)
        guard = _chain_guard(d, name(i, j), j, published)
        cur = Threads.@spawn guard(() -> init(i, j))
        for k in 1:j-1
            if i == j
                cur = _chain_step(guard, d, cur, (final_tile(j, k),), (C, A) -> kern(3, BLAS.syrk!, 'L', 'N', -1.0, A, 1.0, C))
            else
                cur = _chain_step(guard, d, cur, (final_tile(i, k), final_tile(j, k)), (C, A, B) -> kern(4, BLAS.gemm!, 'N', 'T', -1.0, A, B, 1.0, C))
            end
        end
        if i == j
            cur = _chain_final(guard, d, cur, (), potrf_step, name(i, j), j, published)
        else
            cur = _chain_final(guard, d, cur, (final_tile(j, j),), (C, L) -> kern(2, BLAS.trsm!, 'R', 'L', 'T', 'N', 1.0, L, C), name(i, j), j, published)
        end
        push!(chains, (i, j) => cur)
    end
    finish_slots!(cache)
    return Dict{Tuple{Int,Int},Matrix{Float64}}(ij => fetch(t) for (ij, t) in chains)
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
    cache = _SlotCache(d)
    final_tile(i, j) = _get_slot!(cache, (i, j)) do
        tile_fetch(d, name(i, j), min(i, j); owner=owner(i, j))
    end
    chains = Pair{Tuple{Int,Int},Task}[]
    for j in 1:nt, i in 1:nt
        owner(i, j) == me || continue
        m = min(i, j)
        published = Threads.Atomic{Bool}(false)
        guard = _chain_guard(d, name(i, j), m, published)
        cur = Threads.@spawn guard(() -> init(i, j))
        for k in 1:m-1
            cur = _chain_step(guard, d, cur, (final_tile(i, k), final_tile(k, j)), (C, A, B) -> kern(4, BLAS.gemm!, 'N', 'N', -1.0, A, B, 1.0, C))
        end
        if i == j
            cur = _chain_final(guard, d, cur, (), C -> (kern(1, LinearAlgebra.lu!, C, LinearAlgebra.NoPivot()); nothing), name(i, j), m, published)
        elseif i > j
            cur = _chain_final(guard, d, cur, (final_tile(j, j),), (C, U) -> kern(2, BLAS.trsm!, 'R', 'U', 'N', 'N', 1.0, U, C), name(i, j), m, published)
        else
            cur = _chain_final(guard, d, cur, (final_tile(i, i),), (C, L) -> kern(2, BLAS.trsm!, 'L', 'L', 'N', 'U', 1.0, L, C), name(i, j), m, published)
        end
        push!(chains, (i, j) => cur)
    end
    finish_slots!(cache)
    return Dict{Tuple{Int,Int},Matrix{Float64}}(ij => fetch(t) for (ij, t) in chains)
end

"""
    tile_gemm!(d, mt, nt, kt, initA, initB, initC, ownerA, ownerB, ownerC; space=UInt64(1), probe=nothing) -> Dict{Tuple{Int,Int},Matrix}

`C += A * B` for `A` of `mt` x `kt` tiles, `B` of `kt` x `nt` tiles and `C` of `mt` x `nt` tiles. Each rank publishes the tiles of `A` and `B` it owns (version 0) and makes the chain of
`kt` updates for each tile of `C` it owns, pulling the tiles of `A` and `B` it needs, once each. `initX(i, j)` gives a tile on its owner; `C` is updated in place. Returns the tiles of `C`
of this rank (their final version is `kt`; `A`, `B` and `C` use `space`, `space + 1` and `space + 2`).
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
    cacheA = _SlotCache(d); cacheB = _SlotCache(d)
    slotA(i, k) = _get_slot!(cacheA, (i, k)) do; tile_fetch(d, nameA(i, k), 0; owner=ownerA(i, k)); end
    slotB(k, j) = _get_slot!(cacheB, (k, j)) do; tile_fetch(d, nameB(k, j), 0; owner=ownerB(k, j)); end
    gemm_step = (C, A, B) -> kern(4, BLAS.gemm!, 'N', 'N', 1.0, A, B, 1.0, C)
    chains = Pair{Tuple{Int,Int},Task}[]
    for j in 1:nt, i in 1:mt
        ownerC(i, j) == me || continue
        published = Threads.Atomic{Bool}(false)
        guard = _chain_guard(d, nameC(i, j), kt, published)
        cur = Threads.@spawn guard(() -> initC(i, j))
        for k in 1:kt-1
            cur = _chain_step(guard, d, cur, (slotA(i, k), slotB(k, j)), gemm_step)
        end
        cur = _chain_final(guard, d, cur, (slotA(i, kt), slotB(kt, j)), gemm_step, nameC(i, j), kt, published)
        push!(chains, (i, j) => cur)
    end
    finish_slots!(cacheA); finish_slots!(cacheB)
    return Dict{Tuple{Int,Int},Matrix{Float64}}(ij => fetch(t) for (ij, t) in chains)
end
