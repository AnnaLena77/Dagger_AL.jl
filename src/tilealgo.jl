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
function tile_cholesky!(d::TileDirectory, nt::Int, init, owner; space::UInt64=UInt64(1), prefetch::Bool=true)
    me = d.me
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
                        BLAS.syrk!('L', 'N', -1.0, A, 1.0, C)
                        C
                    end
                end
            else
                cur = let prev = cur, sa = early(i, k), sb = early(j, k)
                    Threads.@spawn guard() do
                        C = fetch(prev); A = fetch(late(sa, i, k)); B = fetch(late(sb, j, k))
                        BLAS.gemm!('N', 'T', -1.0, A, B, 1.0, C)
                        C
                    end
                end
            end
        end
        if i == j
            cur = let prev = cur
                Threads.@spawn guard() do
                    C = fetch(prev)
                    _, info = LAPACK.potrf!('L', C)
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
                    BLAS.trsm!('R', 'L', 'T', 'N', 1.0, L, C)
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
