# DArray algorithms on the TileFlow layer (src/tileflow.jl).  Opt-in (JULIA_DAGGER_TILEFLOW=1 or `Dagger.tileflow_linalg!(true)`), and only for arrays it can
# handle exactly: dense square-tiled CPU matrices of Float32/Float64/ComplexF32/ComplexF64 whose tiles are plain `Matrix` chunks of this process (or of the
# MPI communicator). Anything else returns `nothing` and the caller takes the ordinary Datadeps path, so no functionality is lost.
# The tiles ARE the DArray's chunk arrays (no copy): the algorithm writes the result into the array in place, on the rank that owns the chunk.

const TILEFLOW_LINALG = Ref(get(ENV, "JULIA_DAGGER_TILEFLOW", "0") == "1")
"`tileflow_linalg!(on::Bool)`: let supported `DArray` factorizations run on the TileFlow layer."
tileflow_linalg!(on::Bool) = (TILEFLOW_LINALG[] = on)

struct TilePlan
    space::TileSpace
    nt::Int
    tr::TileTransport
end

# The decision must be the same on every rank: it only reads replicated metadata (types, owners, sizes), never array data.
function tileflow_plan(A::DArray{T,2}) where T
    TILEFLOW_LINALG[] || return nothing
    T <: Union{Float32,Float64,ComplexF32,ComplexF64} || return nothing
    mb, nb = A.partitioning.blocksize
    mb == nb || return nothing
    Ac = A.chunks
    mt, nt = size(Ac)
    mt == nt || return nothing
    tr = tile_transport(current_acceleration())
    np = tile_nranks(tr)
    np <= 64 || return nothing
    chunks = Matrix{Chunk}(undef, mt, nt)
    for I in CartesianIndices(Ac)
        c = Ac[I]
        c isa DTask && (c = fetch(c; raw=true))
        c isa Chunk || return nothing
        chunktype(c) <: Matrix{T} || return nothing
        o = tile_chunk_owner(c)
        (0 <= o < np) || return nothing
        tile_chunk_transport_ok(tr, c) || return nothing
        chunks[I] = c
    end
    me = tile_rank(tr)
    sizes = [size(A.subdomains[i, j]) for i in 1:mt, j in 1:nt]
    id(i, j) = (j - 1) * nt + i
    own = Vector{Int}(undef, nt * nt); tsz = Vector{Tuple{Int,Int}}(undef, nt * nt)
    for j in 1:nt, i in 1:mt
        own[id(i, j)] = tile_chunk_owner(chunks[i, j]); tsz[id(i, j)] = (sizes[i, j][1], sizes[i, j][2])
    end
    # init is called on the owner only: hand out the chunk's array itself
    init = k -> (i = (k - 1) % nt + 1; j = (k - 1) ÷ nt + 1; tile_chunk_array(chunks[i, j]))
    space = TileSpace(nt * nt, k -> own[k], k -> tsz[k], init; T, transport=tr)
    return TilePlan(space, nt, tr)
end

tile_id(nt, i, j) = (j - 1) * nt + i

function tile_potrf!(uplo, A, info::Threads.Atomic{Int})
    _, inf = LAPACK.potrf!(uplo, A)
    inf != 0 && Threads.atomic_max!(info, Int(inf))      # not thrown: a failure must not stop the transfers other ranks wait for
    return nothing
end

"""
    tileflow_chol!(A::DArray, uplo) -> Union{Nothing, Tuple}

Cholesky factorization of `A` in place on the TileFlow layer; `nothing` when `A` is not supported (the caller then uses Datadeps).
Returns `(factor wrapper, info)` like `LinearAlgebra._chol!`.
"""
function tileflow_chol!(A::DArray{T,2}, uplo::Char) where T
    timing = HIER_TIMING[]
    t0 = timing ? time_ns() : UInt64(0)
    plan = tileflow_plan(A)
    plan === nothing && return nothing
    t1 = timing ? time_ns() : UInt64(0)
    sp = plan.space; nt = plan.nt
    id(i, j) = tile_id(nt, i, j)
    zone = one(T); mzone = -one(T); rzone = one(real(T)); rmzone = -one(real(T))
    iscomplex = T <: Complex
    trans = iscomplex ? 'C' : 'T'
    info = Threads.Atomic{Int}(0)
    spawn = tile_spawn!
    if uplo == 'L'
        for k in 1:nt
            spawn(sp, A_ -> tile_potrf!('L', A_, info), TileRW(id(k, k)))
            for m in k+1:nt
                spawn(sp, (L, B) -> (BLAS.trsm!('R', 'L', trans, 'N', zone, L, B); nothing), TileIn(id(k, k)), TileRW(id(m, k)))
            end
            for n in k+1:nt
                if iscomplex
                    spawn(sp, (X, C) -> (BLAS.herk!('L', 'N', rmzone, X, rzone, C); nothing), TileIn(id(n, k)), TileRW(id(n, n)))
                else
                    spawn(sp, (X, C) -> (BLAS.syrk!('L', 'N', rmzone, X, rzone, C); nothing), TileIn(id(n, k)), TileRW(id(n, n)))
                end
                for m in n+1:nt
                    spawn(sp, (X, Y, C) -> (BLAS.gemm!('N', trans, mzone, X, Y, zone, C); nothing),
                          TileIn(id(m, k)), TileIn(id(n, k)), TileRW(id(m, n)))
                end
            end
        end
    else
        for k in 1:nt
            spawn(sp, A_ -> tile_potrf!('U', A_, info), TileRW(id(k, k)))
            for n in k+1:nt
                spawn(sp, (U, B) -> (BLAS.trsm!('L', 'U', trans, 'N', zone, U, B); nothing), TileIn(id(k, k)), TileRW(id(k, n)))
            end
            for m in k+1:nt
                if iscomplex
                    spawn(sp, (X, C) -> (BLAS.herk!('U', 'C', rmzone, X, rzone, C); nothing), TileIn(id(k, m)), TileRW(id(m, m)))
                else
                    spawn(sp, (X, C) -> (BLAS.syrk!('U', 'T', rmzone, X, rzone, C); nothing), TileIn(id(k, m)), TileRW(id(m, m)))
                end
                for n in m+1:nt
                    spawn(sp, (X, Y, C) -> (BLAS.gemm!(trans, 'N', mzone, X, Y, zone, C); nothing),
                          TileIn(id(k, m)), TileIn(id(k, n)), TileRW(id(m, n)))
                end
            end
        end
    end
    t2 = timing ? time_ns() : UInt64(0)
    tile_sync!(sp)
    inf = tile_allreduce_max(plan.tr, info[])
    if timing
        t3 = time_ns()
        @info "tileflow_chol!: plan $(round((t1 - t0) / 1e6; digits=1)) ms, walk $(round((t2 - t1) / 1e6; digits=1)) ms, sync (run) $(round((t3 - t2) / 1e6; digits=1)) ms, total $(round((t3 - t0) / 1e6; digits=1)) ms; $(tile_stats(sp))"
    end
    return (uplo == 'L' ? LowerTriangular(A) : UpperTriangular(A)), inf
end
