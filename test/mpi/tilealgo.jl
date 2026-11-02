# The tiled Cholesky on exchangeable tiles over MPI (src/tilealgo.jl): included by cpu.jl, which has set up MPI acceleration, `comm`, `nranks`.
@testset "Tiled Cholesky on exchangeable tiles (MPI)" begin
    me = MPI.Comm_rank(comm)
    d = Dagger.tile_directory()
    nt, nb = 6, 32
    n = nt * nb
    Random.seed!(5)
    X = rand(n, n); A = X * X' + n * I                     # the same matrix on every rank (same seed)
    pr = max(1, floor(Int, sqrt(nranks))); while nranks % pr != 0; pr -= 1; end; pc = nranks ÷ pr
    owner(i, j) = mod(i - 1, pr) * pc + mod(j - 1, pc)      # a 2D cyclic grid of the ranks
    tile(i, j) = copy(A[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    # Received arrays are recycled: the first factorization fills the pool, the second one receives into the same arrays. With TILE_POISON a released array is filled with NaN, so a task that
    # still read it (a release too early) makes the result wrong instead of silently using another tile's data.
    Dagger.TILE_POISON[] = true
    hits0 = Dagger.TILE_POOL_HITS[]
    for (rep, space) in enumerate((UInt64(77), UInt64(177)))
        mine = Dagger.tile_cholesky!(d, nt, tile, owner; space)
        @test all(ij -> owner(ij...) == me, keys(mine))
        @test length(mine) == count(ij -> owner(ij...) == me, [(i, j) for j in 1:nt for i in j:nt])
        all_tiles = Dagger.allgather_value(mine; comm)         # cooperative: the directory service must keep answering the other ranks
        L = zeros(n, n)
        for res in all_tiles, ((i, j), T) in res
            L[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb] .= T
        end
        L = LowerTriangular(L)
        @test all(isfinite, L)
        @test norm(L * L' - A) / norm(A) < 1e-12
        @test norm(L - cholesky(Symmetric(A, :L)).L) / norm(L) < 1e-10
        Dagger.barrier(; comm)
    end
    nranks > 1 && @test sum(Dagger.allgather_value(Dagger.TILE_POOL_HITS[] - hits0; comm)) > 0     # the pool really was used
    @test isempty(d.errors)

    # a matrix that is not positive definite: every rank gets an error (named), nobody waits for a tile that will never be published
    Y = rand(n, n); B = Y + Y'
    tileb(i, j) = copy(B[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    run = Threads.@spawn Dagger.tile_cholesky!(d, nt, tileb, owner; space=UInt64(78))
    @test timedwait(() -> istaskdone(run), 120.0; pollint=0.01) === :ok
    err = try; fetch(run); nothing; catch e; e; end
    @test err !== nothing
    Dagger.barrier(; comm)
    @test isempty(d.errors)

    # LU without pivoting
    Z = rand(n, n) + n * I
    tilez(i, j) = copy(Z[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    lures = Dagger.allgather_value(Dagger.tile_lu_nopiv!(d, nt, tilez, owner; space=UInt64(79)); comm)
    F = zeros(n, n); for res in lures, ((i, j), T) in res; F[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb] .= T; end
    @test norm((tril(F, -1) + I) * triu(F) - Z) / norm(Z) < 1e-10
    # matrix product C += A * B on three different layouts
    mt, ntc, kt = 4, 3, 5
    Am = rand(mt * nb, kt * nb); Bm = rand(kt * nb, ntc * nb); Cm = rand(mt * nb, ntc * nb)
    blk(M, i, j) = copy(M[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    gres = Dagger.allgather_value(Dagger.tile_gemm!(d, mt, ntc, kt, (i, k) -> blk(Am, i, k), (k, j) -> blk(Bm, k, j), (i, j) -> blk(Cm, i, j),
                                                    (i, k) -> mod(i + k, nranks), (k, j) -> mod(k + 2j, nranks), (i, j) -> mod(i + j, nranks); space=UInt64(80)); comm)
    G = zeros(mt * nb, ntc * nb); for res in gres, ((i, j), T) in res; G[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb] .= T; end
    @test norm(G - (Cm + Am * Bm)) / norm(Cm + Am * Bm) < 1e-12
    Dagger.barrier(; comm)
    @test isempty(d.errors)
    Dagger.TILE_POISON[] = false
end
