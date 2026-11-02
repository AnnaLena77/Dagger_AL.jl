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
    mine = Dagger.tile_cholesky!(d, nt, tile, owner; space=UInt64(77))
    @test all(ij -> owner(ij...) == me, keys(mine))
    @test length(mine) == count(ij -> owner(ij...) == me, [(i, j) for j in 1:nt for i in j:nt])
    all_tiles = Dagger.allgather_value(mine; comm)         # cooperative: the directory service must keep answering the other ranks
    L = zeros(n, n)
    for res in all_tiles, ((i, j), T) in res
        L[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb] .= T
    end
    L = LowerTriangular(L)
    @test norm(L * L' - A) / norm(A) < 1e-12
    @test norm(L - cholesky(Symmetric(A, :L)).L) / norm(L) < 1e-10
    @test isempty(d.errors)
end
