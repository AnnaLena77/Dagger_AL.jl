using Dagger, Test, Random, LinearAlgebra
using Dagger: TileDirectory, tile_cholesky!

include("tiledir_defs.jl")

# the matrix as tiles; every rank gets the tiles it owns
function chol_case(ids, nt, nb, owner; seed=1)
    Random.seed!(seed)
    n = nt * nb
    X = rand(n, n); A = X * X' + n * I
    net, ds, tasks = dir_cluster(ids, ids)
    tile(i, j) = copy(A[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    results = Dict{Int,Any}()
    rlock = ReentrantLock()
    runs = map(ids) do r
        Threads.@spawn begin
            res = tile_cholesky!(ds[r], nt, tile, owner)
            lock(() -> results[r] = res, rlock)
        end
    end
    foreach(fetch, runs)
    L = zeros(n, n)
    seen = 0
    for (r, res) in results, ((i, j), T) in res
        owner(i, j) == r || error("rank $r returned a tile it does not own")
        L[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb] .= T
        seen += 1
    end
    L = LowerTriangular(L)
    ok = norm(L * L' - A) / norm(A) < 1e-12 && norm(L - cholesky(Symmetric(A, :L)).L) / norm(L) < 1e-10
    errs = all(d -> isempty(d.errors), values(ds))
    dir_close(net, tasks)
    return ok, seen == nt * (nt + 1) ÷ 2, errs
end

@testset "Tiled Cholesky on exchangeable tiles" begin
    @testset "one rank" begin
        @test all(chol_case([0], 4, 32, (i, j) -> 0))
    end
    @testset "four ranks, cyclic 2 x 2" begin
        ids = [0, 1, 2, 3]
        @test all(chol_case(ids, 6, 32, (i, j) -> ids[mod(i - 1, 2) * 2 + mod(j - 1, 2) + 1]))
    end
    @testset "rank identifiers are not positions: three ranks 10, 20, 30" begin
        ids = [10, 20, 30]
        @test all(chol_case(ids, 7, 24, (i, j) -> ids[mod(i + 2j, 3) + 1]; seed=3))
    end
    @testset "all tiles on one of several ranks, the others only read" begin
        ids = [0, 1, 2]
        @test all(chol_case(ids, 5, 32, (i, j) -> 1))
    end
end
