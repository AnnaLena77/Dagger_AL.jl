using Dagger, Test, Random, LinearAlgebra
using Dagger: TileDirectory, tile_cholesky!, tile_lu_nopiv!, tile_gemm!

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

    @testset "a failing kernel fails every rank instead of hanging them" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        nt, nb = 5, 16
        Random.seed!(2)
        X = rand(nt * nb, nt * nb); A = X + X'                             # symmetric but not positive definite
        owner(i, j) = ids[mod(i - 1, 2) * 2 + mod(j - 1, 2) + 1]
        tile(i, j) = copy(A[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
        runs = map(ids) do r; Threads.@spawn tile_cholesky!(ds[r], nt, tile, owner); end
        @test timedwait(() -> all(istaskdone, runs), 90.0; pollint=0.05) === :ok       # nobody waits for a tile that will never come
        errs = [try; fetch(t); nothing; catch e; e; end for t in runs if istaskdone(t)]
        @test length(errs) == 4 && all(e -> e !== nothing, errs)
        @test any(e -> occursin("PosDefException", sprint(showerror, e)), errs)         # and the cause is named
        dir_close(net, tasks)
    end
end

# the tiles of a result gathered from all ranks into one matrix
function assemble(results, nb, mt, nt, owner)
    F = zeros(mt * nb, nt * nb)
    seen = 0
    for (r, res) in results, ((i, j), T) in res
        owner(i, j) == r || error("rank $r returned a tile it does not own")
        F[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb] .= T; seen += 1
    end
    return F, seen == mt * nt
end

function lu_case(ids, nt, nb, owner; seed=1)
    Random.seed!(seed)
    n = nt * nb
    A = rand(n, n) + n * I
    net, ds, tasks = dir_cluster(ids, ids)
    tile(i, j) = copy(A[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    results = Dict{Int,Any}(); rlock = ReentrantLock()
    foreach(fetch, map(ids) do r
        Threads.@spawn (res = tile_lu_nopiv!(ds[r], nt, tile, owner); lock(() -> results[r] = res, rlock))
    end)
    F, complete = assemble(results, nb, nt, nt, owner)
    L = tril(F, -1) + I; U = triu(F)
    ok = norm(L * U - A) / norm(A) < 1e-10
    errs = all(d -> isempty(d.errors), values(ds))
    dir_close(net, tasks)
    return ok, complete, errs
end

function gemm_case(ids, mt, nt, kt, nb, oA, oB, oC; seed=1)
    Random.seed!(seed)
    A = rand(mt * nb, kt * nb); B = rand(kt * nb, nt * nb); C0 = rand(mt * nb, nt * nb)
    net, ds, tasks = dir_cluster(ids, ids)
    blk(M, i, j) = copy(M[(i-1)*nb+1:i*nb, (j-1)*nb+1:j*nb])
    results = Dict{Int,Any}(); rlock = ReentrantLock()
    foreach(fetch, map(ids) do r
        Threads.@spawn (res = tile_gemm!(ds[r], mt, nt, kt, (i, k) -> blk(A, i, k), (k, j) -> blk(B, k, j), (i, j) -> blk(C0, i, j), oA, oB, oC); lock(() -> results[r] = res, rlock))
    end)
    F, complete = assemble(results, nb, mt, nt, oC)
    ok = norm(F - (C0 + A * B)) / norm(C0 + A * B) < 1e-12
    errs = all(d -> isempty(d.errors), values(ds))
    dir_close(net, tasks)
    return ok, complete, errs
end

@testset "LU without pivoting and the matrix product on exchangeable tiles" begin
    @testset "LU, one rank" begin
        @test all(lu_case([0], 4, 24, (i, j) -> 0))
    end
    @testset "LU, four ranks (2 x 2), and three ranks 10/20/30" begin
        ids = [0, 1, 2, 3]
        @test all(lu_case(ids, 6, 24, (i, j) -> ids[mod(i - 1, 2) * 2 + mod(j - 1, 2) + 1]))
        ids = [10, 20, 30]
        @test all(lu_case(ids, 5, 24, (i, j) -> ids[mod(i + 2j, 3) + 1]; seed=2))
    end
    @testset "matrix product, one rank and four ranks with three different layouts" begin
        @test all(gemm_case([0], 3, 2, 4, 16, (i, k) -> 0, (k, j) -> 0, (i, j) -> 0))
        ids = [0, 1, 2, 3]
        @test all(gemm_case(ids, 4, 3, 5, 16, (i, k) -> ids[mod(i + k, 4) + 1], (k, j) -> ids[mod(k + 2j, 4) + 1], (i, j) -> ids[mod(i - 1, 2) * 2 + mod(j - 1, 2) + 1]; seed=4))
    end
    @testset "a zero pivot fails every rank instead of hanging them" begin
        ids = [0, 1, 2]
        net, ds, tasks = dir_cluster(ids, ids)
        nt, nb = 4, 16
        owner(i, j) = ids[mod(i + j, 3) + 1]
        tile(i, j) = i == 1 && j == 1 ? zeros(nb, nb) : rand(nb, nb)               # the first pivot block is singular
        runs = map(ids) do r; Threads.@spawn tile_lu_nopiv!(ds[r], nt, tile, owner); end
        @test timedwait(() -> all(istaskdone, runs), 90.0; pollint=0.05) === :ok
        @test all(istaskfailed, runs)
        dir_close(net, tasks)
    end
end

