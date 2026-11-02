# spawn_datadeps regions on the TileFlow layer over MPI (included by cpu.jl, which has set up MPI acceleration and `comm`): unchanged library code,
# results equal the Datadeps path, the region counter proves the layer ran, failures are agreed on every rank, and the contract holds.
@testset "TileFlow regions (MPI)" begin
    old = Dagger.TILEFLOW_REGIONS[]
    me = MPI.Comm_rank(comm)
    function cmp(f)
        Dagger.tileflow_regions!(false); ref = f()
        Dagger.tileflow_regions!(true); n0 = Dagger.TILEFLOW_REGION_RUNS[]
        got = f(); runs = Dagger.TILEFLOW_REGION_RUNS[] - n0
        Dagger.tileflow_regions!(false)
        return ref, got, runs
    end
    try
        n, nb = 96, 16
        Random.seed!(11)
        for T in (Float64, ComplexF64), W in (LowerTriangular, UpperTriangular)
            X = rand(T, n, n); A = X * X' + n * I; A = (A + A') / 2
            chol = () -> begin
                D = zeros(Blocks(nb, nb), T, n, n); copyto!(D, A)
                r, info = LinearAlgebra._chol!(D, W)
                (collect(r.data), info)
            end
            ref, got, runs = cmp(chol)
            F = W === LowerTriangular ? tril : triu
            @test runs >= 1
            @test got[2] == 0 && ref[2] == 0
            @test norm(F(got[1]) - F(ref[1])) / norm(A) < 1e-10
        end
        # not positive definite: the same nonzero info on every rank, nobody hangs
        A = rand(64, 64); A = A + A'
        nonpd = () -> (D = zeros(Blocks(16, 16), Float64, 64, 64); copyto!(D, A); LinearAlgebra._chol!(D, LowerTriangular)[2])
        # N.B. the Datadeps path itself is not uniform here (rank 1 reported info = 0 where rank 0 reported 2, see BUGS B27): only the tile layer is checked
        ref, got, runs = cmp(nonpd)
        @test runs >= 1
        @test got > 0
        infos = MPI.Allgather(Int(got), comm)
        @test all(==(infos[1]), infos)
        # LU without pivoting and a matrix product
        A = rand(n, n) + n * I
        lunp = () -> (D = zeros(Blocks(nb, nb), Float64, n, n); copyto!(D, A); F = lu(D, NoPivot()); (collect(F.L), collect(F.U)))
        ref, got, runs = cmp(lunp)
        @test runs >= 1
        @test norm(got[1] * got[2] - A) / norm(A) < 1e-9
        Am = rand(64, 48); Bm = rand(48, 80)
        mm = () -> (Da = zeros(Blocks(16, 16), Float64, 64, 48); copyto!(Da, Am); Db = zeros(Blocks(16, 16), Float64, 48, 80); copyto!(Db, Bm); collect(Da * Db))
        ref, got, runs = cmp(mm)
        @test runs >= 1
        @test norm(got - Am * Bm) / norm(Am * Bm) < 1e-12
        # a non-finite input fails with DTaskFailedException on every rank
        Anan = rand(48, 48) + 48I; Anan[3, 3] = NaN
        Dagger.tileflow_regions!(true)
        D = zeros(Blocks(16, 16), Float64, 48, 48); copyto!(D, Anan)
        @test_throws Dagger.DTaskFailedException lu(D, NoPivot())
        Dagger.tileflow_regions!(false)
    finally
        Dagger.tileflow_regions!(old)
    end
end
