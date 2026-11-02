# spawn_datadeps regions executed on the TileFlow layer (JULIA_DAGGER_TILEFLOW_REGIONS=1): the library code is unchanged, results must equal the
# Datadeps path, and the counter proves the layer really ran (the fallback gives the same numbers).
using Dagger, LinearAlgebra, Random, Test

# Only in-place (`!`) functions are lowered (`Dagger._region_inplace`): any other task may produce a block or value that the rest of Dagger needs as a Chunk.
tf_sum!(x) = sum(x)
tf_boom!(x) = error("boom")
tf_noop!(x, y) = nothing

function tf_region_compare(f)
    old = Dagger.TILEFLOW_REGIONS[]
    try
        Dagger.tileflow_regions!(false)
        ref = f()
        Dagger.tileflow_regions!(true)
        n0 = Dagger.TILEFLOW_REGION_RUNS[]
        got = f()
        return ref, got, Dagger.TILEFLOW_REGION_RUNS[] - n0
    finally
        Dagger.tileflow_regions!(old)
    end
end

@testset "spawn_datadeps regions on the TileFlow layer" begin
    n, nb = 96, 16
    @testset "Cholesky ($T, $W)" for T in (Float64, ComplexF64), W in (LowerTriangular, UpperTriangular)
        X = rand(T, n, n); A = X * X' + n * I; A = (A + A') / 2
        chol = () -> begin
            D = distribute(copy(A), Blocks(nb, nb))
            r, info = LinearAlgebra._chol!(D, W)
            (collect(r.data), info)
        end
        ref, got, runs = tf_region_compare(chol)
        F = W === LowerTriangular ? tril : triu
        @test runs >= 1
        @test got[2] == 0 && ref[2] == 0
        @test norm(F(got[1]) - F(ref[1])) / norm(A) < 1e-10
    end
    @testset "Cholesky of a matrix that is not positive definite reports the same info" begin
        A = rand(64, 64); A = A + A'
        nonpd = () -> (D = distribute(copy(A), Blocks(16, 16)); LinearAlgebra._chol!(D, LowerTriangular)[2])
        ref, got, runs = tf_region_compare(nonpd)
        @test runs >= 1
        @test ref > 0 && got > 0
    end
    @testset "LU without pivoting ($T)" for T in (Float64, ComplexF64)
        A = rand(T, n, n) + n * I
        lunp = () -> begin
            D = distribute(copy(A), Blocks(nb, nb))
            F = lu(D, NoPivot())
            (collect(F.L), collect(F.U))
        end
        ref, got, runs = tf_region_compare(lunp)
        @test runs >= 1
        @test norm(got[1] * got[2] - A) / norm(A) < 1e-9
        @test norm(got[1] - ref[1]) / norm(A) < 1e-9
    end
    @testset "matrix product ($T)" for T in (Float64, Float32)
        A = rand(T, 64, 48); B = rand(T, 48, 80)
        ref, got, runs = tf_region_compare(() -> collect(distribute(A, Blocks(16, 16)) * distribute(B, Blocks(16, 16))))
        @test runs >= 1
        @test norm(got - A * B) / norm(A * B) < (T === Float32 ? 1e-5 : 1e-12)
    end
    @testset "the decision does not depend on timing (freshly created arrays, used at once)" begin
        # A DArray's chunks are created asynchronously; a region must be lowered whether or not they are finished (ranks could otherwise disagree)
        old = Dagger.TILEFLOW_REGIONS[]
        try
            Dagger.tileflow_regions!(true)
            n0 = Dagger.TILEFLOW_REGION_RUNS[]
            for _ in 1:10
                X = rand(64, 64); A = X * X' + 64I
                D = distribute(copy(A), Blocks(16, 16))
                _, info = LinearAlgebra._chol!(D, LowerTriangular)
                @test info == 0
            end
            @test Dagger.TILEFLOW_REGION_RUNS[] == n0 + 10
        finally
            Dagger.tileflow_regions!(old)
        end
    end
    @testset "non-finite input still fails like Datadeps" begin
        A = rand(48, 48) + 48I; A[3, 3] = NaN
        old = Dagger.TILEFLOW_REGIONS[]
        try
            for on in (false, true)
                Dagger.tileflow_regions!(on)
                @test_throws Dagger.DTaskFailedException lu(distribute(copy(A), Blocks(16, 16)), NoPivot())
            end
        finally
            Dagger.tileflow_regions!(old)
        end
    end
    @testset "task values, use of results, failures, fallback" begin
        old = Dagger.TILEFLOW_REGIONS[]
        try
            Dagger.tileflow_regions!(true)
            D = distribute(fill(2.0, 32, 32), Blocks(16, 16)); Ac = D.chunks
            n0 = Dagger.TILEFLOW_REGION_RUNS[]
            t = nothing
            Dagger.spawn_datadeps() do
                t = Dagger.@spawn tf_sum!(Dagger.In(Ac[1, 1]))
            end
            @test Dagger.TILEFLOW_REGION_RUNS[] == n0 + 1
            @test fetch(t) == 2.0 * 16 * 16                                     # fetch gives the value
            # a task of a lowered region is an ordinary argument of a later task: its value is passed
            @test fetch(Dagger.@spawn (x -> x + 1)(t)) == 2.0 * 16 * 16 + 1
            # a failing task raises DTaskFailedException from the region
            @test_throws Dagger.DTaskFailedException Dagger.spawn_datadeps() do
                Dagger.@spawn tf_boom!(Dagger.InOut(Ac[1, 1]))
            end
            # an unsupported argument (a Ref) sends the whole region to the ordinary planner
            n1 = Dagger.TILEFLOW_REGION_RUNS[]
            r = Ref(0)
            Dagger.spawn_datadeps() do
                Dagger.@spawn tf_noop!(Dagger.In(Ac[1, 1]), r)
            end
            @test Dagger.TILEFLOW_REGION_RUNS[] == n1
            # a task that is not an in-place function may produce a block or a value for others: it is never lowered, and its result stays an ordinary Chunk
            n2 = Dagger.TILEFLOW_REGION_RUNS[]
            Dagger.spawn_datadeps() do
                Dagger.@spawn sum(Dagger.In(Ac[1, 1]))
            end
            @test Dagger.TILEFLOW_REGION_RUNS[] == n2
            E = distribute(fill(1.0, 32, 32), Blocks(16, 16))
            @test Dagger.TILEFLOW_REGION_RUNS[] == n2
            @test all(c -> fetch(c; raw=true) isa Dagger.Chunk, E.chunks)
        finally
            Dagger.tileflow_regions!(old)
        end
    end
end
