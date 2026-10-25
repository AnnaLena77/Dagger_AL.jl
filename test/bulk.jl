@testset "Dynamic loop" begin
    @testset "every index exactly once" begin
        n = 5000
        hits = zeros(Int, n); acc = Threads.Atomic{Int}(0)
        Dagger.dynamic_for(i -> (hits[i] += 1; Threads.atomic_add!(acc, i)), n)
        @test all(==(1), hits)
        @test acc[] == n * (n + 1) ÷ 2
    end
    @testset "edge cases" begin
        c = Threads.Atomic{Int}(0)
        Dagger.dynamic_for(i -> Threads.atomic_add!(c, 1), 3)                   # fewer iterations than workers
        Dagger.dynamic_for(i -> Threads.atomic_add!(c, 1), 1; workers=8)
        Dagger.dynamic_for(i -> Threads.atomic_add!(c, 1), 10; chunk=100)       # chunk larger than the loop
        Dagger.dynamic_for(i -> Threads.atomic_add!(c, 1), 0)
        @test c[] == 14
        @test_throws ArgumentError Dagger.dynamic_for(identity, 10; chunk=0)
    end
    @testset "failure propagates" begin
        @test_throws Exception Dagger.dynamic_for(i -> i == 7 ? error("boom") : nothing, 100)
    end
    @testset "async form returns the workers" begin
        c = Threads.Atomic{Int}(0)
        tasks = Dagger.dynamic_for_async(i -> Threads.atomic_add!(c, 1), 200; workers=2)
        @test length(tasks) == 2
        foreach(fetch, tasks)
        @test c[] == 200
    end
end
