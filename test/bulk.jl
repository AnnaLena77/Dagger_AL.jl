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
    @testset "per-worker counts" begin
        counts = Dagger.dynamic_for(i -> nothing, 1000; workers=3, chunk=7)
        @test length(counts) == 3 && sum(counts) == 1000
        @test sum(fetch.(Dagger.dynamic_for_async(i -> nothing, 55; workers=2))) == 55
        @test sum(Dagger.dynamic_for(i -> nothing, 0)) == 0     # no iterations: nothing taken
    end
    @testset "async form returns the workers" begin
        c = Threads.Atomic{Int}(0)
        tasks = Dagger.dynamic_for_async(i -> Threads.atomic_add!(c, 1), 200; workers=2)
        @test length(tasks) == 2
        foreach(fetch, tasks)
        @test c[] == 200
    end
    @testset "task arguments: the loop waits for DTasks and chunks" begin
        # `f(i, args...)`; the first argument is produced by a slow task, so the loop can only start after it
        slow_vec(n) = (sleep(0.3); collect(1:n))
        scale(i, v, w, out) = (out[i] = v[i] * w; nothing)
        n = 400
        out = zeros(Int, n)
        v = Dagger.@spawn slow_vec(n)                      # a DTask argument
        w = Dagger.tochunk(3)                              # a Chunk argument
        t0 = time()
        tasks = Dagger.dynamic_for_async(scale, n, v, w, out; workers=2)
        @test time() - t0 < 0.25                           # returned at once: the caller did not wait for `v`
        foreach(fetch, tasks)
        @test out == 3 .* (1:n)
        out2 = zeros(Int, n)
        Dagger.dynamic_for(scale, n, v, 5, out2)           # blocking form, plain value argument
        @test out2 == 5 .* (1:n)
    end
    @testset "called from inside tasks, with every thread busy" begin
        # each outer task holds a processor while its loop runs: the loop must still finish
        c = Threads.Atomic{Int}(0)
        outer(c) = (Dagger.dynamic_for(i -> Threads.atomic_add!(c, 1), 50; workers=2); nothing)
        ts = [Dagger.@spawn outer(c) for _ in 1:max(2, Threads.nthreads())]
        @test all(t -> fetch(t) === nothing, ts)
        @test c[] == 50 * length(ts)
    end
end
