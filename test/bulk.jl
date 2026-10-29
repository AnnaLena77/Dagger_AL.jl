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
        # `f(i, args...)`; the first argument is produced by a task that waits for `gate`, so no worker can start before it
        gate = Base.Event()
        hold_range(g, n) = (wait(g); collect(1:n))
        scale(i, v, w, out) = (out[i] = v[i] * w; nothing)
        n = 400
        out = zeros(Int, n)
        v = Dagger.@spawn hold_range(gate, n)               # a DTask argument
        w = Dagger.tochunk(3)                               # a Chunk argument
        tasks = Dagger.dynamic_for_async(scale, n, v, w, out; workers=2)   # returns without waiting for `v`
        sleep(0.3)
        @test !any(istaskdone, tasks) && all(==(0), out)    # nothing ran: the workers are waiting for `v`
        notify(gate)
        foreach(fetch, tasks)
        @test out == 3 .* (1:n)
        out2 = zeros(Int, n)
        Dagger.dynamic_for(scale, n, v, 5, out2)            # blocking form, plain value argument
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
    @testset "a vector of scopes: one per worker" begin
        n = 40
        out = zeros(Int, n)
        sc = [Dagger.scope(worker=myid()), Dagger.scope(worker=myid())]
        Dagger.dynamic_for((i, o) -> (o[i] = i; nothing), n, out; workers=3, scope=sc)   # cycled over the workers
        @test out == collect(1:n)
        @test Dagger.dynamic_reduce((i, v) -> v[i], +, n, collect(1:n); workers=2, scope=sc) == sum(1:n)
    end
    @testset "dynamic_reduce: fixed combination order" begin
        # floating-point sum: bitwise the same for every worker count and run, and equal to the chunked fold in order
        x = [1 / (1 + i)^1.1 * (iseven(i) ? -1 : 1) for i in 1:5000]
        ref(chunk) = begin
            parts = [foldl(+, (x[i] for i in (lo + 2):min(lo + chunk, 5000)); init=x[lo + 1]) for lo in 0:chunk:4999]
            foldl(+, parts)
        end
        for chunk in (1, 7, 100)
            results = [Dagger.dynamic_reduce((i, v) -> v[i], +, 5000, x; chunk, workers=w) for w in (1, 2, 8, 8)]
            @test all(r -> r === ref(chunk), results)       # same bits, not just approximately equal
        end
        @test Dagger.dynamic_reduce((i, v) -> v[i], +, 5000, x) === Dagger.dynamic_reduce((i, v) -> v[i], +, 5000, x; workers=3)
        # not commutative: the order of the indices is kept
        @test Dagger.dynamic_reduce(i -> string(i, ","), *, 40; chunk=6, workers=4) == join(string.(1:40, ","))
        # init, empty range, a DTask argument, a failure
        @test Dagger.dynamic_reduce(i -> i, +, 100; init=1000) == 1000 + 5050
        @test Dagger.dynamic_reduce(i -> i, +, 0; init=7) == 7
        @test_throws ArgumentError Dagger.dynamic_reduce(i -> i, +, 0)
        slow_range(n) = (sleep(0.2); collect(1:n))
        v = Dagger.@spawn slow_range(100)
        @test Dagger.dynamic_reduce((i, vec) -> vec[i], +, 100, v) == 5050
        @test_throws Exception Dagger.dynamic_reduce(i -> i == 7 ? error("boom") : 1, +, 100)
    end
    @testset "allreduce without MPI folds in list order" begin
        tasks = [Dagger.@spawn(identity(i)) for i in 1:10]
        @test Dagger.allreduce(+, tasks) == 55
        @test Dagger.allreduce(*, [Dagger.@spawn(identity(string(i))) for i in 1:5]) == "12345"
        @test Dagger.allreduce(+, tasks; init=100) == 155
    end
end

