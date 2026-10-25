using Dagger, Test, Distributed, LinearAlgebra
import Dagger: ThreadProc, Processor
import Dagger.Sch: LOCAL_FASTPATH, COMPAT_PROCS_CACHE

# With workers, unscoped tasks may run anywhere (and take the general path), so the code the tasks
# run exists everywhere; what the fast path decides is checked where it can be seen, on one process.
@everywhere begin
    using Dagger
    struct NoThreadFn end
    (::NoThreadFn)(x) = x
    Dagger.iscompatible_func(::Dagger.ThreadProc, opts, ::Type{NoThreadFn}) = false
    FP_TINY(x) = x + 1
    fp_proc() = (sleep(0.002); Dagger.task_processor())
end

# The fast path's view of every cached compatible set that qualified
fast_entries() = count(e -> e.fast isa Dagger.Sch.LocalFast, values(Dagger.payload(COMPAT_PROCS_CACHE).cache))

# Total reserved pressure of the eager scheduler; zero once everything has finished
function reserved_pressure()
    state = Dagger.Sch.EAGER_STATE[]
    total = UInt64(0)
    lock(state.worker_time_pressure) do wtp
        for pm in values(wtp), c in values(pm)
            total += c[]
        end
    end
    return total
end

@testset "Local fast path" begin
    saved = LOCAL_FASTPATH[]
    try
        @testset "tasks complete and release their pressure" begin
            LOCAL_FASTPATH[] = true
            Dagger.Sch.invalidate_compat_procs_cache!()
            ts = [Dagger.@spawn FP_TINY(i) for i in 1:500]
            @test sum(fetch.(ts)) == sum(2:501)
            Dagger.@spawn(FP_TINY(0)) |> fetch   # one more so the last completions have been handled
            # unscoped tasks on a plain process qualify (with workers they may run anywhere)
            nprocs() == 1 && @test fast_entries() >= 1
            sleep(0.2)
            @test reserved_pressure() == 0
        end

        @testset "placement spreads over the threads" begin
            LOCAL_FASTPATH[] = true
            nt = Threads.nthreads()
            if nt > 1 && nprocs() == 1
                ps = Set{Processor}(fetch.([Dagger.@spawn fp_proc() for _ in 1:20 * nt]))
                @test length(ps) >= min(nt, 4)
                @test all(p -> p isa ThreadProc && p.owner == myid(), ps)
            end
        end

        @testset "options the fast path does not model take the general path" begin
            LOCAL_FASTPATH[] = true
            @test fetch(Dagger.@spawn proclist=[ThreadProc] FP_TINY(1)) == 2
            @test fetch(Dagger.@spawn single=1 FP_TINY(1)) == 2
            @test fetch(Dagger.@spawn occupancy=Dict(ThreadProc=>0.5) FP_TINY(3)) == 4
            @test fetch(Dagger.@spawn time_util=Dict(ThreadProc=>1_000_000) FP_TINY(4)) == 5
            @test fetch(Dagger.@spawn alloc_util=Dict(ThreadProc=>UInt64(1024)) FP_TINY(5)) == 6
            # a scope is honoured
            p = fetch(Dagger.@spawn scope=Dagger.scope(worker=myid(), thread=1) fp_proc())
            @test p == ThreadProc(myid(), 1)
            # so is a scope on a chunk argument
            ch = Dagger.tochunk(5, ThreadProc(myid(), 1), Dagger.ExactScope(ThreadProc(myid(), 1)))
            @test fetch(Dagger.@spawn FP_TINY(ch)) == 6
        end

        @testset "a thread that refuses the function is respected" begin
            LOCAL_FASTPATH[] = true
            t = Dagger.@spawn NoThreadFn()(1)
            @test_throws Exception fetch(t)
        end

        @testset "same results with the fast path off" begin
            LOCAL_FASTPATH[] = false
            ts = [Dagger.@spawn FP_TINY(i) for i in 1:200]
            @test sum(fetch.(ts)) == sum(2:201)
        end

        @testset "Datadeps" begin
            for flag in (true, false)
                LOCAL_FASTPATH[] = flag
                A = [zeros(8) for _ in 1:12]
                Dagger.spawn_datadeps() do
                    for a in A
                        Dagger.@spawn fill!(Dagger.InOut(a), 3.0)
                    end
                    for a in A
                        Dagger.@spawn rmul!(Dagger.InOut(a), 2.0)
                    end
                end
                @test all(a -> all(==(6.0), a), A)
            end
        end
    finally
        LOCAL_FASTPATH[] = saved
    end
end
