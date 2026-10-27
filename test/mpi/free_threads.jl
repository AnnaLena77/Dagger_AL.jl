# `Dagger.free_mpi_threads!`: an MPI task whose thread nobody asked for runs on any free
# thread of its rank instead of waiting behind a busy one.
hog_busy(seconds::Float64) = (t0 = time(); while time() - t0 < seconds; end; 1)   # does not yield
free_quick(x::Int) = x + 1
free_stamp(x::Int) = (x + 1, time())   # also when it ran
free_where() = Threads.threadid()

@testset "Free MPI threads" begin
    if Threads.nthreads() >= 3
        cpus = filter(p -> p.innerProc isa Dagger.ThreadProc, mpi_procs())
        rank0 = sort(filter(p -> p.rank == 0, cpus); by=p -> p.innerProc.tid)
        Dagger.free_mpi_threads!(true)
        try
            # Warm up first: the first fetch of a task compiles for a second on slow nodes, which would be measured otherwise
            foreach(fetch, Dagger.with_options(; scope=Dagger.scope(mpi_rank=0)) do
                [Dagger.@spawn(free_quick(i)) for i in 1:16]
            end)
            foreach(fetch, [Dagger.@spawn(scope=Dagger.ExactScope(p), hog_busy(0.01)) for p in rank0])
            Dagger.barrier(; comm)
            # Occupy every thread of rank 0 but the first with a pinned (so bound) task that never yields
            hogs = [Dagger.@spawn(scope=Dagger.ExactScope(p), hog_busy(1.5)) for p in rank0[2:end]]
            sleep(0.3)
            Dagger.barrier(; comm)
            # warm up the stamping variant as well
            foreach(fetch, Dagger.with_options(; scope=Dagger.scope(mpi_rank=0)) do
                [Dagger.@spawn(free_stamp(i)) for i in 1:16]
            end)
            Dagger.barrier(; comm)
            t0 = time()
            # Scope = the whole rank, 16 tasks: the scheduler names every thread in turn, busy ones included
            tasks = Dagger.with_options(; scope=Dagger.scope(mpi_rank=0)) do
                [Dagger.@spawn(free_stamp(i)) for i in 1:16]
            end
            res = fetch.(tasks)
            @test first.(res) == [i + 1 for i in 1:16]
            # Without the switch, the tasks the scheduler places on a busy thread wait for the hogs (measured: 3.3 s of 4 s).
            # N.B. What is asserted is when the tasks *ran*, not when `fetch` returned them: with every thread but one busy
            # the delivery of a result to the other ranks (the broadcast relay shares that one thread) is delayed
            # by up to half a second on top (measured), which is not what this switch is about.
            last_start = maximum(last.(res)) - t0
            rank == 0 && @test last_start < 1.0    # the hogs still have over a second to run
            # An explicit thread is still a promise
            where = fetch(Dagger.@spawn scope=Dagger.ExactScope(last(rank0)) free_where())
            @test where == last(rank0).innerProc.tid
            foreach(wait, hogs)
        finally
            Dagger.free_mpi_threads!(false)
        end
    end
    Dagger.barrier(; comm)
end

# The task of an MPI processor starts pinned to the thread of its inner processor, so
# `execute!(::ThreadProc)` runs the function inline instead of creating a second pinned task
# and waiting for it (measured: 160-190 us per task)
@testset "MPI task starts pinned to its processor's thread" begin
    cpus = filter(p -> p.innerProc isa Dagger.ThreadProc, mpi_procs())
    mine = first(sort(filter(p -> p.rank == rank, cpus); by=p -> p.innerProc.tid))
    others = filter(p -> p.rank != rank, cpus)
    tid_for = Dagger.Sch.task_tid_for_processor
    spec(opts) = (; options = opts)    # the hook only reads `task.options`
    # N.B. The first testset is skipped on fewer than 3 threads, so do not rely on it having left the switch off
    Dagger.free_mpi_threads!(false)
    @test tid_for(mine, spec(Dagger.Options())) == mine.innerProc.tid
    # A processor of another rank has no thread here
    isempty(others) || @test tid_for(first(others), spec(Dagger.Options())) === nothing
    # With free threads on, only a task that names its thread is pinned
    Dagger.free_mpi_threads!(true)
    try
        @test tid_for(mine, spec(Dagger.Options())) === nothing
        @test tid_for(mine, spec(Dagger.Options(; scope=Dagger.ExactScope(mine)))) == mine.innerProc.tid
        # A Datadeps task (tagged) names its processor for the plan; it is bound only if free_datadeps_threads! is off
        datadeps_opts = Dagger.Options(; scope=Dagger.ExactScope(mine), tag=1)
        Dagger.free_datadeps_threads!(false)
        try
            @test tid_for(mine, spec(datadeps_opts)) == mine.innerProc.tid
            Dagger.free_datadeps_threads!(true)
            @test tid_for(mine, spec(datadeps_opts)) === nothing
            # ... while an explicit pin without a tag is still a promise
            @test tid_for(mine, spec(Dagger.Options(; scope=Dagger.ExactScope(mine)))) == mine.innerProc.tid
        finally
            Dagger.free_datadeps_threads!(false)    # the default
        end
    finally
        Dagger.free_mpi_threads!(false)
    end
    # And it runs where it was pinned
    if rank == 0
        @test fetch(Dagger.@spawn(scope=Dagger.ExactScope(mine), free_where())) == mine.innerProc.tid
    else
        fetch(Dagger.@spawn(scope=Dagger.ExactScope(first(filter(p -> p.rank == 0, cpus))), free_where()))
    end
    Dagger.barrier(; comm)
end
