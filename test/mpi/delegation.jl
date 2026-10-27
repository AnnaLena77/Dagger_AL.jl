# Home-based task delegation (`lend=true`, `Dagger.enable_delegation!`): the owner of a ready
# task may lend its *execution* to an idle rank and gets the value (or the exception) back.
# Part 1 tests the claim protocol of the lend pool alone, with real threads. Part 2 runs
# the whole protocol across the ranks of the job.

const LendPool = MPIExt.LendPool
const LendRecord = MPIExt.LendRecord

lend_rec(pool, i) = MPIExt.LendRecord(MPIExt.lend_pool_new_uid!(pool), identity, (i,), (;))

@testset "Delegation: lend pool" begin
    @testset "sequence: oldest runs here, newest is lent" begin
        pool = LendPool(1)
        recs = [lend_rec(pool, i) for i in 1:6]
        @test MPIExt.lend_pool_enter!(pool, recs[1])            # takes the only slot
        @test !any(r -> MPIExt.lend_pool_enter!(pool, r), recs[2:6])
        @test pool.queue == recs[2:6]
        # a thief takes from the new end, skipping what must stay
        recs[6].nolend = true
        @test MPIExt.lend_pool_steal!(pool, 3) === recs[5]
        @test MPIExt.lend_pool_steal!(pool, 3) === recs[4]
        @test recs[5].state == MPIExt.LEND_LENT && recs[5].thief == 3
        # a freed slot goes to the oldest queued task, in order
        MPIExt.lend_pool_release!(pool)
        @test recs[2].state == MPIExt.LEND_LOCAL && recs[2].event.set
        MPIExt.lend_pool_release!(pool)
        @test recs[3].state == MPIExt.LEND_LOCAL
        @test pool.free == 0 && pool.queue == [recs[6]]
        # a refused claim goes back to the old end, behind nothing
        @test MPIExt.lend_pool_requeue!(pool, recs[4].uid)
        @test pool.queue == [recs[4], recs[6]] && recs[4].state == MPIExt.LEND_QUEUED
        # an unknown or already answered uid is ignored
        @test !MPIExt.lend_pool_requeue!(pool, UInt64(10^9))
        @test MPIExt.lend_pool_complete!(pool, UInt64(10^9), false, 1) === nothing
        done = MPIExt.lend_pool_complete!(pool, recs[5].uid, false, 25)
        @test done === recs[5] && done.state == MPIExt.LEND_DONE && done.result == 25
        @test MPIExt.lend_pool_complete!(pool, recs[5].uid, false, 26) === nothing   # one answer only
        # a claim whose slot freed up meanwhile runs at once
        @test MPIExt.lend_pool_steal!(pool, 2) === recs[4]
        MPIExt.lend_pool_release!(pool); MPIExt.lend_pool_release!(pool)   # recs[6] gets one, one slot is free
        @test recs[6].state == MPIExt.LEND_LOCAL && pool.free == 1
        @test MPIExt.lend_pool_requeue!(pool, recs[4].uid)
        @test recs[4].state == MPIExt.LEND_LOCAL && pool.free == 0
    end

    @testset "shutdown fails what is still lent" begin
        pool = LendPool(1)
        a, b = lend_rec(pool, 1), lend_rec(pool, 2)
        MPIExt.lend_pool_enter!(pool, a)
        MPIExt.lend_pool_enter!(pool, b)
        @test MPIExt.lend_pool_steal!(pool, 1) === b
        waiter = Threads.@spawn (wait(b.event); (b.state, b.failed, b.result))
        yield()
        @test MPIExt.lend_pool_fail_lent!(pool, ErrorException("stopped")) == 1
        state, failed, err = fetch(waiter)
        @test state == MPIExt.LEND_DONE && failed && err isa ErrorException
    end

    # Owners wait, local slots are released, thieves steal and answer, all at once, with a tiny
    # capacity so the pool is always at its boundary. Every owner must get *its own* answer
    # exactly once: counts alone would not see an answer delivered to the wrong task.
    @testset "claim race with real threads (capacity $cap)" for cap in (1, 2)
        pool = LendPool(cap)
        n = 400
        ran_local = Threads.Atomic{Int}(0)
        ran_lent = Threads.Atomic{Int}(0)
        stop = Threads.Atomic{Bool}(false)
        thieves = [Threads.@spawn begin
            did = 0
            while !stop[]
                rec = MPIExt.lend_pool_steal!(pool, 1)
                if rec === nothing
                    sleep(0.0001)
                    continue
                end
                # sometimes hand it back, as a thief without room does
                if did % 5 == 4
                    MPIExt.lend_pool_requeue!(pool, rec.uid)
                else
                    MPIExt.lend_pool_complete!(pool, rec.uid, false, rec.args[1]^2)
                end
                did += 1
            end
        end for _ in 1:2]
        function run_owner(i)
            rec = lend_rec(pool, i)
            if MPIExt.lend_pool_enter!(pool, rec) || (wait(rec.event); rec.state == MPIExt.LEND_LOCAL)
                Threads.atomic_add!(ran_local, 1)
                sleep(0.0001)
                MPIExt.lend_pool_release!(pool)
                return i^2
            end
            Threads.atomic_add!(ran_lent, 1)
            return rec.result
        end
        results = fetch.([Threads.@spawn(run_owner(i)) for i in 1:n])
        stop[] = true
        foreach(wait, thieves)
        @test results == [i^2 for i in 1:n]
        @test ran_local[] + ran_lent[] == n
        @test ran_lent[] > 0 && ran_local[] > 0
        @test pool.free == cap && isempty(pool.queue) && isempty(pool.lent)
    end
end

# --- across ranks ----------------------------------------------------------------------

deleg_busy(i::Int, secs::Float64) = (sleep(secs); (i * i, MPI.Comm_rank(MPI.COMM_WORLD)))
deleg_boom(i::Int, secs::Float64) = (sleep(secs); error("boom $i from rank $(MPI.Comm_rank(MPI.COMM_WORLD))"))
deleg_where(x) = (MPI.Comm_rank(MPI.COMM_WORLD), Threads.threadid(), x)
deleg_plain(i::Int) = i + 1
deleg_chunk_busy(c, i::Int, secs::Float64) = deleg_busy(i, secs)

function deleg_counters()
    st = Dagger.delegation_stats()
    return [st.lent, st.stolen, st.local_, st.returned, st.failed]
end
# sum of a counter vector over the ranks (no tasks are in flight when this is called)
deleg_total(v) = Dagger.allreduce_value(+, v; comm)

@testset "Delegation: across ranks" begin
    @test Dagger.delegation_stats() === nothing          # nothing is enabled by default
    # lend=true without delegation enabled is plain execution
    @test fetch(Dagger.@spawn lend=true deleg_plain(1)) == 2
    @test Dagger.delegation_stats() === nothing
    Dagger.barrier(; comm)

    Dagger.enable_delegation!(slots=2)
    @test Dagger.delegation_stats() == (lent=0, failed=0, returned=0, stolen=0, local_=0)
    owner_rank = 0
    # every task prefers the owner of its chunk argument, so rank 0 is flooded while the others idle
    chunk = Dagger.tochunk(0, proc_for_rank(owner_rank), space_for_rank(owner_rank))

    @testset "results of lent tasks" begin
        n = 24
        tasks = [Dagger.@spawn lend=true deleg_busy(i, 0.05) for i in 1:n]
        res = fetch.(tasks)
        @test first.(res) == [i * i for i in 1:n]
        @test all(r -> 0 <= last(r) < nranks, res)
        Dagger.barrier(; comm)
        c = deleg_counters()
        tot = deleg_total(c)
        # everything lendable ran somewhere, either at home or on a thief: nothing twice, nothing lost
        @test tot[3] + tot[1] == n       # `local_` ran at home, `lent` were answered by a thief
        @test tot[1] == tot[2]           # every answer is the run of exactly one thief
    end

    @testset "flooded rank lends to idle ranks" begin
        if nranks > 1
            before = deleg_total(deleg_counters())
            n = 24
            tasks = [Dagger.@spawn lend=true deleg_chunk_busy(chunk, i, 0.08) for i in 1:n]
            res = fetch.(tasks)
            @test first.(res) == [i * i for i in 1:n]
            Dagger.barrier(; comm)
            after = deleg_total(deleg_counters())
            @test all(fetch(t; raw=true).handle.rank == owner_rank for t in tasks)   # the owner never changed
            @test after[1] - before[1] > 0                  # some were lent
            @test after[1] - before[1] == after[2] - before[2]     # lent == run by thieves
            @test after[3] - before[3] + after[1] - before[1] == n   # each task ran once
            ranks_that_ran = unique(last.(res))
            @test length(ranks_that_ran) > 1                # and really ran elsewhere
        end
    end

    @testset "exceptions come back" begin
        before = deleg_total(deleg_counters())
        n = 12
        tasks = [Dagger.@spawn lend=true deleg_boom(i, 0.05) for i in 1:n]
        msgs = String[]
        for (i, t) in enumerate(tasks)
            err = try
                fetch(t)
                nothing
            catch e
                e
            end
            @test err !== nothing
            push!(msgs, sprint(showerror, err))
            @test occursin("boom $i from rank", msgs[end])
        end
        # the failures that were lent reached the owner as failures, not as a hang
        Dagger.barrier(; comm)
        after = deleg_total(deleg_counters())
        @test after[5] - before[5] <= after[1] - before[1]
        # and the runtime is still usable
        @test fetch(Dagger.@spawn lend=true deleg_busy(3, 0.01))[1] == 9
    end

    @testset "mixed pinned and free tasks" begin
        cpus = filter(p -> p.innerProc isa Dagger.ThreadProc, mpi_procs())
        target = last(cpus)
        Dagger.barrier(; comm)
        before = deleg_counters()
        pinned = [Dagger.@spawn(scope=Dagger.ExactScope(target), lend=true, deleg_where(i)) for i in 1:8]
        plain = [Dagger.@spawn deleg_plain(i) for i in 1:16]
        for (i, t) in enumerate(pinned)
            r = fetch(t)
            # a task that names its processor is never lent
            @test r == (target.rank, target.innerProc.tid, i)
        end
        @test fetch.(plain) == [i + 1 for i in 1:16]
        Dagger.barrier(; comm)
        # neither pinned nor plain tasks pass through the pool: enabling delegation changes nothing for them
        @test deleg_counters() == before
        free = [Dagger.@spawn lend=true deleg_busy(i, 0.03) for i in 1:16]
        @test first.(fetch.(free)) == [i * i for i in 1:16]
    end

    @testset "shutdown with an open request" begin
        # every idle rank holds a request at its peers; stopping must not hang or fail anything
        sleep(0.2)
        Dagger.barrier(; comm)
        Dagger.disable_delegation!()
        @test Dagger.delegation_stats() === nothing
        @test fetch(Dagger.@spawn lend=true deleg_plain(41)) == 42
        # and it can be switched on again
        Dagger.enable_delegation!(slots=2)
        res = fetch.([Dagger.@spawn lend=true deleg_busy(i, 0.02) for i in 1:8])
        @test first.(res) == [i * i for i in 1:8]
        Dagger.barrier(; comm)
        Dagger.disable_delegation!()
    end
end
