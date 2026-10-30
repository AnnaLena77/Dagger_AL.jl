# Dagger.skip_tasks!: a rank that does not take part in a task leaves the @spawn out; ids stay uniform (BH entry 66)
const SKIP_SEEN = Int[]
record_skip(k) = (push!(SKIP_SEEN, k); k)
@testset "skip_tasks!: ids stay uniform when ranks skip foreign tasks" begin
    nr = MPI.Comm_size(comm); me = MPI.Comm_rank(comm)
    own_task(x) = x + 1
    old = Dagger.ELIDE_FOREIGN_TASKS[]
    Dagger.elide_foreign_tasks!(true)
    try
        # Warm-up with an ordinary task on every rank: the first task of a process draws one extra id on the rank that runs it
        fetch(Dagger.@spawn scope=Dagger.scope(mpi_rank=0) return_type=Int own_task(0))
        # ASYMMETRIC on purpose: rank 0 takes part in all N tasks, the others in none (a symmetric batch hides counter drift).
        # No fetch of a skipped task: a fetch is a collective (the owner sends the value to every rank) and draws ids.
        N = 7
        empty!(SKIP_SEEN)
        ts = Any[]
        for k in 1:N
            if me == 0
                push!(ts, Dagger.@spawn scope=Dagger.scope(mpi_rank=0) return_type=Int record_skip(k))
            else
                Dagger.skip_tasks!(1)
            end
        end
        if me == 0
            t0 = time(); while length(SKIP_SEEN) < N && time() - t0 < 30; sleep(0.01); end      # a failure-only task is not waited for by `wait`
            @test sort(SKIP_SEEN) == collect(1:N)
        end
        # the next task of the shared counter has the same id on every rank, past the batch
        nxt = Dagger.@spawn scope=Dagger.scope(mpi_rank=0) return_type=Int own_task(0)
        uids = MPI.Allgather([Int(nxt.uid)], comm)
        @test allequal(uids)
        @test fetch(nxt) == 1
        if me == 0
            @test [Int(t.uid) for t in ts] == collect(Int(ts[1].uid):Int(ts[1].uid) + N - 1)    # consecutive: nothing else drew from the counter
        end
        # several at once, none, and a negative count
        Dagger.skip_tasks!(0)
        @test_throws ArgumentError Dagger.skip_tasks!(-1)
    finally
        Dagger.elide_foreign_tasks!(old)
    end
    @test_throws ArgumentError (Dagger.elide_foreign_tasks!(false); try Dagger.skip_tasks!(1) finally Dagger.elide_foreign_tasks!(old) end)
end
