# `Dagger.allgather`: every rank gets the values of tasks owned by all ranks, without a gather task or a broadcast,
# and without taking task or reference ids.
ag_make(i::Int, n::Int) = fill(Float64(i), n)
ag_sum(x::Vector{Float64}) = sum(x)
ag_bad(i::Int) = i > 0 ? error("allgather failure $i") : i

@testset "Dagger.allgather" begin
    last_thread(r) = Dagger.ExactScope(last(sort(filter(p -> p.rank == r && p.innerProc isa Dagger.ThreadProc, mpi_procs());
                                                  by=p -> p.innerProc.tid)))
    Dagger.check_uniformity!(false)
    # The number of partners per round decides how many rounds a rank takes; every fanout must give the same answer
    @testset "native collective" begin
        MPIExt.MPI_ALLGATHER_NATIVE[] = true
        Dagger.elide_foreign_tasks!(true)
        n = 2 * nranks + 1
        ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 50 + i)) for i in 1:n]
        vals = Dagger.allgather(ts)
        @test all(i -> vals[i] == fill(Float64(i), 50 + i), 1:n)
        # sizes differ a lot between ranks, some are empty
        mixed = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, i == 1 ? 0 : 20_000 * i)) for i in 1:n]
        mv = Dagger.allgather(mixed)
        @test all(i -> length(mv[i]) == (i == 1 ? 0 : 20_000 * i), 1:n)
        Dagger.elide_foreign_tasks!(false)
    end
    # The rounds of the hand-written version: the number of partners per round decides how many rounds a rank takes;
    # every fanout must give the same answer
    @testset "rounds, fanout $k" for k in (1, 2, 3, 4)
        MPIExt.MPI_ALLGATHER_NATIVE[] = false
        MPIExt.MPI_ALLGATHER_FANOUT[] = k
        Dagger.elide_foreign_tasks!(true)
        n = 2 * nranks + 1
        ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 50 + i)) for i in 1:n]
        vals = Dagger.allgather(ts)
        @test all(i -> vals[i] == fill(Float64(i), 50 + i), 1:n)
        Dagger.elide_foreign_tasks!(false)
    end
    MPIExt.MPI_ALLGATHER_FANOUT[] = 4
    MPIExt.MPI_ALLGATHER_NATIVE[] = nothing          # the default again (the rounds, unless the environment says otherwise)
    for elide in (true, false)
        Dagger.elide_foreign_tasks!(elide)
        try
            @testset "owners spread over the ranks, elision=$elide" begin
                n = 2 * nranks + 1                       # uneven: some ranks own more tasks than others
                ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 100 + 7i)) for i in 1:n]
                vals = Dagger.allgather(ts)
                @test length(vals) == n
                @test all(i -> vals[i] == fill(Float64(i), 100 + 7i), 1:n)
                # the order of the list decides the order of the result, whatever the owners are
                vals2 = Dagger.allgather(reverse(ts))
                @test vals2 == reverse(vals)
                # the element type is the concrete one the values share, or the one asked for
                @test eltype(vals) == Vector{Float64}
                @test typeof(Dagger.allgather(ts, Vector{Float64})) == Vector{Vector{Float64}}
                s = Dagger.allgather([Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Int, ag_bad(0)) for i in 1:nranks])
                @test s isa Vector{Int}
            end
            @testset "empty list and a large payload, elision=$elide" begin
                @test Dagger.allgather(Dagger.DTask[]) == []
                big = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Vector{Float64}, ag_make(i, 1_000_000)) for i in 1:nranks]
                vals = Dagger.allgather(big)
                @test all(i -> length(vals[i]) == 1_000_000 && vals[i][end] == Float64(i), 1:nranks)
            end
            @testset "a failed task reaches every rank, elision=$elide" begin
                ok = Dagger.@spawn scope=last_thread(0) return_type=Int ag_bad(0)
                bad = Dagger.@spawn scope=last_thread(min(1, nranks - 1)) return_type=Int ag_bad(1)
                @test_throws Exception Dagger.allgather([ok, bad])
                # and the next call still works, ranks are in step
                @test Dagger.allgather([ok]) == [0]
            end
            @testset "ids stay in agreement, elision=$elide" begin
                before = Dagger.@spawn scope=last_thread(0) return_type=Int ag_bad(0)
                fetch(before)
                Dagger.allgather([before])
                t = Dagger.@spawn scope=last_thread(nranks - 1) return_type=Int ag_bad(0)
                @test fetch(t) == 0
                @test allequal(MPI.Allgather(Int(t.uid), comm))
            end
        finally
            Dagger.elide_foreign_tasks!(false)
        end
    end
    Dagger.check_uniformity!(true)
    MPI.Barrier(comm)
end

# `Dagger.allreduce`: the fold is in list order, so the floating-point result is bitwise the same on every rank
# and the same as folding the gathered list on one process.
ar_part(i::Int) = 1 / (1 + i)^1.3 * (iseven(i) ? -1.0 : 1.0)
@testset "Dagger.allreduce" begin
    last_thread(r) = Dagger.ExactScope(last(sort(filter(p -> p.rank == r && p.innerProc isa Dagger.ThreadProc, mpi_procs());
                                                  by=p -> p.innerProc.tid)))
    Dagger.check_uniformity!(false)
    Dagger.elide_foreign_tasks!(true)
    try
        n = 3 * nranks + 2
        ts = [Dagger.@spawn(scope=last_thread(mod(i, nranks)), return_type=Float64, ar_part(i)) for i in 1:n]
        total = Dagger.allreduce(+, ts)
        @test total === foldl(+, ar_part.(1:n))
        @test allequal(MPI.Allgather(reinterpret(UInt64, [total])[1], comm))       # identical bits on every rank
        @test Dagger.allreduce(+, ts; init=10.0) === foldl(+, ar_part.(1:n); init=10.0)
    finally
        Dagger.elide_foreign_tasks!(false)
        Dagger.check_uniformity!(true)
    end
    MPI.Barrier(comm)
end

