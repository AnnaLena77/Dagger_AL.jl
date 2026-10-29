# `Dagger.dynamic_for` under MPI: the workers run on one rank (their shared counter lives there),
# every index runs exactly once, and a failure in `f` reaches every rank.
const BULK_N = 4000
const bulk_hits = zeros(Int, BULK_N)
bulk_mark(i) = (bulk_hits[i] += 1; nothing)
bulk_gate(n) = n
bulk_mark_with(i, n) = (i <= n && (bulk_hits[i] += 1); nothing)
bulk_report() = (count(==(1), bulk_hits), count(>(1), bulk_hits), count(==(0), bulk_hits))

@testset "Dynamic loop under MPI" begin
    last_rank = MPI.Comm_size(comm) - 1
    scope = Dagger.scope(mpi_rank=last_rank)
    fill!(bulk_hits, 0)
    Dagger.barrier(; comm)
    Dagger.dynamic_for(bulk_mark, BULK_N; scope)
    ones_, dups, misses = fetch(Dagger.@spawn(scope=scope, bulk_report()))
    @test (ones_, dups, misses) == (BULK_N, 0, 0)
    @test_throws Exception Dagger.dynamic_for(i -> i == 3 ? error("boom") : nothing, 100; scope)

    # task arguments: the loop starts once its DTask argument exists, on the rank of the scope
    fill!(bulk_hits, 0)
    Dagger.barrier(; comm)
    gate = Dagger.@spawn scope=scope bulk_gate(BULK_N)
    Dagger.dynamic_for(bulk_mark_with, BULK_N, gate; scope)
    @test fetch(Dagger.@spawn(scope=scope, bulk_report())) == (BULK_N, 0, 0)
end
Dagger.barrier(; comm)

# The default scope of the loops (no `scope=`): the processors of this rank, in a graph of one rank. It used to be
# `Dagger.scope(worker=myid())`, which matches no MPI processor ("No processors available"). In the shared graph a per-rank
# default would make the ranks disagree about where a worker runs, so it must be an error there, not a silent default.
bulk_item(i) = [i]
@testset "Dynamic loop and reduce: default scope under MPI" begin
    Dagger.barrier(; comm)
    if MPI.Comm_size(comm) > 1
        @test_throws ArgumentError Dagger.dynamic_for(i -> nothing, 60)
        @test_throws ArgumentError Dagger.dynamic_reduce(bulk_item, vcat, 30; init=Int[], chunk=1)
    end
    r = Dagger.with_rank_local() do
        (sum(Dagger.dynamic_for(i -> nothing, 60)),
         Dagger.dynamic_reduce(bulk_item, vcat, 30; init=Int[], chunk=1))
    end
    @test r == (60, collect(1:30))
    @test Dagger.with_rank_local(Dagger.local_scope) isa Dagger.UnionScope
end
Dagger.barrier(; comm)
