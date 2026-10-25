# `Dagger.dynamic_for` under MPI: the workers run on one rank (their shared counter lives there),
# every index runs exactly once, and a failure in `f` reaches every rank.
const BULK_N = 4000
const bulk_hits = zeros(Int, BULK_N)
bulk_mark(i) = (bulk_hits[i] += 1; nothing)
bulk_report() = (count(==(1), bulk_hits), count(>(1), bulk_hits), count(==(0), bulk_hits))

@testset "Dynamic loop under MPI" begin
    last_rank = MPI.Comm_size(comm) - 1
    scope = Dagger.scope(mpi_rank=last_rank)
    fill!(bulk_hits, 0)
    MPI.Barrier(comm)
    Dagger.dynamic_for(bulk_mark, BULK_N; scope)
    ones_, dups, misses = fetch(Dagger.@spawn(scope=scope, bulk_report()))
    @test (ones_, dups, misses) == (BULK_N, 0, 0)
    @test_throws Exception Dagger.dynamic_for(i -> i == 3 ? error("boom") : nothing, 100; scope)
end
MPI.Barrier(comm)
