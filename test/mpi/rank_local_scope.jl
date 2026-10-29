# A scope that names this rank means the same in the shared and in a rank-local graph (BH entry 58)
whoami_tid() = Threads.threadid()
@testset "rank scopes inside a rank-local graph" begin
    me = MPI.Comm_rank(comm); other = (me + 1) % MPI.Comm_size(comm)
    nthr = Threads.nthreads()
    Dagger.with_rank_local() do
        # the rank's own world scope runs on this graph's processors
        @test fetch(Dagger.@spawn(scope=Dagger.scope(mpi_rank=me), 1 + 1)) == 2
        # a loop given the world rank scope expands its workers from the graph's processors
        r = fetch(Dagger.dynamic_reduce_async((x, i) -> [Threads.threadid()], vcat, 64, 1; init=Int[], chunk=1, workers=min(3, nthr), scope=Dagger.scope(mpi_rank=me)))
        @test length(r) == 64
        # another rank's scope still fails loudly, with the hint
        msg = try fetch(Dagger.@spawn(scope=Dagger.scope(mpi_rank=other), 1 + 1)); "no error" catch e; sprint(showerror, e) end
        @test occursin("No processors available", msg)
    end
end
