# Dagger.rebalance: a rank-consistent assignment of chunks from costs, and the moves that carry it out.
@testset "rebalance_plan: pure, deterministic, sticky" begin
    # balanced: nothing moves
    @test isempty(Dagger.rebalance_plan([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0], 2))
    # all the work on rank 0: half of it moves, in a deterministic way
    plan = Dagger.rebalance_plan([0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0], 2)
    @test sort(last.(plan)) == [1, 1]
    @test Dagger.rebalance_plan([0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0], 2) == plan
    # the chunk closest to half of the gap moves; a chunk bigger than the gap never does
    @test Dagger.rebalance_plan([0, 0, 1], [5.0, 3.0, 1.0], 2) == [2 => 1]
    @test isempty(Dagger.rebalance_plan([0, 1], [10.0, 1.0], 2; tolerance=100))   # within the (huge) tolerance
    @test isempty(Dagger.rebalance_plan([0], [10.0], 2))                           # one indivisible chunk: nothing to gain
    @test length(Dagger.rebalance_plan([0, 0, 0, 0, 0, 0], fill(1.0, 6), 3; max_moves=1)) == 1
    @test_throws ArgumentError Dagger.rebalance_plan([0, 3], [1.0, 1.0], 2)
    @test_throws ArgumentError Dagger.rebalance_plan([0, 1], [1.0, -1.0], 2)
end

@testset "rebalance: chunks change owner, data intact, tasks follow" begin
    nr = MPI.Comm_size(comm)
    if nr >= 2
        P = 2 * nr
        procs_by_rank = Dict(r => first(filter(p -> p.rank == r, mpi_procs())) for r in 0:nr-1)
        # every chunk starts on rank 0, with its index as data
        chunks = [Dagger.tochunk(fill(Float64(i), 1000), procs_by_rank[0], Dagger.memory_space(procs_by_rank[0])) for i in 1:P]
        @test all(c -> Dagger.owner_rank(c) == 0, chunks)
        costs = fill(1.0, P)
        new, moves = Dagger.rebalance(chunks, costs)
        owners = Int[Dagger.owner_rank(c) for c in new]
        load = [count(==(r), owners) for r in 0:nr-1]
        @test maximum(load) - minimum(load) <= 1
        @test length(moves) == count(!=(0), owners)
        # the data arrived and the tasks bound to the new owner run there
        sums = [fetch(Dagger.@spawn(scope=Dagger.scope(mpi_rank=Dagger.owner_rank(c)), sum(c))) for c in new]
        @test sums == [1000.0 * i for i in 1:P]
        # a second call with the new owners and equal costs moves nothing
        new2, moves2 = Dagger.rebalance(new, costs)
        @test isempty(moves2)
        @test all(i -> new2[i] === new[i], 1:P)
    end
    @test_throws ArgumentError Dagger.with_rank_local() do
        Dagger.rebalance(Dagger.Chunk[], Float64[])
    end
end
Dagger.barrier(; comm)

@testset "rebalance: groups move together, DTask elements are accepted" begin
    nr = MPI.Comm_size(comm)
    if nr >= 2
        procs_by_rank = Dict(r => first(filter(p -> p.rank == r, mpi_procs())) for r in 0:nr-1)
        mkc(owner, v) = Dagger.tochunk(fill(Float64(v), 100), procs_by_rank[owner], Dagger.memory_space(procs_by_rank[owner]))
        P = 2 * nr
        # a partition = (buf, meta, red), all on rank 0 at first
        parts = [(mkc(0, i), mkc(0, 10i), mkc(0, 100i)) for i in 1:P]
        new, moves = Dagger.rebalance(parts, fill(1.0, P))
        @test all(g -> g isa AbstractVector && length(g) == 3, new)
        owners = Int[Dagger.owner_rank(first(g)) for g in new]
        @test all(g -> all(c -> Dagger.owner_rank(c) == Dagger.owner_rank(first(g)), g), new)   # a group never splits
        @test maximum([count(==(r), owners) for r in 0:nr-1]) - minimum([count(==(r), owners) for r in 0:nr-1]) <= 1
        sums = [[fetch(Dagger.@spawn(scope=Dagger.scope(mpi_rank=Dagger.owner_rank(c)), sum(c))) for c in g] for g in new]
        @test sums == [[100.0i, 1000.0i, 10000.0i] for i in 1:P]
        # DTask elements (as in DArray.chunks)
        dts = Any[Dagger.@spawn(scope=Dagger.scope(mpi_rank=0), identity(fill(Float64(i), 100))) for i in 1:P]
        new2, _ = Dagger.rebalance(dts, fill(1.0, P))
        @test all(c -> c isa Dagger.Chunk, new2)
        @test [fetch(Dagger.@spawn(scope=Dagger.scope(mpi_rank=Dagger.owner_rank(c)), sum(c))) for c in new2] == [100.0i for i in 1:P]
        # a group whose members have different owners is refused
        @test_throws ArgumentError Dagger.rebalance([(mkc(0, 1), mkc(1, 1))], [1.0])
    end
end
Dagger.barrier(; comm)
