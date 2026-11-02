# The tile directory over MPI (src/tiledir.jl, ext/MPIExt.jl); included by cpu.jl, which has set up MPI acceleration, `comm`, `nranks`.
@testset "Tile directory (MPI)" begin
    me = MPI.Comm_rank(comm)
    d = Dagger.tile_directory()                       # collective: duplicates the communicator
    @test d.me == me && d.ranks == collect(0:nranks-1)
    names = [Dagger.TileName(11, i) for i in 1:48]
    owner(n) = n.index % nranks
    # every rank asks for every tile BEFORE anybody announces: pull futures that wait at the homes
    futs = [Dagger.dir_future(d, n; minversion=1) for n in names]
    Dagger.barrier(; comm)
    for n in names; owner(n) == me && Dagger.dir_put!(d, n, me, 1); end
    es = [fetch(f) for f in futs]
    @test all(i -> es[i] == Dagger.DirEntry(owner(names[i]), 1), eachindex(names))
    # the entries live at their homes, and at no other rank
    Dagger.barrier(; comm)
    @test all(n -> Dagger.tile_home(n, d.ranks) == me, keys(d.entries))
    @test sum(Dagger.allgather_value(length(d.entries); comm)) == length(names)
    # a new version on another rank: the readers wait for it, and the old entry in their cache does not answer
    futs2 = [Dagger.dir_future(d, n; minversion=2) for n in names]
    Dagger.barrier(; comm)
    for n in names; (owner(n) + 1) % nranks == me && Dagger.dir_put!(d, n, me, 2); end
    es2 = [fetch(f) for f in futs2]
    @test all(i -> es2[i] == Dagger.DirEntry((owner(names[i]) + 1) % nranks, 2), eachindex(names))
    @test isempty(d.errors)
    # the same directory is returned again (no second communicator)
    @test Dagger.tile_directory() === d
    nranks > 1 && @test sum(Dagger.allgather_value(d.sent[]; comm)) > 0     # messages did cross the network
end
