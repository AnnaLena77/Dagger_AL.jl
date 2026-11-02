# Tile versions over MPI (src/tilestore.jl): included by cpu.jl, which has set up MPI acceleration, `comm`, `nranks`.
@testset "Tile versions (MPI)" begin
    me = MPI.Comm_rank(comm)
    d = Dagger.tile_directory()
    val(v, r, n=64) = [Float64(100000v + 1000r + i) for i in 1:n]
    tile(r) = Dagger.TileName(21, r)
    # every rank publishes its tile and every other rank reads it
    readers = nranks - 1
    tv = Dagger.tile_publish!(d, tile(me), 1, val(1, me); readers)
    got = [fetch(Dagger.tile_fetch(d, tile(r), 1)) for r in 0:nranks-1 if r != me]
    @test all(i -> got[i] == val(1, [r for r in 0:nranks-1 if r != me][i]), eachindex(got))
    nranks > 1 && wait(tv.done)                                   # all other ranks have fetched it
    @test lock(() -> !haskey(d.data, (tile(me), 1)), d.lock)
    # the next version only after the readers of the last one were served; the readers wait for it through the directory
    futs = [Dagger.tile_fetch(d, tile(r), 2) for r in 0:nranks-1 if r != me]
    Dagger.barrier(; comm)
    tv2 = Dagger.tile_publish!(d, tile(me), 2, val(2, me); readers)
    got2 = [fetch(f) for f in futs]
    @test all(i -> got2[i] == val(2, [r for r in 0:nranks-1 if r != me][i]), eachindex(got2))
    nranks > 1 && wait(tv2.done)
    Dagger.barrier(; comm)
    @test isempty(d.data) && isempty(d.errors)
    # a tile of 256 x 256 Float64 (512 KB) travels intact
    big = reshape([Float64(i + 7me) for i in 1:256*256], 256, 256)
    tvb = Dagger.tile_publish!(d, Dagger.TileName(22, me), 1, big; readers)
    for r in 0:nranks-1
        r == me && continue
        @test fetch(Dagger.tile_fetch(d, Dagger.TileName(22, r), 1)) == reshape([Float64(i + 7r) for i in 1:256*256], 256, 256)
    end
    nranks > 1 && wait(tvb.done)
    Dagger.barrier(; comm)
    @test isempty(d.data) && isempty(d.errors)
end
