using Dagger, Test, Random
using Dagger: TileName, TileDirectory, DirEntry, tile_home, dir_put!, dir_future, tile_publish!, tile_fetch, tile_drop!

include("tiledir_defs.jl")

val(v, n=8) = [Float64(1000v + i) for i in 1:n]            # the value of version v: easy to check

@testset "Tile versions: publish, pull, lifetime" begin
    @testset "a version is served to as many readers as announced, then it is gone" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(1, 1)
        tv = tile_publish!(ds[0], n, 1, val(1); readers=3)
        got = [await(tile_fetch(ds[r], n, 1)) for r in (1, 2)]
        @test all(==(val(1)), got)
        sleep(0.05)
        @test lock(() -> haskey(ds[0].data, (n, 1)), ds[0].lock)       # one reader still to come: kept
        @test await(tile_fetch(ds[3], n, 1)) == val(1)
        @test timedwait(() -> !lock(() -> haskey(ds[0].data, (n, 1)), ds[0].lock), 5.0; pollint=0.005) === :ok
        wait(tv.done)                                                    # notified when the third fetch was served
        e = try; await(tile_fetch(ds[1], n, 1)); nothing; catch err; err; end   # a fourth reader: the version is gone
        @test e isa ErrorException && occursin("not held", e.msg)
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "the owner reads its own version: no message, and it counts as a reader" begin
        ids = [0, 1]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(2, 1)
        tv = tile_publish!(ds[0], n, 1, val(1); readers=2)
        sleep(0.05)
        s0 = total_sent(ds)
        empty!(ds[0].cache)
        @test await(tile_fetch(ds[0], n, 1)) == val(1)                    # local: the directory home may be rank 1, the data are here
        local_extra = total_sent(ds) - s0                                 # (only the lookup at the home, if the home is the other rank)
        @test await(tile_fetch(ds[1], n, 1)) == val(1)
        wait(tv.done)
        @test isempty(ds[0].data)
        @test local_extra <= 2
        dir_close(net, tasks)
    end

    @testset "a reader that asks before the owner publishes waits for the version" begin
        ids = [0, 1, 2]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(3, 1)
        slot = tile_fetch(ds[2], n, 1)
        sleep(0.1)
        @test !isready(slot)
        tile_publish!(ds[1], n, 1, val(1); readers=1)
        @test await(slot) == val(1)
        dir_close(net, tasks)
    end

    @testset "no readers: nothing is kept, the announcement still happens" begin
        ids = [0, 1]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(4, 1)
        tv = tile_publish!(ds[0], n, 1, val(1); readers=0)
        wait(tv.done)
        @test isempty(ds[0].data)
        @test await(dir_future(ds[1], n; minversion=1)) == DirEntry(0, 1)
        dir_close(net, tasks)
    end

    @testset "the next version waits for the readers of the last one (anti-dependency)" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(5, 1)
        order = Channel{Symbol}(Inf)
        tv1 = tile_publish!(ds[0], n, 1, val(1); readers=2)
        readers = map((1, 2)) do r
            Threads.@spawn begin
                sleep(0.05 * r)
                v = await(tile_fetch(ds[r], n, 1))
                put!(order, :read)
                v == val(1)
            end
        end
        writer = Threads.@spawn begin
            wait(tv1.done)                                                # the anti-dependency: both fetches have been served
            put!(order, tv1.remaining == 0 && !lock(() -> haskey(ds[0].data, (n, 1)), ds[0].lock) ? :write_after_served : :too_early)
            tile_publish!(ds[0], n, 2, val(2); readers=1)
        end
        @test all(fetch, readers)
        tv2 = fetch(writer)
        events = [take!(order) for _ in 1:3]
        @test count(==(:write_after_served), events) == 1 && count(==(:read), events) == 2      # the next version came only after both fetches were served
        @test !(:too_early in events)
        @test await(tile_fetch(ds[3], n, 2)) == val(2)
        wait(tv2.done)
        @test all(d -> isempty(d.data) && isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "a tile that passes from rank to rank, one new owner per version" begin
        ids = [0, 1, 2, 3, 4]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(6, 1)
        K = 40
        tvs = Vector{Any}(undef, K)
        tvs[1] = tile_publish!(ds[0], n, 1, zeros(8); readers=1)
        steps = map(2:K) do k
            Threads.@spawn begin
                me = ids[mod1(k, 5)]
                prev = await(tile_fetch(ds[me], n, k - 1))                # pull the last version from whoever owns it
                tvs[k] = tile_publish!(ds[me], n, k, prev .+ 1.0; readers=1)
            end
        end
        foreach(wait, steps)
        final = await(tile_fetch(ds[2], n, K))
        @test final == fill(Float64(K - 1), 8)
        @test all(tv -> (wait(tv.done); true), tvs[1:K])                  # every version was consumed
        @test all(d -> isempty(d.data) && isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "many tiles, many readers, from many threads" begin
        ids = [10, 20, 30, 40]
        net, ds, tasks = dir_cluster(ids, ids)
        names = [TileName(7, i) for i in 1:150]
        nv = 3
        owner(n) = ids[mod1(n.index, 4)]
        reader_ranks(n, v) = [ids[mod1(n.index + v + j, 4)] for j in 1:3]       # three readers per version (possibly the owner)
        tvs = Vector{Any}()
        lk = ReentrantLock()
        pubs = map(names) do n
            Threads.@spawn for v in 1:nv
                tv = tile_publish!(ds[owner(n)], n, v, val(v); readers=3)
                lock(() -> push!(tvs, tv), lk)
            end
        end
        foreach(wait, pubs)
        bad = Threads.Atomic{Int}(0)
        reads = [Threads.@spawn(begin
                     v = await(tile_fetch(ds[r], n, v0))
                     v == val(v0) || Threads.atomic_add!(bad, 1)
                 end) for n in names for v0 in 1:nv for r in reader_ranks(n, v0)]
        foreach(wait, reads)
        @test bad[] == 0
        @test all(tv -> (wait(tv.done); true), tvs)
        @test all(d -> isempty(d.data) && isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end
end
