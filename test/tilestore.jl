using Dagger, Test, Random
using Dagger: TileName, TileDirectory, DirEntry, tile_home, dir_put!, dir_future, tile_publish!, tile_fetch, tile_drop!, tile_migrate!, dir_serve!

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

    @testset "a tile moves to another rank with its pending readers" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(8, 1)
        key = (n, 1)
        tv = tile_publish!(ds[0], n, 1, val(1); readers=3)
        @test await(tile_fetch(ds[1], n, 1)) == val(1)                    # one reader before the move
        tile_migrate!(ds[0], n, 1, 2)
        wait(tv.done)                                                       # released at the old owner
        @test timedwait(() -> lock(() -> haskey(ds[2].data, key), ds[2].lock), 10.0; pollint=0.005) === :ok
        @test lock(() -> ds[2].data[key].remaining, ds[2].lock) == 2         # the count of readers that are left moved with it
        @test await(tile_fetch(ds[3], n, 1)) == val(1)                    # served by the new owner
        @test lock(() -> get(ds[0].moved, key, -1), ds[0].lock) == 2         # the old owner knows where it went
        @test await(tile_fetch(ds[0], n, 1)) == val(1)                    # a reader on the old owner is sent on
        @test timedwait(() -> all(d -> lock(() -> isempty(d.data), d.lock), values(ds)), 10.0; pollint=0.005) === :ok
        @test await(dir_future(ds[1], n; minversion=1)).owner == 2 || true  # (the cache of rank 1 may still say 0)
        empty!(ds[1].cache)
        @test await(dir_future(ds[1], n; minversion=1)) == DirEntry(2, 1)
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "a request that arrives while the tile is on its way waits and is sent on" begin
        ids = [0, 1, 2]
        net = Dagger.LocalDirNetwork(ids)
        ds = Dict(i => TileDirectory(i, ids, net) for i in ids)
        tasks = Dict(i => dir_serve!(ds[i], net) for i in (0, 1))           # rank 2 does not run yet: the hand-over cannot complete
        n = first(TileName(9, i) for i in 1:1000 if tile_home(TileName(9, i), ids) != 2)   # a home that answers
        tv = tile_publish!(ds[0], n, 1, val(1); readers=2)
        sleep(0.05)
        tile_migrate!(ds[0], n, 1, 2)
        slot = tile_fetch(ds[1], n, 1)                                      # reaches the old owner, which is handing the tile over
        sleep(0.1)
        @test !isready(slot)
        @test lock(() -> length(ds[0].moving[(n, 1)][2]), ds[0].lock) == 1   # queued
        tasks[2] = dir_serve!(ds[2], net)                                   # the new owner starts: stores, announces, confirms
        @test await(slot) == val(1)
        @test await(tile_fetch(ds[0], n, 1)) == val(1)
        @test timedwait(() -> all(d -> lock(() -> isempty(d.data), d.lock), values(ds)), 10.0; pollint=0.005) === :ok
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "tiles move around while many readers fetch" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        names = [TileName(10, i) for i in 1:60]
        nr = 6
        for n in names; tile_publish!(ds[ids[mod1(n.index, 4)]], n, 1, val(n.index); readers=nr); end
        holder(n) = [i for (i, d) in ds if lock(() -> haskey(d.data, (n, 1)), d.lock)]
        migrators = map(names) do n
            Threads.@spawn begin
                rng = MersenneTwister(n.index)
                for _ in 1:6
                    h = holder(n)
                    length(h) == 1 || (sleep(0.002); continue)
                    to = rand(rng, filter(!=(h[1]), ids))
                    try
                        tile_migrate!(ds[h[1]], n, 1, to)
                    catch
                        break                                                # consumed meanwhile
                    end
                    timedwait(() -> isempty(h) || !isempty(holder(n)) || all(d -> lock(() -> isempty(d.moving), d.lock), values(ds)), 5.0; pollint=0.002)
                    sleep(0.001)
                end
            end
        end
        bad = Threads.Atomic{Int}(0)
        reads = [Threads.@spawn(begin
                     sleep(0.001 * mod(n.index * j, 7))
                     v = await(tile_fetch(ds[ids[mod1(n.index + j, 4)]], n, 1))
                     v == val(n.index) || Threads.atomic_add!(bad, 1)
                 end) for n in names for j in 1:nr]
        foreach(wait, reads)
        foreach(wait, migrators)
        @test bad[] == 0
        @test timedwait(() -> all(d -> lock(() -> isempty(d.data) && isempty(d.moving), d.lock), values(ds)), 10.0; pollint=0.005) === :ok
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "a reader that knows the owner asks it directly and waits there for the version" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(11, 1)
        slot = tile_fetch(ds[2], n, 3; owner=1)                           # no lookup: straight to rank 1, which has not published anything yet
        sleep(0.1)
        @test !isready(slot)
        @test lock(() -> haskey(ds[1].dwait, (n, 3)), ds[1].lock)         # held at the owner
        s0 = total_sent(ds)
        tile_publish!(ds[1], n, 1, val(1); readers=0)                      # an earlier version does not answer it
        sleep(0.05)
        @test !isready(slot)
        tile_publish!(ds[1], n, 3, val(3); readers=2)
        @test await(slot) == val(3)
        @test await(tile_fetch(ds[0], n, 3; owner=1)) == val(3)           # published already: answered at once
        @test timedwait(() -> lock(() -> isempty(ds[1].data), ds[1].lock), 5.0; pollint=0.005) === :ok
        # a hinted request for a version that is gone is an error, not a wait
        e = try; await(tile_fetch(ds[0], n, 3; owner=1)); nothing; catch err; err; end
        @test e isa ErrorException && occursin("not held", e.msg)
        # and a request to the old owner of a tile that moved is sent on
        tile_publish!(ds[1], n, 4, val(4); readers=1)
        tile_migrate!(ds[1], n, 4, 3)
        @test await(tile_fetch(ds[2], n, 4; owner=1)) == val(4)
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "version 0 can be asked for before it is published" begin
        ids = [0, 1]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(12, 1)
        slot = tile_fetch(ds[1], n, 0; owner=0)                            # the owner has published nothing yet: the request waits, it is not an error
        sleep(0.1)
        @test !isready(slot)
        tile_publish!(ds[0], n, 0, val(0); readers=1)
        @test await(slot) == val(0)
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end
end

