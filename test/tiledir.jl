using Dagger, Test, Random
using Dagger: TileName, TileDirectory, LocalDirNetwork, DirEntry, tile_home, dir_serve!, dir_put!, dir_future, dir_set_ranks!

include("tiledir_defs.jl")

@testset "Tile directory" begin
    @testset "announce at the home, ask from anywhere" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        names = [TileName(1, i) for i in 1:200]
        owners = Dict(n => ids[mod1(n.index * 7, 4)] for n in names)
        for n in names; announce!(ds, n, owners[n], 1); end
        for n in names
            @test homes_of(ds, n) == [tile_home(n, ids)]                  # exactly one directory holds the entry: its home
        end
        for (r, d) in ds, n in names
            @test await(dir_future(d, n)) == DirEntry(owners[n], 1)       # from every rank
        end
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "messages: none at the home, a round trip elsewhere, none from the cache" begin
        ids = [0, 1, 2]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(5, 1)
        h = tile_home(n, ids); other = first(i for i in ids if i != h)
        announce!(ds, n, other, 1)
        empty!(ds[other].cache)                                           # announcing waited for the entry, which put it into the owner's cache
        s0 = total_sent(ds)
        @test await(dir_future(ds[h], n)).owner == other                  # at the home
        @test total_sent(ds) == s0
        s1 = total_sent(ds)
        @test await(dir_future(ds[other], n)).owner == other              # elsewhere: a request and a reply
        @test total_sent(ds) == s1 + (other == h ? 0 : 2)
        s2 = total_sent(ds)
        @test await(dir_future(ds[other], n)).owner == other              # the cache answers
        @test total_sent(ds) == s2
        dir_close(net, tasks)
    end

    @testset "a pull future waits for its version; an old version never overwrites a new one" begin
        ids = [0, 1, 2, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        n = TileName(2, 9)
        f = dir_future(ds[1], n; minversion=3)
        sleep(0.05)
        @test !isready(f)
        announce!(ds, n, 2, 1)
        sleep(0.05)
        @test !isready(f)                                                  # version 1 is not enough
        announce!(ds, n, 2, 3)
        @test await(f) == DirEntry(2, 3)
        dir_put!(ds[0], n, 0, 2)                                           # late and old: ignored
        g = dir_future(ds[3], n; minversion=3)
        @test await(g) == DirEntry(2, 3)
        announce!(ds, n, 0, 4)                                             # the tile moved to rank 0 with a new version
        @test await(dir_future(ds[2], n; minversion=4)) == DirEntry(0, 4)
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "a rank joins: only the names it wins move, and every name is still found" begin
        old = [0, 1, 2, 3]; new = [0, 1, 2, 3, 4]
        net, ds, tasks = dir_cluster(new, old)
        names = [TileName(3, i) for i in 1:400]
        for n in names; announce!(ds, n, old[mod1(n.index, 4)], 1); end
        ds[4] = TileDirectory(4, old, net); tasks[4] = dir_serve!(ds[4], net)
        for d in values(ds); dir_set_ranks!(d, new); end
        moved = [n for n in names if tile_home(n, old) != tile_home(n, new)]
        ok = timedwait(() -> all(n -> homes_of(ds, n) == [tile_home(n, new)], names), 20.0; pollint=0.005)
        @test ok === :ok
        @test all(n -> tile_home(n, new) == 4, moved)
        @test 0.12 <= length(moved) / length(names) <= 0.30                # about a fifth
        for d in values(ds); empty!(d.cache); end
        for (r, d) in ds, n in names[1:50]
            @test await(dir_future(d, n)) == DirEntry(old[mod1(n.index, 4)], 1)
        end
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "a rank leaves: its entries are handed over, nothing is lost" begin
        all_ids = [0, 1, 2, 3, 4]
        net, ds, tasks = dir_cluster(all_ids, all_ids)
        names = [TileName(4, i) for i in 1:400]
        for n in names; announce!(ds, n, 0, 1); end
        held = count(n -> tile_home(n, all_ids) == 2, names)
        @test held > 0
        rest = [0, 1, 3, 4]
        for d in values(ds); dir_set_ranks!(d, rest); end
        ok = timedwait(() -> all(n -> homes_of(ds, n) == [tile_home(n, rest)], names), 20.0; pollint=0.005)
        @test ok === :ok
        @test isempty(ds[2].entries)
        for d in values(ds); empty!(d.cache); end
        for n in names[1:60]
            @test await(dir_future(ds[3], n)) == DirEntry(0, 1)
        end
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "a request that is waiting follows its tile to the new home" begin
        old = [0, 1, 2, 3]; new = [0, 1, 2, 3, 4]
        net, ds, tasks = dir_cluster(new, old)
        n = first(TileName(6, i) for i in 1:10_000 if tile_home(TileName(6, i), old) != tile_home(TileName(6, i), new))
        f = dir_future(ds[0], n; minversion=2)                             # waits at the old home
        sleep(0.05)
        @test !isready(f)
        ds[4] = TileDirectory(4, old, net); tasks[4] = dir_serve!(ds[4], net)
        for d in values(ds); dir_set_ranks!(d, new); end
        @test tile_home(n, new) == 4
        timedwait(() -> haskey(ds[4].waiters, n), 10.0)
        @test haskey(ds[4].waiters, n)                                      # the waiting request moved with the home
        dir_put!(ds[2], n, 2, 2)                                            # sender 2 already knows the new set
        @test await(f) == DirEntry(2, 2)
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "concurrent announces and requests from many threads" begin
        ids = [10, 20, 30, 40, 50]
        net, ds, tasks = dir_cluster(ids, ids)
        names = [TileName(7, i) for i in 1:300]
        owner(n) = ids[mod1(n.index, 5)]
        nv = 5
        writers = map(1:5) do w
            Threads.@spawn for n in names
                owner(n) == ids[w] || continue
                for v in 1:nv; dir_put!(ds[ids[w]], n, ids[w], v); end
            end
        end
        readers = map(1:4) do r
            Threads.@spawn begin
                rng = MersenneTwister(r)
                bad = 0
                for _ in 1:600
                    n = rand(rng, names); mv = rand(rng, 1:nv)
                    e = await(dir_future(ds[ids[rand(rng, 1:5)]], n; minversion=mv))
                    (e.owner == owner(n) && e.version >= mv) || (bad += 1)
                end
                bad
            end
        end
        foreach(wait, writers)
        @test all(fetch(r) == 0 for r in readers)
        for d in values(ds); empty!(d.cache); end
        for n in names
            @test await(dir_future(ds[ids[1]], n; minversion=nv)) == DirEntry(owner(n), nv)
        end
        @test all(d -> isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "ranks change to the new set one after the other (epochs)" begin
        # a rank leaves: the others learn of it at different times
        ids = [0, 1, 2, 3]; rest = [0, 1, 3]
        net, ds, tasks = dir_cluster(ids, ids)
        names = [TileName(12, i) for i in 1:200]
        for n in names; announce!(ds, n, 0, 1); end
        n = first(m for m in names if tile_home(m, ids) == 2 && tile_home(m, rest) == 3)
        for r in (0, 1); dir_set_ranks!(ds[r], rest); end                  # 0 and 1 are in epoch 1; 2 and 3 are still in epoch 0
        empty!(ds[0].cache)
        f = dir_future(ds[0], n; minversion=1)                              # goes to rank 3, the new home, which does not know the new set yet
        sleep(0.1)
        @test !isready(f)
        @test lock(() -> length(ds[3].held), ds[3].lock) >= 1               # held at rank 3, not passed on to rank 2
        s0 = total_sent(ds); sleep(0.2)
        @test total_sent(ds) == s0                                          # and nothing bounces between 2 and 3
        dir_set_ranks!(ds[3], rest)
        sleep(0.05)
        @test !isready(f)                                                    # rank 3 is the home now, but the entry is still at rank 2
        dir_set_ranks!(ds[2], rest)                                          # rank 2 changes last and hands its entries over
        @test await(f) == DirEntry(0, 1)
        ok = timedwait(() -> all(m -> homes_of(ds, m) == [tile_home(m, rest)], names), 20.0; pollint=0.005)
        @test ok === :ok
        @test all(d -> isempty(d.held) && isempty(d.errors) && d.epoch == 1, values(ds))
        dir_close(net, tasks)
        # a rank joins: the new rank and one old rank are ahead of the others
        old = [0, 1, 2, 3]; new = [0, 1, 2, 3, 4]
        net, ds, tasks = dir_cluster(new, old)
        for m in names; announce!(ds, m, 1, 1); end
        n2 = first(m for m in names if tile_home(m, old) in (0, 2, 3) && tile_home(m, new) == 4)      # its old home changes late
        ds[4] = TileDirectory(4, old, net); tasks[4] = dir_serve!(ds[4], net)
        for r in (4, 1); dir_set_ranks!(ds[r], new); end
        empty!(ds[1].cache)
        g = dir_future(ds[1], n2; minversion=1)                              # to rank 4; the entry is still at its old home
        sleep(0.1)
        @test !isready(g)
        for r in (0, 2, 3); dir_set_ranks!(ds[r], new); end                  # the old home hands it over
        @test await(g) == DirEntry(1, 1)
        @test timedwait(() -> all(m -> homes_of(ds, m) == [tile_home(m, new)], names), 20.0; pollint=0.005) === :ok
        @test all(d -> isempty(d.held) && isempty(d.errors), values(ds))
        dir_close(net, tasks)
    end

    @testset "an epoch only goes up" begin
        net, ds, tasks = dir_cluster([0, 1], [0, 1])
        dir_set_ranks!(ds[0], [0, 1]; epoch=5)
        @test ds[0].epoch == 5
        @test_throws ArgumentError dir_set_ranks!(ds[0], [0, 1]; epoch=5)
        @test_throws ArgumentError dir_set_ranks!(ds[0], [0, 1]; epoch=3)
        dir_close(net, tasks)
    end
end
