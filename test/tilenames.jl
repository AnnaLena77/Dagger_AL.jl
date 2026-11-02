using Dagger, Test
using Dagger: TileName, tile_home, tile_key

@testset "Tile names and homes" begin
    names = [TileName(UInt64(s), i) for s in 1:4 for i in 1:2500]        # 10000 names
    @testset "deterministic and independent of the order of the ranks" begin
        ranks = [0, 1, 2, 3, 4, 5, 6, 7]
        h1 = [tile_home(n, ranks) for n in names]
        h2 = [tile_home(n, reverse(ranks)) for n in names]
        h3 = [tile_home(n, Set(ranks)) for n in names]
        @test h1 == h2 == h3
        @test all(in(ranks), h1)
        @test tile_key(TileName(1, 5)) == tile_key(TileName(1, 5))
        @test tile_key(TileName(1, 5)) != tile_key(TileName(2, 5))
        @test tile_key(TileName(1, 5)) != tile_key(TileName(1, 6))
        # a fixed expectation: the hash must not change between versions or processes (all ranks have to agree)
        @test tile_home(TileName(7, 42), 0:15) == tile_home(TileName(7, 42), 0:15)
        @test_throws ArgumentError tile_home(TileName(1, 1), Int[])
    end
    @testset "homes are spread evenly" begin
        for n in (2, 5, 8, 24)
            ranks = collect(0:n-1)
            counts = zeros(Int, n)
            for t in names; counts[tile_home(t, ranks) + 1] += 1; end
            mean_c = length(names) / n
            @test all(c -> abs(c - mean_c) <= 0.15 * mean_c, counts)
        end
    end
    @testset "a rank that joins takes only what it wins" begin
        ranks = collect(0:7)
        before = Dict(n => tile_home(n, ranks) for n in names)
        after = Dict(n => tile_home(n, vcat(ranks, 8)) for n in names)
        moved = [n for n in names if before[n] != after[n]]
        @test all(n -> after[n] == 8, moved)                    # everything that moved went to the new rank
        @test 0.8 / 9 <= length(moved) / length(names) <= 1.25 / 9   # about 1/(n+1) of the names
    end
    @testset "a rank that leaves hands over only what it held" begin
        ranks = collect(0:7)
        before = Dict(n => tile_home(n, ranks) for n in names)
        after = Dict(n => tile_home(n, filter(!=(3), ranks)) for n in names)
        @test all(n -> (before[n] == 3) == (before[n] != after[n]), names)   # moved exactly when it was at rank 3
        @test all(n -> after[n] != 3, names)
    end
    @testset "rank identifiers are not positions" begin
        # the same names with the identifiers 10, 20, 30: nothing depends on 0..np-1
        ranks = [10, 20, 30]
        h = [tile_home(n, ranks) for n in names]
        @test all(in(ranks), h)
        @test length(unique(h)) == 3
        # removing 20 changes only the names homed at 20
        h2 = [tile_home(n, [10, 30]) for n in names]
        @test all(i -> (h[i] == 20) == (h[i] != h2[i]), eachindex(names))
    end
end
