# Global names of tiles and the rank that keeps their directory entry (the "home").
#
# This is the first building block of exchangeable tiles (docs/mpi-branches/feat__mpi-owner-planning.md, "distributed directory"): a tile has a name that
# means the same on every rank and does not depend on how many ranks there are, and the rank that knows where the tile currently is (its home) follows
# from the name and from the *set of rank identifiers* alone. Nothing here looks at `np` or at a communicator:
#   * ranks are identified by a stable integer (for MPI today the rank in the world; with a changing set of processes the identifier of the process, which
#     does not change when others come and go), not by a position in a list;
#   * the home is chosen by rendezvous (highest random weight) hashing, so a rank that joins takes over only the names it wins, and a rank that leaves
#     hands over only the names it held. Everything else keeps its home and its directory entry.
# The hash is our own (splitmix64 mixing), not `Base.hash`, so all processes agree on it whatever the Julia version or the process.

"""
    TileName(space, index)

The global name of a tile: `space` identifies a collection of tiles (the same value on every rank), `index` the tile within it.
"""
struct TileName
    space::UInt64
    index::Int
end

@inline function _mix64(x::UInt64)         # splitmix64 finalizer: a bijection with good avalanche
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

"The 64-bit key of a tile name."
@inline tile_key(n::TileName) = _mix64(n.space ⊻ _mix64(UInt64(n.index) + 0x9e3779b97f4a7c15))

@inline _rank_weight(key::UInt64, rank_id::Integer) = _mix64(key ⊻ _mix64(UInt64(rank_id) * 0xff51afd7ed558ccd + 0x2545f4914f6cdd1d))

"""
    tile_home(name, ranks) -> Int

The identifier in `ranks` (any collection of distinct rank identifiers, in any order) that keeps the directory entry of the tile `name`: the one with the highest
weight for this name. Ties (only possible for equal weights of different ranks) go to the smaller identifier, so the result does not depend on the order of `ranks`.
"""
function tile_home(name::TileName, ranks)
    isempty(ranks) && throw(ArgumentError("tile_home needs at least one rank"))
    key = tile_key(name)
    best = 0; bestw = UInt64(0); first = true
    for r in ranks
        w = _rank_weight(key, r)
        if first || w > bestw || (w == bestw && r < best)
            best = Int(r); bestw = w; first = false
        end
    end
    return best
end
