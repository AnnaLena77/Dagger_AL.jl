# Moving work between ranks between steps: a deterministic, sticky assignment of chunks (partitions) to ranks from measured costs,
# and the moves that carry it out.  The application supplies one cost per chunk and a single place per step where a re-plan may happen.

"""
    rebalance_plan(owners, costs, nranks; tolerance=1.05, max_moves=length(owners)) -> Vector{Pair{Int,Int}}

The moves (`chunk index => new rank`, ranks counted from 0) that bring the per-rank sum of `costs` toward its mean, given the current
`owners` (rank of each chunk). Pure and deterministic: every rank that calls it with the same arguments gets the same plan, which is
what a collective needs. Greedy and sticky: while the most loaded rank exceeds `tolerance` times the mean, the chunk of that rank whose
cost is closest to half the gap to the least loaded rank moves there, provided it narrows the gap; at most `max_moves` single moves, a chunk that
would move twice moves once to its final rank, and a chunk whose rank does not change is not listed. Nothing moves when the load is
already within `tolerance`.
"""
function rebalance_plan(owners::AbstractVector{<:Integer}, costs::AbstractVector{<:Real}, nranks::Integer;
                        tolerance::Real=1.05, max_moves::Integer=length(owners))
    length(owners) == length(costs) || throw(ArgumentError("`owners` and `costs` must have the same length"))
    nranks >= 1 || throw(ArgumentError("nranks must be positive"))
    owner = Int[o for o in owners]
    all(o -> 0 <= o < nranks, owner) || throw(ArgumentError("owners must be ranks in 0:$(nranks - 1)"))
    c = Float64[x for x in costs]
    all(x -> isfinite(x) && x >= 0, c) || throw(ArgumentError("costs must be finite and non-negative"))
    load = zeros(Float64, nranks)
    for i in eachindex(owner)
        load[owner[i] + 1] += c[i]
    end
    total = sum(load)
    total > 0 || return Pair{Int,Int}[]
    mean_load = total / nranks
    moved = 0
    while moved < max_moves
        hi = argmax(load); lo = argmin(load)
        load[hi] <= tolerance * mean_load && break
        gap = load[hi] - load[lo]
        best = 0; bestcost = 0.0
        for i in eachindex(owner)
            owner[i] == hi - 1 || continue
            ci = c[i]
            (ci > 0 && ci < gap) || continue
            if best == 0 || abs(ci - gap / 2) < abs(bestcost - gap / 2)
                best = i; bestcost = ci
            end
        end
        best == 0 && break
        owner[best] = lo - 1
        load[hi] -= bestcost; load[lo] += bestcost
        moved += 1
    end
    return Pair{Int,Int}[i => owner[i] for i in eachindex(owner) if owner[i] != owners[i]]
end

"""
    rebalance(chunks, costs; tolerance=1.05, max_moves=length(chunks)) -> (new_chunks, moves)

(MPI) Collective: every rank calls it at the same point with the same `chunks` and the same `costs` (one non-negative number per chunk, e.g. the
measured time of the chunk's tasks in the last step, made equal on all ranks with [`allreduce_value`](@ref)). Computes [`rebalance_plan`](@ref) and
moves each chunk that has to change owner by running `identity` on the new rank: the result is a new chunk owned by that rank, with the data copied
over (about 0.15 s for 100 MB between two ranks of a laptop; see `LLM_info/bench/migrate_cost.jl`). Returns the chunk vector to use from now on
(unchanged entries are the same objects) and the list of moves. An element of `chunks` may be a `Chunk`, a `DTask` that produces one (what `DArray.chunks`
holds before the tasks finish), or a tuple / vector of those that must move together (the buffers of one partition): the members must have the same owner,
move as a unit, and come back as a vector in the same position. The old chunks are not freed by the call: their memory is released by the garbage collector
once nothing refers to them, which took two full collections for a chunk no task had used and three for one that tasks had used (200 MB chunks, `bench/chunk_release.jl`);
`Dagger.unsafe_free!` only releases libc-allocated buffers. The application must schedule the tasks bound to a chunk on its new owner, e.g.
`Dagger.scope(mpi_rank=Dagger.owner_rank(chunk))`, and drop its references to the old chunks so their memory can be reclaimed.
Other accelerations throw.
"""
rebalance(chunks::AbstractVector, costs::AbstractVector; kwargs...) = rebalance(current_acceleration(), chunks, costs; kwargs...)
rebalance(::Acceleration, chunks::AbstractVector, costs::AbstractVector; kwargs...) =
    throw(ArgumentError("`Dagger.rebalance` moves chunks between MPI ranks: run it under `Dagger.accelerate!(:mpi)`"))

"""
    owner_rank(chunk) -> Int

The rank that owns `chunk` under MPI (its data lives there; tasks bound to it should run there), or 0 for chunks of other accelerations.
"""
owner_rank(::Any) = 0
