export TileSpace, TileIn, TileRW, tile_spawn!, tile_sync!, tile_data, tile_stats

# TileFlow: owner-computes execution of task graphs over whole tiles.  Opt-in and additive; the Datadeps path is untouched.
#
# Every rank walks the same program (the loop nest of the algorithm) and calls `tile_spawn!` for EVERY task, but only does real work for the tasks it
# owns.  Per task each rank updates one tiny replicated table: per tile a version and a bitmask of the ranks that hold the current version.  From
# it every rank derives, without a single message, which transfers happen (sender = lowest holder, receiver = the task's rank, tag = per-pair
# sequence number) and whether it takes part.  Own tasks run as `@light` tasks, so there is no Thunk, Chunk, MemPool, scope or processor per task.
#
# A task is `tile_spawn!(space, f, TileIn(id), TileRW(id)...)`.  It runs on the rank of its first `TileRW` tile (or `at=`); `f` receives the tile
# arrays in argument order and modifies the `TileRW` ones in place.  Tiles are whole objects: no views or sub-ranges (use Datadeps for those), no
# tasks created inside tasks.  See AGENTS.md lessons 138-139 for why tiles are created by many threads and received into pooled buffers.
#
# Transports (`TileTransport`) carry the arrays: `LocalTileTransport` (one process, nothing to send) is the default; MPIExt provides one for MPI.

abstract type TileTransport end
"One process, one rank: every tile is local and nothing is ever sent."
struct LocalTileTransport <: TileTransport end
"`tile_transport(accel)`: the transport for an acceleration (MPIExt adds the MPI one)."
tile_transport(accel) = LocalTileTransport()
tile_rank(::LocalTileTransport) = 0
tile_nranks(::LocalTileTransport) = 1
"`tile_post_send(tr, arr, dest, tag)` / `tile_post_recv(tr, arr, source, tag)`: start a non-blocking transfer, return a `Base.Event` that is notified when it is done."
function tile_post_send end
function tile_post_recv end
"`tile_barrier(tr)`: a barrier that does not block the OS thread (cooperative)."
tile_barrier(::LocalTileTransport) = nothing

struct TileIn; id::Int; end
struct TileRW; id::Int; end

mutable struct LocalTile                # the local copy of the CURRENT version of a tile
    arr::Any
    replica::Bool                       # a received copy: its buffer goes back to the pool once nobody reads it any more
    ready::Any                          # nothing | LightTask that produces/fills it
    readers::Vector{Any}                # LightTasks that read this version (a later writer must wait for them)
end

mutable struct TileSpace
    tr::TileTransport
    me::Int
    np::Int
    holders::Vector{UInt64}             # replicated: bit r set = rank r holds the current version
    version::Vector{Int32}              # replicated
    tiles::Vector{Union{Nothing,LocalTile}}
    own::Function                       # id -> rank that owns the data initially
    tsize::Function                     # id -> (m, n) of the tile (needed to allocate receive buffers)
    T::DataType                         # element type of every tile (an isbits type, so that tiles are plain buffers)
    pairseq::Vector{Int}                # np*np sequence numbers of transfers per (sender, dest)
    ntasks::Int; nown::Int; nxfer::Int; nsent::Int; nrecv::Int
    all_tasks::Vector{Any}
    pool::Dict{Tuple{Int,Int},Vector{Matrix}}
    pool_lock::ReentrantLock
end

"""
    TileSpace(ntiles, own, tsize, init; transport=LocalTileTransport())

A set of `ntiles` tiles identified by `1:ntiles`.  `own(id)` is the rank that initially holds tile `id`, `tsize(id)` its size `(m, n)` (Float64
matrices), `init(id)` creates it (called on the owner, from many threads: first touch of the memory should be spread over the machine).
Every rank must build the same space and make the same `tile_spawn!` calls in the same order.
"""
function TileSpace(ntiles::Int, own::Function, tsize::Function, init::Function; T::DataType=Float64, transport::TileTransport=tile_transport(current_acceleration()))
    isbitstype(T) || throw(ArgumentError("TileSpace needs an isbits element type, got $T"))
    me = tile_rank(transport); np = tile_nranks(transport)
    np <= 64 || throw(ArgumentError("TileSpace supports at most 64 ranks (holder bitmask)"))
    sp = TileSpace(transport, me, np, fill(UInt64(0), ntiles), zeros(Int32, ntiles), Vector{Union{Nothing,LocalTile}}(nothing, ntiles),
                   own, tsize, T, zeros(Int, np * np), 0, 0, 0, 0, 0, Any[], Dict{Tuple{Int,Int},Vector{Matrix}}(), ReentrantLock())
    mine = Int[]
    for id in 1:ntiles
        o = own(id); (0 <= o < np) || throw(ArgumentError("own($id) = $o is not a rank of 0:$(np-1)"))
        sp.holders[id] = UInt64(1) << o
        o == me && push!(mine, id)
    end
    # first touch by many threads: on a NUMA machine the creating thread would otherwise put every tile in one domain
    tl = Vector{Any}(undef, length(mine))
    Threads.@threads :static for i in eachindex(mine)
        tl[i] = init(mine[i])
    end
    for (i, id) in enumerate(mine); sp.tiles[id] = LocalTile(tl[i], false, nothing, Any[]); end
    return sp
end

function _pool_take(sp::TileSpace, m, n)
    lock(sp.pool_lock) do
        v = get(sp.pool, (m, n), nothing)
        (v !== nothing && !isempty(v)) ? pop!(v) : Matrix{sp.T}(undef, m, n)
    end
end
function _pool_give(sp::TileSpace, a::Matrix)
    lock(sp.pool_lock) do
        push!(get!(Vector{Matrix}, sp.pool, size(a)), a)
    end
    return nothing
end

struct _TileRelease; sp::TileSpace; arr::Matrix; end
(r::_TileRelease)(deps...) = _pool_give(r.sp, r.arr)
struct _TileRunner; f::Any; arrs::Vector{Any}; end
(g::_TileRunner)(deps...) = (g.f(g.arrs...); nothing)
# A send goes out even when the task that produced the tile failed (`LightAfter`): the receiving rank is already waiting for it, and a rank that
# never gets its tile would hang the whole run instead of failing it. The error is agreed on in `tile_sync!`.
_tile_send(tr, arr, dest, tag, ready...) = (wait(tile_post_send(tr, arr, dest, tag)); nothing)
_tile_wait(ev) = (wait(ev); nothing)

const TAG_MOD = 4_000_000     # tags stay below Open MPI's tag_ub (8388607); matching is by (pair, tag) order, tags are unique per pair in flight

"""
    tile_spawn!(space, f, args...; at=nothing)

Run `f(arrays...)` on the owner of the first `TileRW` argument (or on rank `at`).  Call it on every rank, identically.  Returns `nothing`.
"""
function tile_spawn!(sp::TileSpace, f, args::Union{TileIn,TileRW}...; at::Union{Nothing,Int}=nothing)
    isempty(args) && throw(ArgumentError("tile_spawn! needs at least one tile"))
    sp.ntasks += 1
    r = at
    if r === nothing
        for a in args
            a isa TileRW && (r = sp.own(a.id); break)
        end
        r === nothing && (r = sp.own(first(args).id))
    end
    # the task's rank for a tile that moved: a tile always lives where it was last written, not where it started
    me = sp.me; rbit = UInt64(1) << r
    for a in args
        id = a.id
        h = sp.holders[id]
        if h & rbit == 0
            sender = trailing_zeros(h)
            k = sender * sp.np + r + 1
            sp.pairseq[k] += 1; tag = sp.pairseq[k] % TAG_MOD
            sp.nxfer += 1
            if me == sender
                t = sp.tiles[id]::LocalTile
                st = light_spawn(_tile_send, sp.tr, t.arr, r, tag, t.ready === nothing ? nothing : LightAfter(t.ready))
                push!(t.readers, st); push!(sp.all_tasks, st); sp.nsent += 1
            elseif me == r
                m, n = sp.tsize(id)
                arr = _pool_take(sp, m, n)
                ev = tile_post_recv(sp.tr, arr, sender, tag)          # posted now, in program order: rendezvous sends find their receive at once
                rt = light_spawn(_tile_wait, ev)
                sp.tiles[id] = LocalTile(arr, true, rt, Any[]); sp.nrecv += 1
            end
            sp.holders[id] = h | rbit
        end
    end
    if me == r
        sp.nown += 1
        deps = Any[]; arrs = Any[]
        for a in args
            t = sp.tiles[a.id]::LocalTile
            push!(arrs, t.arr); t.ready === nothing || push!(deps, t.ready)
            a isa TileRW && append!(deps, t.readers)             # write-after-read
        end
        task = light_spawn(_TileRunner(f, arrs), deps...)
        push!(sp.all_tasks, task)
        for a in args
            t = sp.tiles[a.id]::LocalTile
            if a isa TileRW
                t.ready = task; empty!(t.readers)
            else
                push!(t.readers, task)
            end
        end
    end
    for a in args
        if a isa TileRW
            sp.version[a.id] += 1; sp.holders[a.id] = rbit
            t = sp.tiles[a.id]
            if t !== nothing && t.replica
                if me == r
                    t.replica = false                            # the owner now writes into it: a live tile, not a pooled copy
                else
                    deps = Any[]; t.ready === nothing || push!(deps, t.ready); append!(deps, t.readers)
                    push!(sp.all_tasks, light_spawn(_TileRelease(sp, t.arr), deps...))
                    sp.tiles[a.id] = nothing
                end
            end
        end
    end
    return nothing
end

"""
    tile_sync!(space)

Wait for everything this rank started, then agree with the other ranks on whether any kernel failed anywhere: the barrier is reached on every rank
even when a kernel failed here, the first local error is rethrown, and a rank that saw no error of its own still throws when another rank did.
"""
function tile_sync!(sp::TileSpace)
    err = nothing
    for t in sp.all_tasks
        try
            fetch(t)
        catch e
            err === nothing && (err = e)
        end
    end
    empty!(sp.all_tasks)
    tile_barrier(sp.tr)
    anyfail = tile_allreduce_max(sp.tr, err === nothing ? 0 : 1)
    err === nothing || throw(err)
    anyfail == 0 || error("TileFlow: a task failed on another rank")
    return nothing
end

"`tile_allreduce_max(tr, x::Int)`: the maximum of `x` over all ranks (cooperative; every rank must call it)."
tile_allreduce_max(::LocalTileTransport, x::Int) = x
"""
Generic allreduce on the tile send/receive primitives: everybody sends its value to rank 0, rank 0 sends the maximum back (tags above `TAG_MOD`,
so they never meet a tile transfer).
"""
function tile_allreduce_max(tr::TileTransport, x::Int)
    me = tile_rank(tr); np = tile_nranks(tr)
    np == 1 && return x
    tag = TAG_MOD + 1000
    buf = [Float64(x)]
    if me == 0
        m = x
        bufs = [zeros(1) for _ in 1:np-1]
        evs = [tile_post_recv(tr, bufs[r], r, tag) for r in 1:np-1]
        for ev in evs; wait(ev); end
        for b in bufs; m = max(m, Int(b[1])); end
        out = [Float64(m)]
        sevs = [tile_post_send(tr, out, r, tag + 1) for r in 1:np-1]
        for ev in sevs; wait(ev); end
        return m
    else
        sev = tile_post_send(tr, buf, 0, tag)
        res = zeros(1)
        rev = tile_post_recv(tr, res, 0, tag + 1)
        wait(sev); wait(rev)
        return Int(res[1])
    end
end

"The local array of a tile (only valid on a rank that holds the current version, after `tile_sync!`)."
tile_data(sp::TileSpace, id::Int) = (t = sp.tiles[id]; t === nothing ? nothing : t.arr)
tile_stats(sp::TileSpace) = (tasks=sp.ntasks, own=sp.nown, transfers=sp.nxfer, sent=sp.nsent, received=sp.nrecv)


# --- Chunks as tiles: which rank holds a `Chunk`, and its array on that rank ---------------------------------------------------------------
"`tile_chunk_owner(chunk) -> Int`: the rank of the transport that holds the chunk's array, or -1 when this layer cannot use it (MPIExt adds MPI chunks)."
tile_chunk_owner(c::Chunk) = c.handle isa MemPool.DRef && c.handle.owner == myid() && myid() == 1 ? 0 : -1
"`tile_chunk_array(chunk)`: the array itself (no copy) on the holding rank, `nothing` elsewhere."
tile_chunk_array(c::Chunk) = c.handle isa MemPool.DRef ? MemPool.poolget(c.handle) : nothing
"`tile_chunk_transport_ok(tr, chunk)`: whether the chunk belongs to the communicator of `tr`."
tile_chunk_transport_ok(tr::TileTransport, c::Chunk) = tr isa LocalTileTransport
