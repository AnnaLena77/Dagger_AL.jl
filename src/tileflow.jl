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
    pairseq::Vector{Int}                # np*np sequence numbers of transfers per (sender, dest)
    ntasks::Int; nown::Int; nxfer::Int; nsent::Int; nrecv::Int
    all_tasks::Vector{Any}
    pool::Dict{Tuple{Int,Int},Vector{Matrix{Float64}}}
    pool_lock::ReentrantLock
end

"""
    TileSpace(ntiles, own, tsize, init; transport=LocalTileTransport())

A set of `ntiles` tiles identified by `1:ntiles`.  `own(id)` is the rank that initially holds tile `id`, `tsize(id)` its size `(m, n)` (Float64
matrices), `init(id)` creates it (called on the owner, from many threads: first touch of the memory should be spread over the machine).
Every rank must build the same space and make the same `tile_spawn!` calls in the same order.
"""
function TileSpace(ntiles::Int, own::Function, tsize::Function, init::Function; transport::TileTransport=tile_transport(current_acceleration()))
    me = tile_rank(transport); np = tile_nranks(transport)
    np <= 64 || throw(ArgumentError("TileSpace supports at most 64 ranks (holder bitmask)"))
    sp = TileSpace(transport, me, np, fill(UInt64(0), ntiles), zeros(Int32, ntiles), Vector{Union{Nothing,LocalTile}}(nothing, ntiles),
                   own, tsize, zeros(Int, np * np), 0, 0, 0, 0, 0, Any[], Dict{Tuple{Int,Int},Vector{Matrix{Float64}}}(), ReentrantLock())
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
        (v !== nothing && !isempty(v)) ? pop!(v) : Matrix{Float64}(undef, m, n)
    end
end
function _pool_give(sp::TileSpace, a::Matrix{Float64})
    lock(sp.pool_lock) do
        push!(get!(Vector{Matrix{Float64}}, sp.pool, size(a)), a)
    end
    return nothing
end

struct _TileRelease; sp::TileSpace; arr::Matrix{Float64}; end
(r::_TileRelease)(deps...) = _pool_give(r.sp, r.arr)
struct _TileRunner; f::Any; arrs::Vector{Any}; end
(g::_TileRunner)(deps...) = (g.f(g.arrs...); nothing)
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
                st = light_spawn(_tile_send, sp.tr, t.arr, r, tag, t.ready)
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

"Wait for everything this rank started (errors of the tasks are rethrown), then a cooperative barrier over all ranks."
function tile_sync!(sp::TileSpace)
    # The barrier must be reached on every rank even when a kernel failed here, or the other ranks wait for it forever: remember the first
    # error, pass the barrier, then throw.
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
    err === nothing || throw(err)
    return nothing
end

"The local array of a tile (only valid on a rank that holds the current version, after `tile_sync!`)."
tile_data(sp::TileSpace, id::Int) = (t = sp.tiles[id]; t === nothing ? nothing : t.arr)
tile_stats(sp::TileSpace) = (tasks=sp.ntasks, own=sp.nown, transfers=sp.nxfer, sent=sp.nsent, received=sp.nrecv)
