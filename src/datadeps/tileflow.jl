# A `spawn_datadeps` region executed on the TileFlow layer (src/tileflow.jl).  Opt-in (JULIA_DAGGER_TILEFLOW_REGIONS=1 or `Dagger.tileflow_regions!(true)`).
#
# The region is recorded as usual (every `@spawn` becomes a `DTaskPair` in `queue.seen_tasks`); before the Datadeps planner starts, `tileflow_region!`
# asks whether the WHOLE region can run on the tile layer.  It can when every task has default options and only these arguments:
#   * a dense `Chunk` (an isbits `Array`, e.g. a tile of a DArray) as `In`/`Out`/`InOut` or bare, i.e. a whole-tile dependency,
#   * a plain `Array` of an isbits type (every rank has its own copy: a small result such as the `info` of a factorization) as a dependency,
#   * constants (isbits values, symbols, strings, types, functions).
# Anything else (views and sub-ranges, scopes and other options, results of tasks, a task without a tile) sends the region to the unchanged planner.
# The decision reads replicated metadata only, so every rank takes the same path.
#
# Contract of a lowered region: its tasks have no Thunk. A task's value is available on the rank that ran it (`fetch` on another rank raises), a task
# cannot be an argument of a later task, a failing task raises `DTaskFailedException` from the region on every rank.

const TILEFLOW_REGIONS = Ref(false)         # set from JULIA_DAGGER_TILEFLOW_REGIONS in __init__ (a precompiled const would bake in the build-time value)
"How many regions ran on the TileFlow layer in this process (tests assert on it: the fallback gives the same numbers)."
const TILEFLOW_REGION_RUNS = Threads.Atomic{Int}(0)
"`tileflow_regions!(on::Bool)`: let `spawn_datadeps` regions that qualify run on the TileFlow layer."
tileflow_regions!(on::Bool) = (TILEFLOW_REGIONS[] = on)

const _REGION_OK_OPTIONS = (:name, :acceleration, :return_type)
"Print why a region was not lowered (JULIA_DAGGER_TILEFLOW_WHY=1 or `Dagger.TILEFLOW_WHY[] = true`)."
const TILEFLOW_WHY = Ref(false)
_region_no(why) = (TILEFLOW_WHY[] && @info "TileFlow region not lowered: $why"; nothing)

struct RegionSlot
    k::Int                                  # which of the tile arrays handed to the kernel
end

struct RegionCall
    f::Any
    template::Vector{Any}                   # positional arguments: constants and RegionSlots
    kwargs::Vector{Pair{Symbol,Any}}
    task::DTask
    fargs::Vector{Argument}                 # for the failure summary
    accel::Any                              # the acceleration the region was recorded under
end
function region_failure(c::RegionCall, e)
    s = ThunkSummary(Int(c.task.uid), c.fargs)
    return DTaskFailedException{typeof(e)}(s, s, e)
end
function (c::RegionCall)(arrs...)
    args = Any[a isa RegionSlot ? arrs[a.k] : a for a in c.template]
    # Library kernels ask for `task_processor()` & co.: give the light task the task-local state a Dagger task has (as streams do)
    set_tls!(ThreadProc(myid(), Threads.threadid()), typemax(UInt64), nothing, [], CancelToken(), false, c.accel)
    try
        v = isempty(c.kwargs) ? c.f(args...) : c.f(args...; c.kwargs...)
        put!(c.task.future, v)
        return nothing
    catch e
        isready(c.task.future) || put!(c.task.future, region_failure(c, e); error=true)
        rethrow()
    end
end

mutable struct RegionTile
    kind::Symbol                            # :chunk | :array
    obj::Any                                # the Chunk or the Array
    T::DataType
    dims::Union{Nothing,Tuple}              # nothing: not known on this rank (a chunk of another rank is a placeholder); filled in by the exchange
    owner::Int
end

_region_const(v) = isbitstype(typeof(v)) || v isa Union{Symbol,AbstractString,Type,Function,Nothing}

# Classify one argument: (:const, v) | (:tile, key, RegionTile, write::Bool) | nothing (unsupported)
function _region_classify(v, tr, np::Int)
    v0, deps = unwrap_inout(v)
    wrapped = v isa Union{In,Out,InOut,Deps}
    if v0 isa DTask
        (istaskstarted(v0) && v0.stub === nothing && isready(v0.future)) || return _region_no("an argument is a task that has not finished (a result inside the region)")
        r = try fetch(v0; raw=true, move_value=false, unwrap=false) catch; return _region_no("fetching an argument task failed") end
        r isa Chunk || return _region_no("an argument task's result is a $(typeof(r)), not a Chunk")
        v0 = r
    end
    if v0 isa Chunk
        (length(deps) == 1 && deps[1][1] === identity) || return _region_no("a chunk argument is not a whole-tile dependency")
        ct = chunktype(v0)
        (ct <: Array && isbitstype(eltype(ct)) && isconcretetype(ct)) || return _region_no("a chunk holds a $ct, not a dense isbits Array")
        o = tile_chunk_owner(v0)
        (0 <= o < np) || return _region_no("a chunk's owner $o is not a rank of the transport")
        tile_chunk_transport_ok(tr, v0) || return _region_no("a chunk belongs to another communicator")
        # the chunk of another rank is a placeholder without a size: the owner tells everybody (see `_exchange_dims!`)
        dims = o == tile_rank(tr) ? (try Tuple(size(v0.domain)) catch; () end) : nothing
        return (:tile, _identity_hash(v0), RegionTile(:chunk, v0, eltype(ct), dims, o), deps[1][3])
    elseif typeof(v0) <: Array && isbitstype(eltype(v0))
        (length(deps) == 1 && deps[1][1] === identity) || return _region_no("an array argument is not a whole dependency")
        return (:tile, (:array, objectid(v0)), RegionTile(:array, v0, eltype(v0), size(v0), 0), deps[1][3])
    elseif !wrapped && _region_const(v0)
        return (:const, v0)
    end
    return _region_no("unsupported argument of type $(typeof(v0))")
end

function _region_plan(queue, tr, np::Int)
    pairs = queue.seen_tasks
    isempty(pairs) && return nothing
    tiles = RegionTile[]; index = Dict{Any,Int}()
    calls = Any[]
    for pair in pairs
        spec = pair.spec; task = pair.task
        opts = getfield(spec, :options)
        scope_rank = nothing                  # the rank a scope pins the task to (adopted as the task's rank)
        for f in fieldnames(Options)
            f in _REGION_OK_OPTIONS && continue
            v = getfield(opts, f)
            v === nothing && continue
            if f in (:scope, :compute_scope, :exec_scope)
                r = tile_scope_rank(v)
                r === nothing && return _region_no("task option $f is set to a scope the tile layer does not understand")
                (scope_rank === nothing || scope_rank == r) || return _region_no("task scopes name different ranks")
                scope_rank = r
                continue
            end
            # `propagates` lists the names of scoped options in effect when the task was spawned; `task_queue` is the region's own queue
            f === :propagates && v isa Vector && all(==(:task_queue), v) && continue
            return _region_no("task option $f is set ($v)")
        end
        fargs = Argument[Argument(ArgPosition(a.pos), a.value) for a in spec.fargs]
        isempty(fargs) && return _region_no("a task without arguments")
        (ispositional(fargs[1]) && fargs[1].pos.idx == 0) || return _region_no("unexpected function position")
        fn = value(fargs[1])
        positional = sort!([a for a in fargs[2:end] if ispositional(a)]; by=a -> a.pos.idx)
        template = Any[]; kwargs = Pair{Symbol,Any}[]; slots = Tuple{Int,Bool}[]
        for a in positional
            c = _region_classify(value(a), tr, np)
            c === nothing && return nothing
            if c[1] === :const
                push!(template, c[2])
            else
                _, key, tile, write = c
                id = get(index, key, 0)
                if id == 0
                    push!(tiles, tile); id = length(tiles); index[key] = id
                elseif tile.kind === :chunk && (tiles[id].obj::Chunk).handle !== (tile.obj::Chunk).handle
                    return _region_no("two chunk objects for one datum")
                end
                push!(slots, (id, write)); push!(template, RegionSlot(length(slots)))
            end
        end
        for a in fargs[2:end]
            ispositional(a) && continue
            c = _region_classify(value(a), tr, np)
            (c === nothing || c[1] !== :const) && return _region_no("a keyword argument is not a constant")
            push!(kwargs, pos_kw(a) => c[2])
        end
        isempty(slots) && return _region_no("a task without a tile argument")
        w = findfirst(s -> s[2], slots)
        rank = scope_rank !== nothing ? scope_rank : tiles[(w === nothing ? slots[1] : slots[w])[1]].owner
        (0 <= rank < np) || return _region_no("a scope names rank $rank, outside the transport")
        push!(calls, (; call=RegionCall(fn, template, kwargs, task, fargs, current_acceleration()), slots, rank))
    end
    return (; tiles, calls)
end

# Every rank needs the dimensions of every tile (to allocate receive buffers), but a chunk of another rank is only a placeholder. Each rank sends the
# dimensions of the tiles it owns to all others (at most 3 dimensions); a tile whose owner could not tell makes every rank fall back. Returns false then.
function _exchange_dims!(tr, tiles::Vector{RegionTile})
    tile_nranks(tr) == 1 && return all(t -> t.dims !== nothing && !isempty(t.dims), tiles)
    me = tile_rank(tr)
    v = zeros(Int, 4 * length(tiles))
    for (i, t) in enumerate(tiles)
        t.owner == me || continue
        d = t.dims
        (d !== nothing && 1 <= length(d) <= 3) || (v[4i-3] = -1; continue)
        v[4i-3] = length(d); for (j, x) in enumerate(d); v[4i-3+j] = x; end
    end
    all_v = tile_allgather_ints(tr, v)
    for (i, t) in enumerate(tiles)
        w = all_v[t.owner + 1]
        w[4i-3] < 1 && return false
        t.dims = Tuple(w[4i-3+j] for j in 1:w[4i-3])
    end
    return true
end

"""
    tileflow_region!(queue) -> Bool

Run the recorded region on the TileFlow layer when it qualifies (see the file header) and return `true`; return `false` and touch nothing otherwise.
"""
function tileflow_region!(queue)::Bool
    TILEFLOW_REGIONS[] || return false
    accel = current_acceleration()
    tr = tile_transport(accel)
    np = tile_nranks(tr)
    np <= 64 || return false
    plan = _region_plan(queue, tr, np)
    plan === nothing && return false
    tiles = plan.tiles; me = tile_rank(tr)
    _exchange_dims!(tr, tiles) || return false
    allmask = np == 64 ? typemax(UInt64) : (UInt64(1) << np) - 1
    mask(id) = tiles[id].kind === :array ? allmask : UInt64(1) << tiles[id].owner
    sp = TileSpace(length(tiles), id -> tiles[id].owner, id -> tiles[id].dims,
                   id -> (t = tiles[id]; t.kind === :chunk ? tile_chunk_array(t.obj) : t.obj);
                   types = id -> tiles[id].T, initial = mask, homes = mask, transport = tr)
    Threads.atomic_add!(TILEFLOW_REGION_RUNS, 1)
    for c in plan.calls
        c.call.task.stub = TileStub(c.rank)
        tile_spawn!(sp, c.call, [w ? TileRW(id) : TileIn(id) for (id, w) in c.slots]...; at=c.rank)
    end
    err, anyfail = tile_restore!(sp; throw_errors=false)
    # every future gets filled: the value (already done by the kernel on its rank), a marker where the task ran elsewhere, a failure where it did not run
    failure = err === nothing ? ErrorException("a task of the region failed on another rank") : err
    for c in plan.calls
        t = c.call.task
        isready(t.future) && continue
        if anyfail != 0 || err !== nothing
            fe = failure isa DTaskFailedException ? failure : region_failure(c.call, failure)
            put!(t.future, fe; error=true)
        else
            put!(t.future, TileRemoteResult(c.rank))
        end
    end
    # The tasks were never enqueued into the surrounding `wait_all`, so nobody else reports their failures: do what it does, in task order
    for c in plan.calls
        fetch(c.call.task; move_value=false, unwrap=false)
    end
    return true
end
