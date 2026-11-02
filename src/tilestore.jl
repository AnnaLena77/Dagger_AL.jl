# The data of exchangeable tiles (see src/tilenames.jl and src/tiledir.jl for the names and the directory).
#
# A tile is a sequence of immutable *versions*. The owner of a tile publishes version `v` (`tile_publish!`: store the value, then announce it in the directory,
# so a reader that learns of the version can always fetch it), and a reader fetches it with `tile_fetch` (a pull: the directory says who owns the tile, the
# owner sends the value). Values are never changed after publication, so a version may be read by any number of ranks while the owner computes the next one.
#
# Lifetime: the owner knows how many readers a version will have (`readers`: the number of tasks that use it, known from the task graph). It keeps the version
# until it has served that many fetches, then drops it and notifies `done`; the task that produces the next version waits for `done` (the anti-dependency).
# `readers = -1` keeps a version until `tile_drop!`. A reader that asks for a version the owner does not hold gets an error, which signals a violated
# anti-dependency (or a tile that moved: migration will hand the pending versions over).
#
# Values are shared, not copied, inside one process; over a network they are serialized, so a receiver may not modify a fetched value.

"One version of a tile at its owner: the value, how many fetches it still has to serve (< 0: until dropped) and the event that says it is consumed."
mutable struct TileVersion
    const value::Any
    remaining::Int
    const done::Base.Event
end

struct DataGet
    name::TileName
    version::Int
    from::Int
    reqid::Int
end
struct DataReply
    reqid::Int
    name::TileName
    version::Int
    value::Any
end
"The version left this rank: ask `owner`."
struct DataMoved
    reqid::Int
    name::TileName
    version::Int
    owner::Int
end
"A version moves to the receiver, with the number of fetches it still has to serve."
struct DataMove
    name::TileName
    version::Int
    value::Any
    remaining::Int
    from::Int
end
struct DataMoveAck
    name::TileName
    version::Int
end

struct DataMissing
    reqid::Int
    name::TileName
    version::Int
    owner::Int
end

"A pull future for a tile's value: `wait` / `fetch` give it, or throw why it could not be fetched."
mutable struct DataSlot
    const ev::Base.Event
    @atomic value::Any
    @atomic err::Union{Nothing,Exception}
    DataSlot() = new(Base.Event(), nothing, nothing)
end
Base.wait(s::DataSlot) = (wait(s.ev); nothing)
Base.isready(s::DataSlot) = (@atomic :acquire s.err) !== nothing || (@atomic :acquire s.value) !== nothing
function Base.fetch(s::DataSlot)
    wait(s)
    e = @atomic :acquire s.err
    e === nothing || throw(e)
    return @atomic :acquire s.value
end
_fill!(s::DataSlot, v) = (@atomic :release s.value = v; notify(s.ev); nothing)
_fail!(s::DataSlot, e::Exception) = (@atomic :release s.err = e; notify(s.ev); nothing)

# one fetch served (the lock is held): returns the value and drops the version when it was the last
function _serve_version!(d::TileDirectory, name::TileName, version::Int)
    tv = get(d.data, (name, version), nothing)
    tv === nothing && return nothing
    if tv.remaining > 0
        tv.remaining -= 1
        if tv.remaining == 0
            delete!(d.data, (name, version)); notify(tv.done)
        end
    end
    return tv
end

"""
    tile_publish!(d, name, version, value; readers=-1) -> TileVersion

Publish `value` as version `version` of the tile `name`, owned by this rank. The value is stored first and announced second. The returned version's `done` event
is notified when `readers` fetches have been served (at once when `readers == 0`).
"""
function tile_publish!(d::TileDirectory, name::TileName, version::Integer, value; readers::Integer=-1)
    tv = TileVersion(value, Int(readers), Base.Event())
    key = (name, Int(version))
    replies = Tuple{Int,Any}[]
    lock(d.lock) do
        d.latest[name] = max(get(d.latest, name, 0), Int(version))
        if readers == 0
            notify(tv.done)                                       # nobody reads it: nothing to keep
        else
            d.data[key] = tv
        end
        ws = pop!(d.dwait, key, nothing)                          # readers that asked this owner before the version existed
        if ws !== nothing
            for g in ws
                sv = _serve_version!(d, name, Int(version))
                push!(replies, (g.from, sv === nothing ? DataMissing(g.reqid, name, Int(version), d.me) : DataReply(g.reqid, name, Int(version), sv.value)))
            end
        end
    end
    for (to, m) in replies; _dir_deliver!(d, to, m); end
    dir_put!(d, name, d.me, version)
    return tv
end

"Drop a version that was published with `readers = -1` (or whose readers will not come)."
function tile_drop!(d::TileDirectory, name::TileName, version::Integer)
    tv = lock(() -> pop!(d.data, (name, Int(version)), nothing), d.lock)
    tv === nothing || notify(tv.done)
    return nothing
end

"""
    tile_fetch(d, name, version; owner=nothing) -> DataSlot

Fetch version `version` of the tile `name`: wait until it is announced, ask its owner (nothing is sent when the owner is this rank) and deliver the value. With
`owner` given the directory is not asked at all.
"""
function tile_fetch(d::TileDirectory, name::TileName, version::Integer; owner::Union{Nothing,Integer}=nothing)
    slot = DataSlot()
    if owner !== nothing
        # The caller knows the owner (a fixed layout, as in HPX's partitioned vectors): no lookup, one round trip. The owner holds the request until it has
        # published the version, and forwards it if the tile has moved. A wrong owner that never publishes this tile leaves the request waiting.
        reqid = lock(d.lock) do
            d.nextid += 1
            d.datapending[d.nextid] = slot
            d.nextid
        end
        _dir_deliver!(d, Int(owner), DataGet(name, Int(version), d.me, reqid))
        return slot
    end
    es = dir_future(d, name; minversion=version)
    Threads.@spawn _tile_fetch_continue(d, name, Int(version), es, slot)
    return slot
end

function _tile_fetch_continue(d::TileDirectory, name::TileName, version::Int, es::DirSlot, slot::DataSlot)
    try
        e = fetch(es)
        # also when the owner is this rank: the request goes through the same handler as a remote one (a message to oneself is handled directly), so it finds
        # a version that is being handed over (it waits) or has left (it is sent on) exactly as a remote reader does
        reqid = lock(d.lock) do
            d.nextid += 1
            d.datapending[d.nextid] = slot
            d.nextid
        end
        _dir_deliver!(d, e.owner, DataGet(name, version, d.me, reqid))
    catch err
        _fail!(slot, err isa Exception ? err : ErrorException(string(err)))
    end
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataGet)
    key = (msg.name, msg.version)
    reply = lock(d.lock) do
        mv = get(d.moving, key, nothing)
        if mv !== nothing
            push!(mv[2], msg)                                     # the version is being handed over: answer when the new owner has confirmed
            return nothing
        end
        to = get(d.moved, key, nothing)
        to === nothing || return DataMoved(msg.reqid, msg.name, msg.version, to)
        tv = _serve_version!(d, msg.name, msg.version)
        if tv === nothing && msg.version > get(d.latest, msg.name, 0)
            push!(get!(Vector{Any}, d.dwait, key), msg)           # not published yet: answered when it is
            return nothing
        end
        tv === nothing ? DataMissing(msg.reqid, msg.name, msg.version, d.me) : DataReply(msg.reqid, msg.name, msg.version, tv.value)
    end
    reply === nothing || _dir_deliver!(d, msg.from, reply)
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataReply)
    slot = lock(() -> pop!(d.datapending, msg.reqid, nothing), d.lock)
    slot === nothing || _fill!(slot, msg.value)
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataMissing)
    slot = lock(() -> pop!(d.datapending, msg.reqid, nothing), d.lock)
    slot === nothing || _fail!(slot, ErrorException("tile $(msg.name) version $(msg.version) is not held by rank $(msg.owner)"))
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataMoved)
    slot = lock(() -> get(d.datapending, msg.reqid, nothing), d.lock)   # the request stays pending: it is asked again, at the new owner
    slot === nothing || _dir_deliver!(d, msg.owner, DataGet(msg.name, msg.version, d.me, msg.reqid))
    return nothing
end

"""
    tile_migrate!(d, name, version, to)

Move version `version` of the tile `name`, which this rank owns, to rank `to` together with the number of fetches it still has to serve. The new owner stores it,
announces itself in the directory and confirms; requests that reach this rank meanwhile wait and are then sent on to the new owner, and requests that arrive later
are sent on too. The `done` event of the version here is notified at once ("released here"): the readers that are left are served by the new owner.
"""
function tile_migrate!(d::TileDirectory, name::TileName, version::Integer, to::Integer)
    key = (name, Int(version))
    to == d.me && return nothing
    tv = lock(d.lock) do
        tv = pop!(d.data, key, nothing)
        tv === nothing && throw(ErrorException("tile $(name) version $(version) is not held by rank $(d.me): cannot migrate it"))
        d.moving[key] = (Int(to), Any[])
        tv
    end
    notify(tv.done)
    _dir_deliver!(d, Int(to), DataMove(name, Int(version), tv.value, tv.remaining, d.me))
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataMove)
    key = (msg.name, msg.version)
    lock(d.lock) do
        d.data[key] = TileVersion(msg.value, msg.remaining, Base.Event())    # stored before it is announced
        delete!(d.moved, key)
    end
    dir_put!(d, msg.name, d.me, msg.version)
    _dir_deliver!(d, msg.from, DataMoveAck(msg.name, msg.version))
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataMoveAck)
    key = (msg.name, msg.version)
    to, waiting = lock(d.lock) do
        to, queued = pop!(d.moving, key)
        d.moved[key] = to
        (to, queued)
    end
    for g in waiting
        _dir_deliver!(d, g.from, DataMoved(g.reqid, g.name, g.version, to))
    end
    return nothing
end
