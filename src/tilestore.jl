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
    if readers == 0
        notify(tv.done)                                           # nobody reads it: nothing to keep
    else
        lock(() -> d.data[(name, Int(version))] = tv, d.lock)
    end
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
    tile_fetch(d, name, version) -> DataSlot

Fetch version `version` of the tile `name`: wait until it is announced, ask its owner (nothing is sent when the owner is this rank) and deliver the value.
"""
function tile_fetch(d::TileDirectory, name::TileName, version::Integer)
    slot = DataSlot()
    es = dir_future(d, name; minversion=version)
    Threads.@spawn _tile_fetch_continue(d, name, Int(version), es, slot)
    return slot
end

function _tile_fetch_continue(d::TileDirectory, name::TileName, version::Int, es::DirSlot, slot::DataSlot)
    try
        e = fetch(es)
        if e.owner == d.me
            tv = lock(() -> _serve_version!(d, name, version), d.lock)
            tv === nothing ? _fail!(slot, ErrorException("tile $(name) version $(version) is not held by its owner, rank $(d.me)")) : _fill!(slot, tv.value)
        else
            reqid = lock(d.lock) do
                d.nextid += 1
                d.datapending[d.nextid] = slot
                d.nextid
            end
            _dir_deliver!(d, e.owner, DataGet(name, version, d.me, reqid))
        end
    catch err
        _fail!(slot, err isa Exception ? err : ErrorException(string(err)))
    end
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DataGet)
    reply = lock(d.lock) do
        tv = _serve_version!(d, msg.name, msg.version)
        tv === nothing ? DataMissing(msg.reqid, msg.name, msg.version, d.me) : DataReply(msg.reqid, msg.name, msg.version, tv.value)
    end
    _dir_deliver!(d, msg.from, reply)
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
