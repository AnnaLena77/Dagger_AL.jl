# The distributed directory of exchangeable tiles (see src/tilenames.jl for the names and the home rule).
#
# Every rank runs a `TileDirectory`. The entry of a tile (where its current version lives and which version that is) is kept by the tile's *home*
# rank only, `tile_home(name, ranks)`; a rank that wants to know asks the home, so the cost of a lookup is one round trip (none when the home is the
# asking rank), and the table is spread over the ranks instead of replicated on every one. The protocol knows nothing about MPI: it needs `dir_send`
# (deliver a message to the directory of another rank, in order per pair) and a service that hands incoming messages to `dir_handle!`.
#
#   DirPut(name, entry)               the owner announces (or moves, or updates) a tile; a lower version never overwrites a higher one
#   DirGet(name, from, reqid, min)    "tell me the entry once its version is >= min": the home answers at once or remembers the asker (a pull future)
#   DirReply(reqid, name, entry)      the answer
# A message that reaches a rank that is no longer (or not yet) the home is forwarded to the home it computes itself, so a sender with an old view of the
# set of ranks is corrected on the way. A change of the rank set is a collective step for now (`dir_set_ranks!` on every rank at a point where no
# directory message is in flight); mid-flight changes need an epoch on the messages, which is the next step.

"Where a tile's current version is, and which version it is."
struct DirEntry
    owner::Int
    version::Int
end

struct DirPut
    name::TileName
    entry::DirEntry
    epoch::Int          # the sender's view of the set of ranks (see `dir_set_ranks!`)
end
struct DirGet
    name::TileName
    from::Int
    reqid::Int
    minversion::Int
    epoch::Int
end
struct DirReply
    reqid::Int
    name::TileName
    entry::DirEntry
end

abstract type DirTransport end
"`dir_send(tr, to, msg)`: deliver `msg` to the directory of rank `to` (asynchronously; messages of one pair arrive in order)."
function dir_send end

"A pull future: `wait` and `fetch` give the `DirEntry` once the home has answered."
mutable struct DirSlot
    const ev::Base.Event
    @atomic entry::Union{Nothing,DirEntry}
    DirSlot() = new(Base.Event(), nothing)
end
Base.wait(s::DirSlot) = (wait(s.ev); nothing)
Base.fetch(s::DirSlot) = (wait(s); (@atomic :acquire s.entry)::DirEntry)
Base.isready(s::DirSlot) = (@atomic :acquire s.entry) !== nothing

mutable struct TileDirectory
    const me::Int
    ranks::Vector{Int}                          # identifiers of the ranks that exist now (not positions)
    const tr::DirTransport
    const lock::ReentrantLock
    const entries::Dict{TileName,DirEntry}      # the tiles whose home is this rank
    const waiters::Dict{TileName,Vector{DirGet}}
    const pending::Dict{Int,DirSlot}            # requests this rank has sent and not had answered
    const cache::Dict{TileName,DirEntry}        # answers seen: a hint, the tile may have moved since
    nextid::Int
    const sent::Threads.Atomic{Int}             # messages that went over the transport (a lookup at the home sends none)
    const errors::Vector{Any}                   # exceptions of the handler (a failing handler must not vanish silently)
    epoch::Int                                  # the number of the current set of ranks; every change of the set is a new epoch
    const held::Vector{Any}                     # messages from a newer epoch than this rank has reached: handled once it gets there
    const data::Dict{Tuple{TileName,Int},Any}   # the versions of tiles this rank owns and that readers may still fetch (src/tilestore.jl)
    const datapending::Dict{Int,Any}            # data requests this rank has sent and not had answered
    const moving::Dict{Tuple{TileName,Int},Tuple{Int,Vector{Any}}}   # versions on their way to another rank: (target, requests that arrived meanwhile)
    const moved::Dict{Tuple{TileName,Int},Int}  # versions that left this rank: where they went (a stale reader is sent on)
    const latest::Dict{TileName,Int}            # the highest version of each tile this rank has published
    const dwait::Dict{Tuple{TileName,Int},Vector{Any}}   # data requests for a version that is not published yet (a reader that knows the owner asks it directly)
    TileDirectory(me::Integer, ranks, tr::DirTransport) =
        new(Int(me), sort!(collect(Int, ranks)), tr, ReentrantLock(), Dict{TileName,DirEntry}(), Dict{TileName,Vector{DirGet}}(),
            Dict{Int,DirSlot}(), Dict{TileName,DirEntry}(), 0, Threads.Atomic{Int}(0), Any[], 0, Any[],
            Dict{Tuple{TileName,Int},Any}(), Dict{Int,Any}(),
            Dict{Tuple{TileName,Int},Tuple{Int,Vector{Any}}}(), Dict{Tuple{TileName,Int},Int}(),
            Dict{TileName,Int}(), Dict{Tuple{TileName,Int},Vector{Any}}())
end

@inline _home(d::TileDirectory, name::TileName) = tile_home(name, d.ranks)

# Send after the lock is released (a transport may block or call back); a message to oneself is handled directly.
function _dir_deliver!(d::TileDirectory, to::Int, msg)
    if to == d.me
        dir_handle!(d, msg)
    else
        Threads.atomic_add!(d.sent, 1)
        dir_send(d.tr, to, msg)
    end
    return nothing
end

"Handle one incoming message. Called by the service of the transport for every message, and directly for messages to oneself."
function dir_handle!(d::TileDirectory, msg::DirPut)
    out = Tuple{Int,Any}[]
    lock(d.lock) do
        if msg.epoch > d.epoch                                   # the sender knows a newer set of ranks than this rank: wait until it has the same
            push!(d.held, msg)
            return
        end
        h = _home(d, msg.name)
        if h != d.me
            push!(out, (h, DirPut(msg.name, msg.entry, d.epoch)))   # not (or no longer) the home: pass it on, in this rank's epoch
            return
        end
        old = get(d.entries, msg.name, nothing)
        (old === nothing || msg.entry.version >= old.version) || return
        d.entries[msg.name] = msg.entry
        ws = get(d.waiters, msg.name, nothing)
        ws === nothing && return
        keep = DirGet[]
        for w in ws
            if msg.entry.version >= w.minversion
                push!(out, (w.from, DirReply(w.reqid, w.name, msg.entry)))
            else
                push!(keep, w)
            end
        end
        isempty(keep) ? delete!(d.waiters, msg.name) : (d.waiters[msg.name] = keep)
    end
    for (to, m) in out; _dir_deliver!(d, to, m); end
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DirGet)
    out = Tuple{Int,Any}[]
    lock(d.lock) do
        if msg.epoch > d.epoch
            push!(d.held, msg)
            return
        end
        h = _home(d, msg.name)
        if h != d.me
            push!(out, (h, DirGet(msg.name, msg.from, msg.reqid, msg.minversion, d.epoch)))
            return
        end
        e = get(d.entries, msg.name, nothing)
        if e !== nothing && e.version >= msg.minversion
            push!(out, (msg.from, DirReply(msg.reqid, msg.name, e)))
        else
            push!(get!(Vector{DirGet}, d.waiters, msg.name), msg)   # a pull future: answered when the owner announces that version
        end
    end
    for (to, m) in out; _dir_deliver!(d, to, m); end
    return nothing
end

function dir_handle!(d::TileDirectory, msg::DirReply)
    slot = lock(d.lock) do
        d.cache[msg.name] = msg.entry
        pop!(d.pending, msg.reqid, nothing)
    end
    if slot !== nothing
        @atomic :release slot.entry = msg.entry
        notify(slot.ev)
    end
    return nothing
end

"The owner announces tile `name` at `version` on rank `owner` (a new tile, a new version, or a move)."
function dir_put!(d::TileDirectory, name::TileName, owner::Integer, version::Integer)
    home, ep = lock(() -> (_home(d, name), d.epoch), d.lock)
    _dir_deliver!(d, home, DirPut(name, DirEntry(Int(owner), Int(version)), ep))
end

"""
    dir_future(d, name; minversion=0) -> DirSlot

Ask for the entry of `name` once its version is at least `minversion`. Resolved at once from the cache when it already holds such an entry.
"""
function dir_future(d::TileDirectory, name::TileName; minversion::Integer=0)
    slot = DirSlot()
    home = 0; reqid = 0; ep = 0
    hit = lock(d.lock) do
        c = get(d.cache, name, nothing)
        if c !== nothing && c.version >= minversion
            return c
        end
        d.nextid += 1; reqid = d.nextid
        d.pending[reqid] = slot
        home = _home(d, name); ep = d.epoch
        return nothing
    end
    if hit !== nothing
        @atomic :release slot.entry = hit
        notify(slot.ev)
        return slot
    end
    _dir_deliver!(d, home, DirGet(name, d.me, reqid, Int(minversion), ep))
    return slot
end

"""
    dir_set_ranks!(d, ranks; epoch=d.epoch + 1)

Change the set of ranks (identifiers) to `ranks`, as epoch `epoch`. Each rank calls it when it learns of the new set; the ranks need not do so at the same time.
The entries and waiting requests whose home is now another rank are handed to it, in the new epoch. A message from a newer epoch that reaches a rank which has not
changed yet waits there (`held`) and is handled when it has, so two ranks with different views never pass a message back and forth. A rank that leaves must keep
running until the others have changed too.
"""
function dir_set_ranks!(d::TileDirectory, ranks; epoch::Integer=d.epoch + 1)
    out = Tuple{Int,Any}[]
    ready = Any[]
    lock(d.lock) do
        epoch > d.epoch || throw(ArgumentError("a new set of ranks needs a higher epoch than $(d.epoch)"))
        d.ranks = sort!(collect(Int, ranks))
        d.epoch = Int(epoch)
        for (name, e) in collect(d.entries)
            h = _home(d, name)
            h == d.me && continue
            push!(out, (h, DirPut(name, e, d.epoch))); delete!(d.entries, name)
        end
        for (name, ws) in collect(d.waiters)
            h = _home(d, name)
            h == d.me && continue
            for w in ws; push!(out, (h, DirGet(w.name, w.from, w.reqid, w.minversion, d.epoch))); end
            delete!(d.waiters, name)
        end
        keep = Any[]
        for m in d.held
            if m.epoch <= d.epoch; push!(ready, m); else push!(keep, m); end
        end
        empty!(d.held); append!(d.held, keep)
    end
    for (to, m) in out; _dir_deliver!(d, to, m); end
    for m in ready; dir_handle!(d, m); end                 # messages that were waiting for this epoch
    return nothing
end

# --- an in-process network: many directories in one process, one mailbox per rank (for tests and for single-process use) ---------------------------

struct LocalDirNetwork <: DirTransport
    boxes::Dict{Int,Channel{Any}}
    LocalDirNetwork(ids) = new(Dict{Int,Channel{Any}}(Int(i) => Channel{Any}(Inf) for i in ids))
end
dir_send(n::LocalDirNetwork, to::Int, msg) = (put!(n.boxes[to], msg); nothing)

"Start the service of a directory on a local network: a task that hands every message of the rank's mailbox to `dir_handle!` until the mailbox is closed."
function dir_serve!(d::TileDirectory, net::LocalDirNetwork)
    box = net.boxes[d.me]
    return Threads.@spawn begin
        for msg in box
            try
                dir_handle!(d, msg)
            catch e
                lock(() -> push!(d.errors, CapturedException(e, catch_backtrace())), d.lock)
            end
        end
    end
end

# --- the directory of this process --------------------------------------------------------------------------------------------------------------

const LOCAL_DIRECTORY = Ref{Union{Nothing,TileDirectory}}(nothing)
const LOCAL_DIRECTORY_LOCK = ReentrantLock()

"""
    tile_directory(accel=current_acceleration()) -> TileDirectory

The directory of this process for an acceleration. Without MPI it is a directory of one rank (every tile is homed here); `MPIExt` adds the one that talks to the
other ranks. The first call under MPI duplicates the communicator, which is collective: make it on every rank at the same point of the program.
"""
function tile_directory(accel=current_acceleration())
    lock(LOCAL_DIRECTORY_LOCK) do
        d = LOCAL_DIRECTORY[]
        if d === nothing
            net = LocalDirNetwork([0])
            d = TileDirectory(0, [0], net)
            dir_serve!(d, net)
            LOCAL_DIRECTORY[] = d
        end
        return d
    end
end
