using Dagger, Test, Random
using Dagger: TileName, TileDirectory, LocalDirNetwork, DirEntry, tile_home, dir_serve!, dir_put!, dir_future, dir_set_ranks!

# Ranks are only identifiers here (3, 7, ... are as good as 0, 1, ...); the boxes of ranks that join later exist from the start.
function dir_cluster(all_ids, active)
    net = LocalDirNetwork(all_ids)
    ds = Dict{Int,TileDirectory}(i => TileDirectory(i, active, net) for i in active)
    tasks = Dict{Int,Task}(i => dir_serve!(ds[i], net) for i in active)
    return net, ds, tasks
end
function dir_close(net, tasks)
    foreach(close, values(net.boxes))
    foreach(wait, values(tasks))
end
function await(slot; timeout=20.0)
    t0 = time()
    while !isready(slot)
        time() - t0 > timeout && error("a directory request did not finish in $(timeout) s")
        sleep(0.001)
    end
    return fetch(slot)
end
# put, and return only once the home has stored it (a future for that version resolves only then)
announce!(ds, name, owner, version) = (dir_put!(ds[owner], name, owner, version); await(dir_future(ds[owner], name; minversion=version)))
total_sent(ds) = sum(d -> d.sent[], values(ds))
homes_of(ds, name) = [i for (i, d) in ds if haskey(d.entries, name)]

