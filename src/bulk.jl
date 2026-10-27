# Self-scheduling parallel loop: one scheduler entry per worker instead of one per chunk.

# Hands out index chunks of `1:n` from the shared counter `next` until none are left
function _dynamic_for_worker(next::Threads.Atomic{Int}, f, n::Int, chunk::Int, args...)
    taken = 0
    while true
        lo = Threads.atomic_add!(next, chunk)
        lo >= n && break
        hi = min(lo + chunk, n)
        for i in (lo + 1):hi
            f(i, args...)
        end
        taken += hi - lo
    end
    return taken
end

"""
    dynamic_for_async(f, n::Integer, args...; chunk=nothing, workers=Threads.nthreads(), scope=nothing) -> Vector{DTask}

Start `f(i, args...)` for every `i` in `1:n` as `workers` self-scheduling tasks and return them.
The result of each task is the number of indices that worker took, so `fetch.(tasks)` shows how
the loop was balanced (the entries add up to `n`).
`args` are ordinary task arguments: a `DTask` or a `Chunk` among them is waited for and
resolved before any worker starts, exactly as for `Dagger.@spawn`, so the loop starts when
its inputs exist and the caller need not wait for them (`f(i)` when there are none).
Each worker takes the next `chunk` indices from a shared atomic counter until none are
left, so the scheduler is entered `workers` times, however many chunks there are; with a
task per chunk, a loop of cheap iterations is dominated by the cost of a task (about
100 us). `chunk` defaults to `cld(n, 4workers)`. Workers that find the counter exhausted
return at once, so asking for more workers than free threads is harmless.

All workers must run in one process (they share the counter): the default scope is the
current worker's processors; under MPI pass a scope inside one rank, e.g.
`scope=Dagger.scope(mpi_rank=r)`. `f` must be safe to call from several threads at
once; the order of iterations is unspecified.
"""
function dynamic_for_async(f, n::Integer, args...; chunk::Union{Nothing,Integer}=nothing,
                           workers::Integer=Threads.nthreads(), scope=nothing)
    n = Int(n)
    workers = max(1, min(Int(workers), n))
    chunk = Int(something(chunk, max(1, cld(n, 4 * workers))))
    chunk >= 1 || throw(ArgumentError("chunk must be positive"))
    scope = something(scope, Dagger.scope(worker=myid()))
    next = Threads.Atomic{Int}(0)
    return DTask[Dagger.@spawn(scope=scope, _dynamic_for_worker(next, f, n, chunk, args...)) for _ in 1:workers]
end

"""
    dynamic_for(f, n::Integer, args...; kwargs...)

Run `f(i, args...)` for every `i` in `1:n` and wait for all of it; rethrows the first failure.
Returns the number of indices each worker took (see [`dynamic_for_async`](@ref)), whose sum is `n`. Called from inside a task, this blocks that task until the
workers are done and the task keeps its processor meanwhile; from a task, prefer
`dynamic_for_async` and hand the returned tasks to a follow-up task as arguments.
"""
function dynamic_for(f, n::Integer, args...; kwargs...)
    tasks = dynamic_for_async(f, n, args...; kwargs...)
    return Int[fetch(t) for t in tasks]
end

# Folds the values of one chunk of indices in index order; the chunk's partial result goes to `parts[c]`
function _dynamic_reduce_worker(next::Threads.Atomic{Int}, parts::Vector{Any}, f, op, n::Int, chunk::Int, nchunks::Int, args...)
    while true
        c = Threads.atomic_add!(next, 1) + 1
        c > nchunks && break
        lo = (c - 1) * chunk
        hi = min(lo + chunk, n)
        acc = f(lo + 1, args...)
        for i in (lo + 2):hi
            acc = op(acc, f(i, args...))
        end
        parts[c] = acc
    end
    return nothing
end

"""
    dynamic_reduce_async(f, op, n::Integer, args...; init, chunk=nothing, workers=Threads.nthreads(), scope=nothing) -> Task-like

Reduce `f(i, args...)` over `i in 1:n` with `op`, on `workers` self-scheduling tasks like
[`dynamic_for_async`](@ref); returns a `DTask` whose value is the result. **The combination order is fixed**: the
indices are cut into chunks of `chunk` (default `cld(n, 64)`, independent of `workers`), each chunk is folded in index
order, and the chunk results are folded in chunk order starting from `init`. So the result does not depend on which
worker took which chunk, on the number of workers, or on timing: a floating-point sum is bitwise the same on every
run (and on every rank that computes it). It does depend on `chunk`, as any reduction tree does: keep it fixed
when states are compared byte for byte. `op` must be associative in the sense that the result is acceptable (it is
not required to be commutative); `n == 0` returns `init` (an `ArgumentError` without one). `f` must be thread-safe.
"""
function dynamic_reduce_async(f, op, n::Integer, args...; init=nothing, chunk::Union{Nothing,Integer}=nothing,
                              workers::Integer=Threads.nthreads(), scope=nothing)
    n = Int(n)
    if n == 0
        init === nothing && throw(ArgumentError("dynamic_reduce of an empty range needs `init`"))
        return Dagger.@spawn identity(init)
    end
    chunk = Int(something(chunk, max(1, cld(n, 64))))
    chunk >= 1 || throw(ArgumentError("chunk must be positive"))
    nchunks = cld(n, chunk)
    workers = max(1, min(Int(workers), nchunks))
    scope = something(scope, Dagger.scope(worker=myid()))
    next = Threads.Atomic{Int}(0)
    parts = Vector{Any}(undef, nchunks)
    tasks = DTask[Dagger.@spawn(scope=scope, _dynamic_reduce_worker(next, parts, f, op, n, chunk, nchunks, args...)) for _ in 1:workers]
    return Dagger.@spawn(scope=scope, _dynamic_reduce_finish(op, parts, init, tasks...))
end
# The tasks are arguments only to order this after them
function _dynamic_reduce_finish(op, parts::Vector{Any}, init, _workers...)
    acc = init === nothing ? parts[1] : op(init, parts[1])
    for c in 2:length(parts)
        acc = op(acc, parts[c])
    end
    return acc
end

"""
    dynamic_reduce(f, op, n::Integer, args...; kwargs...)

[`dynamic_reduce_async`](@ref) and wait for the result.
"""
dynamic_reduce(f, op, n::Integer, args...; kwargs...) = fetch(dynamic_reduce_async(f, op, n, args...; kwargs...))

