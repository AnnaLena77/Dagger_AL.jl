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
