# Self-scheduling parallel loop: one scheduler entry per worker instead of one per chunk.

# Hands out index chunks of `1:n` from the shared counter `next` until none are left
function _dynamic_for_worker(next::Threads.Atomic{Int}, f, n::Int, chunk::Int)
    while true
        lo = Threads.atomic_add!(next, chunk)
        lo >= n && break
        for i in (lo + 1):min(lo + chunk, n)
            f(i)
        end
    end
    return nothing
end

"""
    dynamic_for_async(f, n::Integer; chunk=nothing, workers=Threads.nthreads(), scope=nothing) -> Vector{DTask}

Start `f(i)` for every `i` in `1:n` as `workers` self-scheduling tasks and return them.
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
function dynamic_for_async(f, n::Integer; chunk::Union{Nothing,Integer}=nothing,
                           workers::Integer=Threads.nthreads(), scope=nothing)
    n = Int(n)
    workers = max(1, min(Int(workers), n))
    chunk = Int(something(chunk, max(1, cld(n, 4 * workers))))
    chunk >= 1 || throw(ArgumentError("chunk must be positive"))
    scope = something(scope, Dagger.scope(worker=myid()))
    next = Threads.Atomic{Int}(0)
    return DTask[Dagger.@spawn(scope=scope, _dynamic_for_worker(next, f, n, chunk)) for _ in 1:workers]
end

"""
    dynamic_for(f, n::Integer; kwargs...)

Run `f(i)` for every `i` in `1:n` and wait for all of it; rethrows the first failure.
See [`dynamic_for_async`](@ref).
"""
function dynamic_for(f, n::Integer; kwargs...)
    tasks = dynamic_for_async(f, n; kwargs...)
    foreach(fetch, tasks)
    return nothing
end
