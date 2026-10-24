# Copied from CUDA.jl

struct LockedObject{T}
    lock::ReentrantLock
    payload::T
end
LockedObject(payload) = LockedObject(Threads.ReentrantLock(), payload)

function Base.lock(f, x::LockedObject)
    lock(x.lock)
    try
        return f(x.payload)
    finally
        unlock(x.lock)
    end
end
Base.lock(x::LockedObject) = lock(x.lock)
Base.trylock(x::LockedObject) = trylock(x.lock)
Base.unlock(x::LockedObject) = unlock(x.lock)
payload(x::LockedObject) = x.payload

# TODO: Move these back to MemPool
macro safe_lock1(l, o, ex)
    quote
        temp = $(esc(l))
        lock(temp)
        MemPool.enable_finalizers(false)
        try
            $(esc(o)) = $payload(temp)
            $(esc(ex))
        finally
            unlock(temp)
            MemPool.enable_finalizers(true)
        end
    end
end
# Take a `ReentrantLock` after a bounded spin on `trylock`, then park as usual. The scheduler's
# critical sections are 5-30 us, but a contended `lock` parks the waiter and the release has to
# wake it: on the Fulda Opterons a Datadeps Cholesky waited 50-260 us per task for `state.lock`
# while holding it for 20-30 us, so most of the wait was the hand-over. Spinning for about as long
# as a section lasts avoids the park for most waiters. Never unbounded (the holder may be
# descheduled), and never used where a finalizer may run (see `@safe_lock_spin1`). `spins` is
# `SCHED_LOCK_SPINS` (`JULIA_DAGGER_SCHED_LOCK_SPINS`, 0 = plain `lock`).
const SCHED_LOCK_SPINS = Ref{Int}(0)   # set from the environment in `__init__`
@inline function lock_brief!(l::ReentrantLock)
    for _ in 1:SCHED_LOCK_SPINS[]
        trylock(l) && return
        GC.safepoint()
    end
    lock(l)
    return
end
@inline function lock_brief(f, l::ReentrantLock)
    lock_brief!(l)
    try
        return f()
    finally
        unlock(l)
    end
end
# If we actually want to acquire a lock from a finalizer, we can't cause a task
# switch. As a NonReentrantLock can only be taken by another thread that should
# be running, and not a concurrent task we'd need to switch to, we can safely
# spin.
macro safe_lock_spin1(l, o, ex)
    quote
        temp = $(esc(l))
        while !trylock(temp)
            # we can't yield here
            GC.safepoint()
        end
        MemPool.enable_finalizers(false) # retains compatibility with non-finalizer callers
        try
            $(esc(o)) = $payload(temp)
            $(esc(ex))
        finally
            unlock(temp)
            MemPool.enable_finalizers(true)
        end
    end
end
