const ACCELERATION = TaskLocalValue{Acceleration}(() -> DistributedAcceleration())

current_acceleration() = ACCELERATION[]

default_processor(::DistributedAcceleration) = OSProc(myid())
default_processor(accel::DistributedAcceleration, x) = default_processor(accel)
default_processor() = default_processor(current_acceleration())

accelerate!(accel::Symbol) = accelerate!(Val{accel}())
accelerate!(::Val{:distributed}) = accelerate!(DistributedAcceleration())

function _with_default_acceleration(f)
    old_accel = ACCELERATION[]
    ACCELERATION[] = DistributedAcceleration()
    result = try
        f()
    finally
        ACCELERATION[] = old_accel
    end
    return result
end

# `do_task` runs on pooled/reusable tasks, so `ACCELERATION` (a TaskLocalValue)
# can carry over a previous task's value. Rather than wrapping the whole task
# body in a try/finally closure to restore it, each `do_task` sets it once up
# front: the value is always overwritten before anything reads it, and no other
# code runs on those pooled tasks, so a restore is unnecessary. `nothing` (an
# unstamped task) falls back to the Distributed default.
set_task_acceleration!(accel::Acceleration) = (ACCELERATION[] = accel; nothing)
set_task_acceleration!(::Nothing) = (ACCELERATION[] = DistributedAcceleration(); nothing)

initialize_acceleration!(a::DistributedAcceleration) = nothing
finalize_acceleration!(::Acceleration) = nothing

"Whether two accelerations represent the same installed backend (type + fields)."
same_acceleration(::Acceleration, ::Acceleration) = false
same_acceleration(a::T, b::T) where {T<:Acceleration} = a == b

function accelerate!(accel::Acceleration)
    old = ACCELERATION[]
    if !same_acceleration(old, accel)
        finalize_acceleration!(old)
        initialize_acceleration!(accel)
    end
    ACCELERATION[] = accel
end

accel_matches_proc(accel::DistributedAcceleration, proc::OSProc) = true
accel_matches_proc(accel::DistributedAcceleration, proc) = true

"""
    accel_kind(accel::Acceleration) -> Symbol

Maps an `Acceleration` instance back to the `Symbol` that `accelerate!` accepts
for it (e.g. `accelerate!(:mpi)` <-> `accel_kind(::MPIAcceleration) == :mpi`).
"""
accel_kind(::DistributedAcceleration) = :distributed

function compatible_processors(accel::Union{Acceleration,Nothing}, scope::AbstractScope, procs::Vector{<:Processor})
    comp = compatible_processors(scope, procs)
    accel === nothing && return comp
    # N.B. Explicitly Set{Processor}: `Set(gen)` infers a concrete element
    # type (e.g. Set{MPIProcessor{ThreadProc}} under MPI), which fails the
    # Set{Processor} typeassert in Sch's compatible-processor cache — and this
    # function's contract is `-> Set{Processor}`.
    return Set{Processor}(p for p in comp if accel_matches_proc(accel, p))
end
# accel_matches_proc is identically true for DistributedAcceleration; skip the
# second Set construction
compatible_processors(accel::DistributedAcceleration, scope::AbstractScope, procs::Vector{<:Processor}) =
    compatible_processors(scope, procs)
compatible_processors(accel::Union{Acceleration,Nothing}, scope::AbstractScope=get_compute_scope(), ctx::Context=Sch.eager_context()) =
    compatible_processors(accel, scope, procs(ctx))

uniform_execution(::DistributedAcceleration) = false
uniform_execution() = uniform_execution(current_acceleration())

"""
    with_acceleration(f, accel::Acceleration)

Run `f` with `accel` as the current acceleration of this task. The current acceleration
is task-local, so this affects only the tasks spawned within `f` (and what they spawn);
nothing process-global changes.
"""
function with_acceleration(f, accel::Acceleration)
    old = ACCELERATION[]
    ACCELERATION[] = accel
    try
        return f()
    finally
        ACCELERATION[] = old
    end
end

"""
    with_rank_local(f)

Run `f` so that every task spawned in it belongs to a task graph that only this rank plans
and runs (under MPI: a graph on a communicator of size one). Defined by the MPI extension;
the other accelerations have no such notion.
"""
function with_rank_local end

"""
    with_group(f, comm)

Run `f` so that every task spawned in it belongs to a task graph on the communicator `comm`
(a group of ranks): only its members plan and run these tasks, with their own task ids, and every
member runs the same program inside the block. Collective over `comm` the first time `comm` is
used. Defined by the MPI extension; see `free_group!`.
"""
function with_group end

"Release the bookkeeping of a group used with [`with_group`](@ref); collective over its members."
function free_group! end

"""
    per_rank(f, T)

SPMD operation (every rank calls it, from the driver task): every rank runs `f()` as a
rank-local graph (see [`with_rank_local`](@ref)) and the result, which must have type `T`,
is returned as a vector with one shared chunk per rank, chunk `r` owned by rank `r`, which
shared tasks can take as an argument. Defined by the MPI extension.
"""
function per_rank end

"""
    allgather(tasks) -> Vector

SPMD collective: wait for every task of `tasks` (the same list on every process, in the same
order) and return their values on *every* process, `[fetch(t) for t in tasks]`. Under MPI each
process contributes the values of the tasks whose result it owns and the processes exchange them
in ``\\lceil \\log_2 P \\rceil`` rounds, instead of moving every result to one process, gathering
them in a task and broadcasting that: measured at 26 / 52 / 104 ranks with 4 KB per rank, 15 / 23 /
138 ms for the gather task plus `fetch` against 0.6 / 1.3 / 2.8 ms for the exchange (an
`MPI.Allgather`). A task that failed makes every process throw its exception, after all have taken
part. It takes no task or reference ids, so it can be called wherever every process calls it.
Defined by the MPI extension; elsewhere it is `fetch` of each task.
"""
allgather(tasks::AbstractVector) = allgather(current_acceleration(), tasks)
allgather(::Acceleration, tasks::AbstractVector) = [fetch(t) for t in tasks]

"""
    allgather(tasks, T) -> Vector{T}

[`allgather`](@ref) with the element type given: the values arrive over the wire untyped, so the vector of
`allgather(tasks)` has the narrowest concrete element type its values allow (`Vector{Any}` for an empty list).
Pass `T` when the code that uses the result needs a fixed type, e.g. `allgather(summaries, PartitionSummary)`.
"""
allgather(tasks::AbstractVector, ::Type{T}) where {T} = convert(Vector{T}, allgather(tasks))

"""
    allgather_value(x; comm) -> Vector

SPMD collective for a plain value: every process contributes `x` (anything that serializes) and gets the vector of all
contributions in rank order. Under MPI it is a cooperative exchange: **no thread is parked inside MPI**. That is the
difference to `MPI.Allgather` from the driver, which holds an OS thread in a C call that is not a GC safepoint: a
collection triggered by another thread then waits for it for ever, freezing the whole rank (its relay forwards no
broadcast, so a peer that waits for one never reaches the collective): a deadlock across nodes, where such calls take
longer than in shared memory (the full MPI suite hung on two nodes because of it). `comm` selects another
communicator than the acceleration's. Elsewhere it is `[x]`.
"""
allgather_value(x; kwargs...) = allgather_value(current_acceleration(), x; kwargs...)
allgather_value(::Acceleration, x; kwargs...) = Any[x]

"""
    allreduce_value(op, x; init, comm) -> value

[`allgather_value`](@ref) then a fold with `op` in rank order on every process (identical bits everywhere, whatever the
arrival order), cooperatively. `allreduce_value(+, x)` is the sum of `x` over the processes.
"""
function allreduce_value(op, x; init=nothing, kwargs...)
    vals = allgather_value(x; kwargs...)
    return init === nothing ? reduce(op, vals) : reduce(op, vals; init)
end

"""
    mpi_rank_gate!(on=true; cap, reserve)

(MPI) Turn the rank gate on or off: a ready queue with a concurrency limit and priorities for the tasks a rank runs on its
free threads. Without it every ready task starts at once and Julia runs them first come first served, so a flood of long tasks
delays a small urgent task submitted after it by the whole backlog. With it at most `cap - reserve` ordinary tasks run at once
(`cap` defaults to the number of default threads, `reserve` to 1), `priority > 0` tasks (`Dagger.@spawn priority=1 f(x)`) may use
the reserved slot, and waiting tasks get slots by priority, then arrival. Also `JULIA_DAGGER_MPI_RANK_GATE=1`
(`..._RESERVE=k`). A task holds a slot only while its function runs; a task that blocks on another task of the same rank
inside its function can starve itself of slots, so give such waiters `priority=1`. Defined by the MPI extension.
"""
function mpi_rank_gate! end

"""
    barrier(; comm)

SPMD barrier that waits cooperatively (a non-blocking `MPI_Ibarrier` polled with `yield`): no thread is parked inside MPI
(see [`allgather_value`](@ref)). A no-op without MPI.
"""
barrier(; kwargs...) = barrier(current_acceleration(); kwargs...)
barrier(::Acceleration; kwargs...) = nothing

"""
    allreduce(op, tasks; init) -> value

SPMD collective: [`allgather`](@ref) the values of `tasks` and fold them with `op` **in list order**,
`op(op(op(v1, v2), v3), ...)` (starting from `init` when given), on every process. Every process folds the same
values in the same order, so the result is identical on all of them and independent of the order in which the
results arrived: a floating-point sum is bitwise reproducible. No MPI collective is called from the driver (see
`allgather`). Meant for one small value per task (a partial sum, a count, a bounding box); for large values
combine them inside tasks first.
"""
function allreduce(op, tasks::AbstractVector; kwargs...)
    return reduce(op, allgather(tasks); kwargs...)
end

"""
    per_group(f, comm, T)

[`per_rank`](@ref) on the communicator `comm` instead of the world: every member of `comm` calls it
(from the driver), runs `f()` as a rank-local graph and gets a vector with one chunk per member, chunk
`r` owned by the member with rank `r` in `comm` and usable as an argument of the tasks of the group
(see [`with_group`](@ref)). Defined by the MPI extension.
"""
function per_group end

"""
    fetch_failed_uniform!(accel)

Called when `fetch` of a task with a known owner (`DTask.owner`) throws because the
task failed on this process. Under uniform execution a fetch of a task that did
not fail takes part in a broadcast, which takes a task-id-derived tag on every
process; a process that fails before it gets there must take the same ids, or the
ids (and tags) of everything that follows differ across processes. Does nothing
by default.
"""
fetch_failed_uniform!(::Acceleration) = nothing

# SPMD uniformity checking. Under a uniform-execution acceleration (e.g. MPI),
# planning-time values are hashed and compared across ranks to catch divergence
# early. Core datadeps calls `check_uniform` unconditionally; the real checks
# live in MPIExt, so core owns the toggle plus a no-op fallback that always
# reports uniformity when no such extension is loaded.
const CHECK_UNIFORMITY = Ref{Bool}(false)
check_uniformity!(check::Bool=true) = (CHECK_UNIFORMITY[] = check)
# Generic fallback: when checking is off or execution isn't uniform (e.g. no
# MPIExt loaded), this is a no-op. Otherwise route through `hash` so MPIExt's
# `check_uniform(::Integer, ...)` performs the cross-rank comparison. MPIExt
# adds type-specific methods (more specific than this one) rather than
# overwriting it, which precompilation forbids.
check_uniform(value, original=value) =
    CHECK_UNIFORMITY[] && uniform_execution() ? check_uniform(hash(value), original) : true

"""
    @check_uniform(args...)

Hot-path wrapper for [`check_uniform`](@ref). Every `check_uniform` method (the
core fallback above, and every `check_uniform` MPIExt adds) short-circuits to
`true` when `CHECK_UNIFORMITY[]` is `false`, so testing that flag inline is
exactly equivalent to making the call -- but it avoids a dynamic call that would
box its (usually isbits) arguments on every invocation, and avoids evaluating
the argument expressions at all. `CHECK_UNIFORMITY[]` is set by
`check_uniformity!`, which is the only way any of those checks become active.
"""
macro check_uniform(args...)
    call = Expr(:call, GlobalRef(@__MODULE__, :check_uniform), map(esc, args)...)
    return quote
        $(GlobalRef(@__MODULE__, :CHECK_UNIFORMITY))[] ? $call : true
    end
end

# Scheduler hooks: Sch.jl calls these instead of MPI-specific symbols.
# Uniform-execution accelerations (e.g. MPI) inherit capacity/occupancy
# behavior via `uniform_execution`; move/cleanup hooks default to no-ops /
# async spawn and are overridden per backend where needed.
scheduling_ignore_capacity(accel::Acceleration) = uniform_execution(accel)

scheduling_task_occupancy(accel::Acceleration, est_occupancy::UInt32) =
    uniform_execution(accel) ? UInt32(0) : est_occupancy

# Region-end cleanup (wait_all): reclaim accel state for finished tasks in one
# batch instead of locking on every Sch finish_task!.
cleanup_task_accel!(::Acceleration, task) = nothing
cleanup_tasks_accel!(accel::Acceleration, tasks) =
    foreach(t -> cleanup_task_accel!(accel, t), tasks)

schedule_argument_move(::Acceleration, ::Integer, f::Function) = Threads.@spawn f()

"""
    sched_move(accel, to_proc, value) -> Any

Performs a scheduler argument move of `value` to `to_proc`. Only MPI needs the
`Sch.SCHED_MOVE` scoped value observed inside moves (to suppress tag creation);
constructing a ScopedValues scope (and the closure to enter it) per argument is
pure cost for every other acceleration, so the default is a direct `move` call
and MPIExt overrides this to install the scope.

N.B. No `invokelatest` needed: the executing task was itself entered via
`invokelatest`, so its dynamic calls already see every `move` method defined
before the task dispatched.
"""
sched_move(::Acceleration, to_proc, @nospecialize(value)) = move(to_proc, value)

"""
    argument_move_may_inline(accel, to_proc, value) -> Bool
    argument_move_may_inline(to_proc::Processor, value) -> Bool

Whether an argument move may run inline on the current task rather than being
spawned as a separate Task. The acceleration-level entry gates on
`uniform_execution` (MPI keeps its rank-ordered scheduling) and defers to the
2-argument, per-`Processor` form, which processor backends extend. Cheap local
moves — non-`Chunk` values moving to a CPU processor, `Chunk`s resident on
this worker moving to a CPU processor, or `Chunk`s already resident on the
destination GPU — gain nothing from concurrency, and spawning a Task per
argument dominates their cost. Remote `Chunk`s and cross-device copies stay
async so multiple transfers overlap; unknown processors default to async.
"""
argument_move_may_inline(accel::Acceleration, to_proc, @nospecialize(value)) =
    !uniform_execution(accel) && argument_move_may_inline(to_proc, value)
argument_move_may_inline(::Processor, @nospecialize(value)) = false
argument_move_may_inline(to_proc::ThreadProc, @nospecialize(value)) =
    !(value isa Chunk) || root_worker_id(value.handle) == myid()

"""
    bind_moved_argument(accel, original, moved) -> bound

After a scheduler argument move, choose the value stored on the Argument. The
default simply stores `moved` — the actual moved value — so a genuine `nothing`
(e.g. a `Chunk{Nothing}` from an ordinary task) reaches the kernel as `nothing`,
not as a `Chunk` the user never asked for.

Accelerations that use non-resident placeholders override this: under MPI a
`move` returning `nothing` on a non-owning rank means "data not local", so the
`Chunk` is kept to preserve SPMD-uniform type metadata (and `execute!` later
materializes it), and an owner-unwrapped `Chunk` is rewrapped to keep its
`chunktype` uniform across ranks.
"""
bind_moved_argument(::Acceleration, original, moved) = moved

"Logical execution rank of `proc` used for uniform task placement preference."
processor_owner_rank(accel::Acceleration, proc::Processor) = first(fire_order_key(proc))

"Deterministic ordering key for `proc` under uniform execution."
processor_order_key(accel::Acceleration, proc::Processor) =
    (fire_order_key(proc)..., short_name(proc))

"""
    partition_affinity_id(x) -> Int

Logical owner identity used by hierarchical Datadeps partitioning. Defaults to
`root_worker_id` (Distributed worker id). Accelerations with a different notion
of ownership (e.g. MPI ranks) override for their spaces/processors so affinity
partitioning spreads work across ranks the same way Distributed spreads across
workers.
"""
partition_affinity_id(space::MemorySpace) = root_worker_id(space)
partition_affinity_id(proc::Processor) = partition_affinity_id(only(memory_spaces(proc)))

"Preferred execution rank for `task`, or `nothing` if no preference."
task_processor_preference(accel::Acceleration, task, state) = nothing

"""
    sort_processors_canonical!(procs, accel)

Sort `procs` into the canonical order of `accel`: ascending
[`processor_order_key`](@ref), which begins with the processor's owner rank
(`processor_owner_rank`) and so groups each rank's processors contiguously.
The keys are computed once per processor: a comparison sort that rebuilds them
per comparison (each one formats a `short_name` string) costs `O(n log n)`
string constructions.
"""
function sort_processors_canonical!(procs, accel::Acceleration)
    length(procs) <= 1 && return procs
    keys = map(p -> processor_order_key(accel, p), procs)
    return permute!(procs, sortperm(keys))
end

"""
    preferred_processor_count(procs, accel, rank) -> Int

How many of the processors in canonically sorted `procs` are owned by `rank`. This is the helper of upstream PR #771 (which rotates
the preferred rank's processors by task id); this fork found the range by binary search (`preferred_processor_range`) when it
implemented the same placement, and keeps the upstream name so that code that probes for it, or calls it, works on both.
"""
function preferred_processor_count(procs, @nospecialize(accel), rank)
    lo, hi = preferred_processor_range(procs, accel, rank)
    return lo <= hi ? hi - lo + 1 : 0
end

# Index range of the processors owned by `rank` in canonically sorted `procs`
# (empty, `lo > hi`, when it owns none). Binary search: the canonical order is
# sorted by owner rank. The acceleration is passed through unspecialized, and
# the barrier kept out of line, so an isbits backend is boxed once per search
# instead of once per probed (abstractly typed) processor.
@noinline function preferred_processor_range(procs, @nospecialize(accel), rank)
    n = length(procs)
    lo, hi = 1, n + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if processor_owner_rank(accel, @inbounds procs[mid]) < rank
            lo = mid + 1
        else
            hi = mid
        end
    end
    first_idx = lo
    hi = n + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if processor_owner_rank(accel, @inbounds procs[mid]) <= rank
            lo = mid + 1
        else
            hi = mid
        end
    end
    return first_idx, lo - 1
end

"""
    place_processors_uniform!(procs, accel, task, state)

Given `procs` already in canonical order (see [`sort_processors_canonical!`](@ref)),
apply the per-task part of uniform placement: with a `task`, move the processors
of the rank the acceleration prefers for it (the owner of its argument data) to
the front, then rotate the candidates of that rank (or all of them when there is
no eligible preferred rank) by the task ID, so independent tasks spread across
processors without consulting rank-local load or ready order. Everything is an
in-place reversal: no scratch and no per-candidate allocation.
"""
function place_processors_uniform!(procs, accel::Acceleration, task, state)
    task === nothing && return procs
    n = length(procs)
    pref = task_processor_preference(accel, task, state)
    if pref !== nothing
        lo, hi = preferred_processor_range(procs, accel, pref)
        if lo <= hi
            if lo > 1
                # `[others][preferred]` -> `[preferred][others]`, both parts keep their order
                reverse!(procs, 1, hi)
                reverse!(procs, 1, hi - lo + 1)
                reverse!(procs, hi - lo + 2, hi)
            end
            n = hi - lo + 1
        end
    end
    if n > 1
        offset = mod(task.id - 1, n)
        # Rotate only the preferred prefix, in place and without scratch.
        reverse!(procs, 1, offset)
        reverse!(procs, offset + 1, n)
        reverse!(procs, 1, n)
    end
    return procs
end

"""
    select_processors_uniform!(procs, accel, task=nothing, state=nothing)

Re-sort `procs` with a deterministic, acceleration-dispatched key when
`uniform_execution(accel)` so every rank gets the same order (and with a
`task` the acceleration may prefer the rank owning task argument data).
Rotate the candidates within that preferred rank (or all candidates when
there is no eligible preferred rank) using the task ID, so independent tasks
spread across processors without consulting rank-local load or ready order.
Calls without a task retain the canonical order used by Datadeps and proc grids.
When not under uniform execution, leaves `procs` unchanged.

Callers that place many tasks over the same candidate set should keep the
canonical order (`sort_processors_canonical!`) and call
[`place_processors_uniform!`](@ref) per task instead.
"""
function select_processors_uniform!(procs, accel::Acceleration,
                                    task=nothing, state=nothing)
    uniform_execution(accel) || return procs
    sort_processors_canonical!(procs, accel)
    return place_processors_uniform!(procs, accel, task, state)
end

# Default processor grid for block assignment of new distributed arrays.
# Returns `nothing` to let the scheduler choose freely; uniform-execution
# accelerations (MPI) override this to produce a deterministic assignment
# that is identical on every rank.
default_procgrid(accel, nblocks::NTuple{N,Int}) where N = nothing

"Acceleration-specific finalization after array chunk tasks are spawned."
post_stage_array_chunks!(accel::Acceleration, tasks, eltype::Type, nd::Int) = tasks

"""
    note_chunk_placement!(accel, task::DTask, scope)

Called for every task that produces a block of an array, with the compute scope the block was
placed with (a single processor for the deterministic placements: every rank computes the same
grid). An acceleration that wants to know at spawn time where a block lives records it here;
the default does nothing.
"""
note_chunk_placement!(::Acceleration, task, scope) = nothing

"""
    elision_stats() -> NamedTuple or nothing

How many of the tasks this process planned were elided, run here, or planned in full and why
(`:elided`, `:own_task`, `:no_owner_known`, `:not_failure_only`, `:syncdeps_or_restore`,
`:return_type_not_concrete`, `:function_not_plain`, `:argument_not_positional`,
`:argument_participates`, `:scope_conflict`, `:not_applicable`), cumulative since the start or the
last [`reset_elision_stats!`](@ref). Only the accelerations that elide tasks (MPI) return anything.
"""
elision_stats() = elision_stats(current_acceleration())
elision_stats(::Acceleration) = nothing

"Reset the counters of [`elision_stats`](@ref)."
reset_elision_stats!() = reset_elision_stats!(current_acceleration())
reset_elision_stats!(::Acceleration) = nothing

"""
    enable_delegation!(; slots=max(Threads.nthreads(:default) - 1, 1))

(MPI) Let tasks spawned with `lend=true` have their execution lent to idle ranks. Collective: every
rank must call it at the same point, before the first lendable task (it creates a communicator for
the lending traffic). `slots` is the number of lendable tasks a rank runs at once; further lendable
tasks wait in a queue from which an idle rank may take them. The default leaves one thread of the rank
free: the lending service, the broadcast relay and the thread that plans tasks need one, and a rank
whose threads all run kernels that never yield cannot answer a request until one finishes. Nothing changes for tasks that do not
set `lend=true`. Other accelerations do not support it.
"""
enable_delegation!(; kwargs...) = enable_delegation!(current_acceleration(); kwargs...)
enable_delegation!(::Acceleration; kwargs...) = error("task delegation is only supported by the MPI acceleration")

"Undo [`enable_delegation!`](@ref) (collective as well)."
disable_delegation!() = disable_delegation!(current_acceleration())
disable_delegation!(::Acceleration) = nothing

"""
    delegation_stats() -> NamedTuple or nothing

Counters of this process for [`enable_delegation!`](@ref): `lent` (tasks of this rank whose execution
was lent out and whose answer has arrived), `failed` (of those, the ones that came back as a failure),
`returned` (claims a thief handed back unrun), `stolen` (tasks of other ranks run here), `local_`
(lendable tasks run on their own rank).
`nothing` when delegation is not enabled.
"""
delegation_stats() = delegation_stats(current_acceleration())
delegation_stats(::Acceleration) = nothing

default_processor(space::CPURAMMemorySpace) = OSProc(space.owner)
default_memory_space(accel::DistributedAcceleration) = CPURAMMemorySpace(myid())
# Chunk records must be labeled with the space the value actually resides in
# (e.g. GPU memory for device arrays)
default_memory_space(accel::DistributedAcceleration, x) = value_memory_space(x)
default_memory_space(x) = default_memory_space(current_acceleration(), x)
default_memory_space() = default_memory_space(current_acceleration())

# Whether two logical owners share a physical node (for device IPC fast paths).
# Defaults false; Distributed and MPI provide concrete methods.
same_node(::Acceleration, ::Integer, ::Integer) = false
