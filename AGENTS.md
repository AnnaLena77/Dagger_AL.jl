# Guidance for AI agents working on Dagger.jl

Lessons learned the hard way while working on this codebase. Follow them, and
when you learn a new hard-won truth of your own, record it here as a new
lesson.

## Lessons

1. **Record hard-won truths here.** When you discover something non-obvious
   about this codebase — a subtle invariant, a lifecycle rule, a performance
   cliff — add it to this file as a new lesson so the next agent doesn't have
   to rediscover it.

2. **Make tight commits.** One separable change per commit, with a commit
   message that explains the *why* (the cost being removed, the invariant
   being preserved). Tight commits make review and rebasing far easier.

3. **Avoid allocations when they're easily avoided.** Steady-state code paths
   (scheduling, planning, argument moves, task teardown) run per-task or
   per-argument; a single stray `Set`, closure capture, or splat there
   multiplies by the task count. Prefer reuse (pools, scratch buffers on
   long-lived state, `@reusable_vector`) and plain loops over closure
   pipelines. The `test/allocations.jl` suite enforces upper bounds — keep it
   green, and re-calibrate its bounds only for intentional shifts.

4. **Profile memory before and after.** When a change plausibly affects
   allocation behavior, measure it: deep warmup (~10 iterations plus `GC.gc()`),
   then compare `Base.gc_num()` deltas (count and bytes) over several runs,
   taking the minimum. Cite the numbers in the commit message.

5. **Consider type-stability in every change.** Check that hot-path locals
   infer concretely (`@code_warntype`, Cthulhu, or `--track-allocation`), that
   captured-and-reassigned variables aren't forcing `Core.Box`, and that
   containers have concrete element types.

6. **Use `@nospecialize` only on explicitly type-unstable paths** where the
   value does not need to be type-inspected by the compiler (e.g. it is only
   stored, passed through, or checked with runtime `isa`). Never use it where
   inference of the value's type feeds later dispatch or arithmetic — that
   just moves the dynamic dispatch somewhere less visible.

7. **Long-lived and pooled tasks must not inherit their creator's dynamic
   scope.** Julia copies `current_task().scope` (the `ScopedValues` chain) into
   every new `Task`. A task pool, a processor runner, or the eager scheduler
   task is created lazily by whichever call first needed it — often inside a
   `Dagger.with_options(...)` block — and then serves *every* later caller, so
   an inherited scope silently applies those options to unrelated work for the
   rest of the session. Clear it with `clear_task_scope!` before starting such
   a task (it can only be set before the task starts). Per-task option
   propagation is explicit (`get_propagated_options`), so these tasks want no
   ambient scope at all. Note that `task.scope` only exists on Julia 1.11+; on
   1.10 the `ScopedValues.jl` compat package hides the scope in `task.logstate`,
   so `clear_task_scope!` is version-split — always go through it rather than
   writing the field directly. The same reasoning applies to any other
   creation-time-captured state (e.g. `TaskLocalValue`s) on a reused task.

8. **A `Thunk`'s input slots hold *weak* references, so every path that skips
   creating a dependents edge must resolve the slot itself.** Input slots are
   normally resolved by walking the dependents edges
   (`schedule_dependents!` → `resolve_finished_input!`) when an upstream
   finishes. An upstream that is *already* finished gets no edge — there is
   nothing left to wait on — so nothing will ever resolve its slot, and the
   consumer keeps only a `WeakThunk`. With Thunk pooling that reference dies
   deterministically and immediately: the upstream is recycled and handed back
   out with a fresh id, so `unwrap_weak` returns `nothing` and
   `unwrap_weak_checked` asserts. Resolve eagerly, while the upstream is still
   alive and holding its result. The same applies to any future lifecycle
   shortcut: if you stop registering an edge, you have taken on the job that
   edge was doing.

9. **An exception on a pooled or detached task is a hang, not a crash.**
   Scheduler work runs on `ReusableTaskCache` tasks (whose loop only `@error`s
   a failed payload and moves on) and on bare `Threads.@spawn` (whose exception
   nobody fetches). Anything that has already been credited to
   `running_count` — or that some `fetch` is waiting on — is lost silently if
   an error escapes there, and the symptom is a session that hangs with one
   logged error, which is far harder to diagnose than a crash. Wrap such work
   so a failure becomes a *failed thunk* (`set_failed!` plus the matching
   `running_count` release), and attach the backtrace with `CapturedException`
   so the waiter sees where it actually broke.

10. **In a lock-free handshake, publish last and release last.** Where two
   sides coordinate through a single atomic counter (`ProcessRingBuffer`'s
   `count`), that counter is a *permission grant*, not a bookkeeping detail.
   The producer must fill the slot before incrementing (an incremented count
   entitles the consumer to read it) and the consumer must read the value out
   before decrementing (a decremented count entitles the producer to overwrite
   it). Getting the order wrong is invisible to assertions that only check
   counts and ranges — both sides stay perfectly self-consistent while values
   are silently lost or duplicated — and it only bites when the buffer is at a
   boundary, i.e. exactly under the backpressure the buffer exists to provide.
   Test such a structure by driving it from two real threads with a
   deliberately tiny capacity and checking the *sequence*, not the counts.

11. **Never iterate a shared collection across a point where you drop the
   lock.** Blocking calls in this codebase release and reacquire their lock
   (`wait(store.lock)`, `@lock`-guarded condition waits), and anything else may
   mutate the collection in that gap. Julia will not warn you: `Dict` iteration
   has no modification check, so an insert that triggers a rehash mid-iteration
   silently *revisits* some entries and *skips* others — measured, not
   theoretical. The revisit is the dangerous half, because re-processing an
   entry you already handled can block you on a resource you just consumed and
   hang the loop forever. Snapshot the keys into a `@reusable_vector` before
   the loop (steady-state allocation-free) and re-check membership per
   iteration; the resulting "entries added while we waited are picked up next
   round" semantics is well-defined, which the accidental version was not.

12. **Never wait by spinning on `yield()`.** Julia permanently marks a task
   `sticky` the moment it schedules any sticky task — an `@async`, which plenty
   of library code (Distributed's transport included) does on your behalf.
   `Base.enq_work` says so itself: *"XXX: Ideally we would be able to unset
   this."* A sticky task re-enqueues itself into its **thread-local**
   workqueue on `yield()`, and `trypoptask` drains that queue before it ever
   consults the multiqueue where `Threads.@spawn`ed tasks live. So a sticky
   task spinning on `yield()` stops its thread from picking up spawned work
   *ever again*, and once every default thread is spinning that way, nothing
   newly spawned can start at all — a permanent deadlock, not a slowdown.
   Spin briefly if you want a cheap hand-off, then `sleep`: only a real
   deschedule empties the thread's local queue. Two traps when testing this:
   `enq_work` places default threads at `threadpoolsize(:interactive)+1`
   onward, so pinning probes to tids `1:nthreads` leaves a default thread free
   and hides the bug completely; and the starved task's own `istaskstarted`
   flips the instant the spinners stop, so read it *before* releasing them.

13. **Keep type-stable and type-unstable paths at the right stability level.**
   If both kinds of path exist for an operation (e.g. typed kernel execution
   vs. dynamically-typed planning), consider whether they need to be
   *separate* paths: don't force the dynamic path to specialize per signature
   (compile-time explosion, tuple re-boxing), and don't erase types on the
   path where the compiler genuinely uses them (kernel invocation, argument
   moves). A function barrier at the boundary lets each side be what it is.

14. **`test/allocations.jl` measures scheduling overhead only while every
   task really is pinned.** Its bounds assume the pinned scope holds for the
   whole suite, but `allocate_array` tasks (the ones building each
   `let`-block DArray) take no `Chunk` inputs, so they carry *zero*
   data-transfer cost — nothing anchors them to a worker, and the scheduler
   spreads them the moment it can see per-processor load. The measured call,
   still pinned to worker 1, then pays to pull those chunks back, and a
   scheduler change shows up as a 5x allocation "regression" that is really
   cross-worker data movement. Build the fixtures inside the same
   `with_options(scope=...)` as the measurement. More generally: when this
   suite jumps on a scheduler change, first ask whether placement changed
   before hunting for a stray closure or box.

15. **A per-candidate term folded into a shared scalar is a no-op, and the
   tests won't tell you.** `estimate_task_costs!` compares candidate
   processors, so only terms that *differ per candidate* can change its
   decision. Accumulating one — e.g. summing every candidate's compute
   pressure and adding that total to `est_time_util` — leaves a constant
   offset that cancels out of both the comparison and the `sort!`, so the
   scheduler's ordering is bit-for-bit unchanged while the code reads as
   though the term is now considered. `test/scheduler.jl`'s cost assertions
   run against an *idle* scheduler where every pressure is zero, which makes
   summed and per-processor forms indistinguishable; a test that pins the
   term to one candidate and asserts the *other* one wins is what catches it.
   The behavioral check is cheaper still: 40 sleeping tasks over 4 workers
   land `[1 => 40]` when the term is dead and `10/10/10/10` when it is live.

16. **Only the acceleration x backend cross product exercises the data-movement
   paths, and one of the axes alone will pass while they are broken.** Two
   independent bugs in whole-object (`aliases_as_whole`) sparse tiles survived a
   green single-process-GPU suite *and* a green MPI-CPU suite, and both fell out
   the first time MPI x GPU ran. First: a host-to-device `move!` of a whole
   object only happens when tiles are *built* on the host and then placed on a
   device, which no single-axis configuration does. And such a hook must be
   defined at the `dep_mod` (5-argument) arity, because every GPU extension
   claims `move!(::TheirVRAMSpace, ::OtherSpace, ::AbstractArray{T,N},
   ::AbstractArray{T,N})` for the 4-argument form — a container method with
   unconstrained spaces is genuinely ambiguous with all of them (more specific
   in the value arguments, less specific in the space arguments), so it would
   need a tie-breaker per backend per space pair; nothing outside core defines
   the 5-argument form for arrays, and that is what every Datadeps copy path
   calls anyway. Second: `collect`'s gather tasks run wherever the caller's
   compute scope puts them, so under a GPU scope the `cat` tree runs *on the
   device* — and generic `cat` fills its output element by element, which is
   scalar indexing. Keep shared test bodies in a `test/array/*_defs.jl` file
   (as `stencil_defs.jl` and `sparse_defs.jl` do) and call them from all four
   entry points; `test/mpi/opencl.jl` makes the fourth cell cheap to run
   locally.

17. **Extensions of the same package must not reach into each other.** Load
   order between two extensions of one package is unspecified, so
   `Base.get_extension(Dagger, :MPIExt)` from another extension is a coin flip.
   When `AExt` and `A×BExt` both need to extend the same generic, declare that
   generic in core Dagger and let each extension add methods to it (see
   `inplace_mpi_parts` in `src/memory-spaces.jl`, the
   `mpi_device_direct`/`mpi_remap_space` hooks in `src/gpu.jl`, and the GPU
   processor types / `with_context` in `src/gpu.jl`). GPU×SparseArrays
   extensions import `CuArrayDeviceProc` (etc.) from Dagger and call
   `Dagger.with_context` — never `Base.get_extension(Dagger, :CUDAExt)`. The tempting
   shortcut — adding `B` to `AExt`'s trigger list — is worse than it looks: it
   makes `AExt` refuse to load at all until `B` is loaded, so MPI acceleration
   would have silently required SparseArrays.

18. **The scheduler needs DataStructures 0.19.** `Sch.jl` does
   `popfirst!(::PriorityQueue)`. That method exists only in DataStructures
   0.19; 0.18 resolves, loads, and then the scheduler throws on the first
   pop. Compat is `0.19` only — do not re-add `0.18` to satisfy a downstream
   pin. If a demo package pins 0.18, bump *that* package's compat (as the
   Jutul clone patch does), not Dagger's.

19. **Per-tile AMG can report `stats.solved` while `‖Ax−b‖` is O(1)–O(100).**
   `AMGPreconditioner` is block-diagonal: one V-cycle per diagonal tile,
   applied as Krylov `M` (`ldiv=false`). Left-preconditioned GMRES/BiCGStab
   then converge in the *preconditioned* residual. With many tiles that
   V-cycle is a weak additive-Schwarz operator, so Krylov stops while the
   true residual is huge (seen on Chan, VoronoiFVM penalty rows, and Jutul
   heat). BlockJacobi (exact tile LU) on the same layout is fine. Global AMG
   is `Blocks(n, n)` (one tile). Do not treat `stats.solved` as `Ax≈b` for
   block AMG; check the un-preconditioned residual. CG will also reject AMG
   as non-SPD — use GMRES.

21. **Reassigning a local to a value of a different type deoptimizes the
   *whole* body, not the assignment.** `arg = adopt_sparse_arg!(state, arg,
   deps)` in `_populate_one_arg!` looks like a cheap normalization, but
   inference must pick one type for `arg` over the entire method, so it
   widens to the join and every later use — `type_may_alias(typeof(arg))`,
   `supports_inplace_move`, the `ArgumentWrapper` construction,
   `get_or_make_arg_chunk!` — becomes a dynamic dispatch on a boxed value.
   The cost lands on the *common* path (the branch that never fires) and is
   invisible in a diff of the branch that does. Argument-processing code is
   per-argument per-task, so this is the worst possible place for it. Pass
   the new value forward into a function barrier instead of assigning it
   back; the callee then specializes per concrete type and the conditional
   is confined to choosing which call to make. `get_or_make_arg_chunk!`
   already exists for exactly this reason — extend the pattern rather than
   reintroducing the reassignment next to it.

22. **A re-tiling fallback needs the tile *backend*, and needs to outlive the
   call.** Making non-square-tiled operands work instead of erroring is two
   traps in one. First, allocate the destination through a tile-type-dispatched
   allocator (`allocate_tiled`): `DArray{T}(undef, part, dims)` gives dense
   tiles, so "repartition this sparse operator" silently becomes "densify this
   sparse operator" — an out-of-memory multiplier that a correctness test
   passes. A `DArray`'s type parameters do not record its tile type, so read it
   off a chunk (`chunktype(first(A.chunks))`). Second, `maybe_copy_buffered`
   frees its buffers when its body returns, which is only right when nothing
   escapes; a block preconditioner builds per-tile operators *from* the re-tiled
   tiles in tasks it never awaits, so it needs an ordinary array
   (`repartition`) whose lifetime is its own. And note that a *device* sparse
   tile supports neither end of a sub-range copy — reading it is scalar
   indexing, writing it would insert nonzeros into a CSC in place — so
   `copyto_view!` has to stage the whole thing on the host and re-upload
   through `move`, which is the one hook every GPU sparse extension already
   defines.

23. **A memory space must be keyed on the device, never on a handle that
   records current *ownership*.** `OpenCLExt.memory_space(::CLArray)` looked the
   buffer's `Managed.queue` up in Dagger's registered `QUEUES`. That field is
   OpenCL.jl's ownership tracking, not provenance: `convert(::CLPtr, ::Managed)`
   synchronizes and then re-stamps it with `cl.queue()` whenever the accessing
   task's queue differs, and `cl.queue()` is task-local *and lazily created*, so
   the first touch from a task that never ran `with_context!` re-stamps the
   buffer with a queue Dagger has never seen. The lookup then yields `nothing`
   and `CLMemorySpace(myid(), nothing)` does not even construct — a
   `MethodError` deep inside `aliasing`, arbitrarily far from the access that
   moved the ownership, on an array that was allocated perfectly correctly.
   Nothing about the memory changed; only a mutable bookkeeping field did. Key
   on `queue.device` (matched against `DEVICES`) instead. Suspect this shape
   whenever a space lookup fails for a value that a `Chunk` already carries a
   valid space for: the chunk recorded the space once, the value is being asked
   to re-derive it, and only the second one goes through the mutable field.

24. **`similar(::DArray)` must not fetch the source tiles.** Spawning
   `similar(chunk, T, sz)` per result tile looks like the way to preserve
   sparse/GPU backends, but it is a false data-dependency: `A * A` then moves
   every tile of `A` into the allocation tasks (and those tasks have no
   concrete `return_type`). At the small sizes the dense GEMM bench uses, that
   is a measured ~2×. Use `allocate_tiled` instead — dense stays
   `DArray(undef)` (GPU processors still override `AllocateUndef`), sparse
   stays sparse zeros. `similar(chunk)` is only right when you actually need
   the source value.

25. **Vendor GPU sparse tiles do not speak SparseArrays' names, and CSC×CSC is
   not a given.** `CuSparseMatrixCSC` / `ROCSparseMatrixCSC` store values in
   `nzVal` (and `colPtr`/`rowVal`); a kernel that reaches `.nzval` works on
   host CSC and `DeviceSparseMatrixCSC` and then dies on CUDA. Use `hasfield`
   or `nonzeros`, not a hardcoded field. Separately, `A * B` of two rocSPARSE
   CSC tiles falls into LinearAlgebra's generic matmul (scalar indexing) unless
   that AMDGPU version wraps CSR SpGEMM — tile `matmatmul!` must gather to
   host CSC, as `DeviceSparseMatrixCSC` already does, rather than hoping `*`
   stays on-device. oneAPI's `zeMemOpenIpcHandle` out-param is
   `Ptr{PtrOrZePtr{Cvoid}}`, not `Ptr{Ptr{Cvoid}}`. And once a GPU processor
   type (and its `show`) lives in core so combo extensions can dispatch on it,
   the backend extension must not redefine `show` — precompilation forbids the
   overwrite.

26. **A VRAM-stamped host `Array` is not a device buffer.** Collect's MPI cat
   tree densifies every tile to `Array` (device `cat` is scalar indexing),
   and those gather tasks run on the GPU compute scope. `execute!` used to
   label the result with the *processor's* space, so a host `Matrix` sat in
   a `ROCVRAM`/`CUDAVRAM` chunk. The next hop then took same-node device IPC
   and died in `ipc_export(::Matrix)`. Stamp the result from
   `value_memory_space`, and do not select IPC unless the chunktype is a
   GPU array (`ipc_type_eligible`). Space-only `ipc_eligible` is not enough.

27. **Log emitters in tests must be count-bounded, and chunk lists must be
   memory-bounded.** A `while !stop[]` logger on every default thread will
   starve the task that flips `stop` (lesson 12) and allocate slabs until
   the machine OOMs — measured at 250GB+ virtual across leftover
   `Pkg.test` children after only the parent shell was killed. Use a
   fixed `for i in 1:N` (N on the order of a few chunks), cap published
   slabs (`MAX_CHUNKS`), and when killing a hung Julia test kill the
   whole process group (`kill -- -$PGID`), not just the `julia -e`
   wrapper. `Pkg.test` spawns a child that keeps running if you only
   SIGTERM the wrapper.

28. **`nworkers() == 1` means "this process", not "there are workers".**
   Without `addprocs`, `workers() == [1]`. `remotecall_wait` /
   `remotecall_fetch` to `myid()` deadlocks — the Distributed waiter
   never runs on the calling task. Gate broadcasts with
   `length(procs()) > 1` (then `workers()` are remote only; Dagger does
   not import `nprocs`). `_map_workers` already calls `f()`
   locally when `p == myid()`; do not reintroduce a self-remotecall
   around `enable_logging!` / `get_logs!`.

29. **Do not mutate TimespanLogging globals at another package's toplevel.**
   `@logcategory` used to call `register_category!` while Dagger was
   precompiling. Those writes land in TimespanLogging's arrays in the
   *precompile process* and are discarded when TimespanLogging loads from
   its own image; Dagger's baked `const` IDs then disagree with a fresh
   runtime registry (or collide with MemPool/tests that register later).
   Category IDs are assigned lazily on first `category_id` use. Keep it
   that way — a `const ID = register_category!(...)` at Dagger toplevel
   is not safe.

30. **Benchmark fixtures must be fully awaited.** DArray construction is
   asynchronous: if setup launches `A` and `B` but waits only for `A`, the
   timed body inherits an arbitrary fraction of `B`'s allocation and scheduling
   work. That contamination is placement- and timing-dependent, so a scheduler
   improvement can look like a multi-fold regression. Wait every fixture that
   the measured operation consumes.

31. **SPMD benchmarks need a rank-uniform sample loop.** BenchmarkTools applies
   its `seconds` cutoff independently on each process; small timing differences
   can make one MPI rank stop while a peer enters another collective sample,
   deadlocking the next benchmark. Run one sample per rank at a time, use a
   collective maximum to make the stop decision, and use a cooperatively-waited
   `MPI.Ibarrier` after per-sample GC/teardown before timing the next sample (a
   blocking `MPI.Barrier` has the same progress-engine deadlock risk as lesson
   12). Otherwise a fast rank also charges its next sample for waiting on a
   peer's preceding GC, creating large but fake timing regressions in short
   collective operations.

32. **An empty benchmark suite is a benchmark failure, not a successful no-op.**
   AirspeedVelocity adds its own `time_to_load` result, so an orchestrator that
   swallows every leaf error still emits a plausible-looking green report with
   one row. Require a nonempty manifest, and abort an SPMD run on a leaf error:
   after one rank leaves a failed operation, attempting the next collective is
   unsafe. Also treat BenchmarkTools' generated `samplefunc` as a versioned
   internal API: 1.8 changed it from a two-argument function returning a tuple
   to a four-argument function writing measurements through a `Ref`.

33. **Synchronous and asynchronous submission need different batch sizes.** A
   large asynchronous batch amortizes channel and scheduler overhead while its
   submitter overlaps with continued planning. The same batch size on a
   one-thread `BatchedEnqueueQueue` withholds every task in a short region until
   planning has nearly or completely finished, serializing planning against
   execution. This is especially costly for iterative solvers, which execute
   many small datadeps regions. Tune the two queue types independently.

34. **A benchmark script shared across revisions must probe backend capability.**
   Airspeed runs today's script against old Dagger code, and a generic suite can
   also contain a leaf unsupported by the selected backend. MPI SVD, for
   example, uses a Distributed-only processor grid and divides by zero before
   sampling. Probe optional or newly backend-enabled operations while
   constructing the suite, so unsupported leaves are absent rather than
   aborting the comparison. When an external MPI worker does abort, persist
   each failing rank's `CapturedException` before `MPI.Abort`: the first failure
   may be off rank 0 and can terminate rank 0 before it writes anything, mpiexec
   output may disappear from the parent runner's log, and a generic "worker
   exited" message discards the only actionable diagnosis.

35. **A whole-region copy owner must represent the whole copy batch.** A
   remainder can be assembled by several copy tasks from disjoint source
   spaces. Registering each task as the owner of the whole destination makes
   the last registration replace the earlier ones; a whole-region consumer
   then waits for one piece and a late copy can overwrite its result. Making
   the last task depend on every earlier copy repairs that owner invariant,
   but couples disjoint copies unnecessarily and puts the last one on the
   critical path. Represent the logical write as a batch producer instead
   (lesson 38). Registering every exact destination span is correct too, but
   is a performance cliff for halo exchange: a 64-tile stencil produced
   hundreds of megabytes of interval-tree/overlap bookkeeping and regressed
   by 7–8x.

36. **Distributed benchmark kernels must exist on every worker before
   sampling.** Defining an `@stencil` wrapper only on the driver serializes its
   generated closure to whichever worker happens to receive a tile. Globals
   used by the closure (`Clamp`, `Reflect`, etc.) can then be missing or too new
   for that worker's world age, and even successful leaves randomly pay remote
   compilation during a timed sample. Import macros/globals in one
   `@everywhere` statement, then define the macro-using wrappers in a second
   `@everywhere` statement (the import must be evaluated before remote macro
   expansion). A one-block capability probe is not a distributed warmup.

37. **BenchmarkTools sees only the driver process's allocations.** In a
   Distributed benchmark, arbitrary tile placement makes the reported bytes
   include however many payload tiles happened to execute on the driver. A
   1024² `Float64` allocation consequently varied from 2–8 MiB with no global
   allocation change, and task overhead varied with the driver's share too.
   Give Distributed benchmark fixtures a deterministic balanced proc grid (the
   array and stencil suites use `assignment=:cyclicrow` when
   `length(procs()) > 1`) so both revisions measure the same local fraction of
   the workload. Do not force that assignment under MPI: named cyclic grids are
   built from Distributed processors, and an MPI rank's `procs()` is only `[1]`,
   so the grid is empty and allocation divides by zero. Retain `:arbitrary`
   there so the MPI-aware scheduler places tiles. The reported number is still
   process-local; deterministic placement only makes the comparison meaningful.

38. **A logical copy batch must keep every physical producer in every
   dependency view.** When a `MultiRemainderAliasing` restores one whole-region
   replica from disjoint pieces, launch each copy with its own source readers
   and syncdeps, but do not let each copy rewrite the destination's whole
   `ainfos_owner`, `arg_history`, or `arg_current`. After all copies are
   launched, record one logical writer whose producer is the complete task
   batch. Expand that producer in whole-object read/write dependencies,
   historical remainder dependencies, and free-buffer syncdeps; widening only
   the live owner silently leaves history or teardown waiting for one copy.
   Keep the singleton-copy path direct so ordinary per-argument moves do not
   allocate a batch vector.

39. **Gate diagnostic work at the call site, not only inside the logger.**
   `hier_log!` checks `HIER_TIMING[]`, but Julia evaluates its arguments before
   that check. Unconditional `time_ns()` calls around per-argument aliasing and
   slot creation therefore survived with timing disabled. Gate both the start
   timestamp and the finish/event argument evaluation; a disabled diagnostic
   should not pay for the measurement it discards.

40. **Reusable scratch macros must return their declared container types.**
   `@reusable_vector` and `@reusable_dict` expand through non-const global
   `TaskLocalValue`s. Without a type assertion, their expressions infer as
   `Any` even though the runtime containers are typed: iteration, element
   stores, and closures capturing them lose that type throughout the hot loop.
   The aliasing-result scratch introduced one boxed pair allocation per
   argument this way. Assert the container type inside the macros, before
   `empty!`, rather than relying on each caller to do so. Escape the supplied
   type expressions to resolve caller-defined types correctly, and check the
   return types with `@inferred` as well as the containers' runtime types.

41. **Remote aliasing batches must seed the driver's region memo.** Phase 1
   groups arguments by owner and computes aliasing remotely, but Distributed
   RPCs do not inherit the caller's `CHUNK_AINFO_MEMO` scoped value. Keeping
   those answers only in `arg_to_ainfo` makes Phase 4 ask the owners again.
   Seed the memo when merging each remote batch, keyed by the original local
   `Chunk`/`ChunkView` and dependency modifier, not the deserialized argument
   wrapper. Keep the local/MPI path unchanged: it already fills that memo.
   A typed merge barrier also prevents `remotecall_fetch`'s `Any` result from
   erasing the result container's type in the per-argument loop.

42. **Fix the SPD fixture's destination placement, not just its input's.**
   Giving `G` a cyclic assignment does not give `G * G'` that assignment:
   `similar(G)` allocates an arbitrarily placed result. A Cholesky benchmark
   must explicitly allocate its SPD destination with the same assignment and
   use `mul!` in untimed setup, or its input layout still varies between
   samples/revisions. Keep named grids Distributed-only and leave MPI on its
   native allocator. This controls the fixture, not the measured operation:
   out-of-place multiplication and Cholesky still allocate/place their own
   outputs normally, and neither scheduling heuristics nor thresholds change.

43. **Broadcast delivery must wake only its own consumers.** One shared
   condition for all `(root, tag)` slots makes a delivery wake every unrelated
   waiter: draining 1024 tags one at a time took 395 ms, versus 3 ms with
   per-slot conditions sharing the registry lock. Create conditions only when
   a consumer actually blocks. Keep a slot alive until all its consumers have
   left, including consumers already notified but still reacquiring the lock;
   an empty FIFO alone does not mean its condition can be replaced. Heartbeats
   and shutdown still notify every slot, and an empty consumer must also check
   `running`: a consumer already woken by a heartbeat is outside the shutdown
   notification queues and otherwise goes back to sleep after the relay stops.

44. **An uncontended MPI receive needs a lease, not an event.** Uniformity
   checks made the old receive guard create hundreds of thousands of
   `Base.Event`s per sparse product, although its `(comm, source, tag)` stream
   had only one receiver. Register `nothing` for an owned stream and create
   a one-shot event only when a competitor arrives; delete the stream and
   notify that event when the owner leaves. Never reset/recycle the event:
   notified competitors can still be inside `wait`. Use explicit locked
   blocks, since the old captured `our_event` lowered to a `Core.Box`, and
   release the lease in `finally` on both serialized and in-place paths so a
   failed receive cannot strand every subsequent receiver of that stream.

45. **Task-pool capacity is not a reason to create every slot upfront.**
   Completion and placement tasks can be short-lived, but each owns a
   task-local fire cache with capacity 32. Creating all 32 tasks, channels,
   and monitors on its first dispatch pays for 31 idle slots when that caller
   dispatches only once. Initialize slots on demand without changing capacity
   or the overflow policy; 100 single-use caches fell from 75,100 allocations
   / 4,004,800 bytes to 6,800 / 296,000. Clear the dynamic scope at actual slot
   creation, run its setup before scheduling, and register each dispatch
   before publishing the payload. Finalization must skip unassigned slots.

46. **Repeated calls do not necessarily warm the option-default cache.**
   `BasicLFUCache` can immediately evict a newly inserted frequency-1 key
   when every resident key is more frequent, so a new signature keeps taking
   the default fallback after arbitrarily many warmups. Test that path with
   a full cache in a fresh task, not by changing capacity or eviction policy.
   Splat `Signature.sig` directly for non-keyword calls: splatting its
   identical `sig_nokw` view boxes the view and an iteration pair per type.
   The generic fallback also needs no specialization of its unused arguments;
   leave type-dependent user overrides untouched. A four-type cold default
   population fell from 217 allocations / 9,088 bytes to 49 / 1,792, while
   cache hits stayed at 1 / 256. Keep keyword views instead of adding a copy
   to every signature, especially signatures whose defaults are already cached.

47. **Even a typed `findmin(::Dict)` can box its result in an eviction loop.**
   The defaults LFU allocated one `(frequency, key)` tuple per eviction
   despite its concrete key/frequency types. A direct minimum-frequency scan
   removes that allocation without changing capacity, admission or eviction
   decisions. Initialize from the first dictionary entry and update only on
   strict `<` so equal-frequency ties preserve the original iteration order.
   Compare exact cache contents and frequencies against the old algorithm
   after mixed hits/misses, including zero capacity. Ten thousand repeated
   evictions fell from 10,000 allocations / 320,000 bytes to zero; cache-hit
   behavior is unchanged. Together with lesson 46, cold default population
   is 25 allocations / 1,024 bytes rather than 217 / 9,088.

48. **A BLAS method accepting GPU arrays does not make it GPU BLAS.**
   `LinearAlgebra.BLAS.gemv!` accepts `AbstractArray` inputs on Julia 1.12,
   so a direct call with `ROCArray` tiles reaches host BLAS with device
   pointers and raises `ReadOnlyMemoryError`. Restrict direct host BLAS
   calls to CPU `Array` types, as `matmatmul!` does, and dispatch other
   tile types through the backend's `generic_matvecmul!`. Test with tiles
   allocated inside the GPU scope, including transpose, complex adjoint,
   and nonzero beta across several tiles; CPU-only multiplication tests
   cannot catch this dispatch error.

49. **A distributed view needs both broadcast style and bulk staging.**
   `SubArray` does not inherit `DArray`'s broadcast style. Adding that style
   alone still leaves `Distribute` slicing the wrapper through scalar
   parent indexing. Materialize the selected region with bulk DArray
   indexing before broadcast partitioning. Keep integer-indexed dimensions
   during that copy and drop them on individual tiles: redistributing a
   generic `ReshapedArray` of a DArray also falls back to scalar indexing.
   Use nonuniform values and offset views crossing tile boundaries, with
   scalar indexing disabled, to check placement as well as dispatch.

50. **Uniform scheduling must preserve Datadeps' chosen thread.** The
   planner already chooses a rank-uniform processor, but widening its scope
   to every processor in that memory space discards the thread choice. MPI
   ignores rank-local capacity and uses a deterministic processor order, so
   every task then lands on the first thread of its rank. MPI also disables
   work stealing; it cannot repair that placement later. Retain an exact
   processor scope under uniform execution, while leaving Distributed's
   memory-space scope available for load balancing. Test both flat and
   hierarchical planners with several rounds of tasks, checking actual thread
   IDs as well as uniform processor metadata and restricted user scopes.

51. **Asynchronous benchmark workers need explicit output streams.** Julia's
   `run(cmd; wait=false)` redirects stdout and stderr to `devnull`; it does
   not inherit them as synchronous `run` does. The benchmark orchestrator
   consequently hid all worker progress and errors while CI appeared stuck
   at "Loading benchmark script" for 85 minutes. Launch both plain and MPI
   workers through `pipeline(...; stdout=stdout, stderr=stderr)` so the last
   active benchmark and any failure are visible in the job log.

52. **A missing benchmark spread is not evidence of a repeatable change.**
   AirspeedVelocity omits quartiles when a trial has one timed sample. The
   three-second CI budget included setup, teardown and GC, so even sub-ms
   kernels often produced only one sample. Treating either missing spread as
   automatically significant bypassed the other revision's measured noise:
   a broadcast trial with 0.25 ms and 2.91 ms samples became a +327% regression
   against a single 0.37 ms sample. Report such timing changes as inconclusive
   in both directions, preserve allocation gates, and budget enough wall time
   for repeated samples. Keep the classification in one place and export its
   structured results for aggregation; do not infer significance by parsing
   rendered Markdown.

53. **Performance workers must not silently enable correctness diagnostics.**
    The MPI benchmark worker copied `check_uniformity!(true)` from the test
    bootstrap. That hashes and compares planning values across ranks for every
    task and argument, so short kernels mostly measured diagnostic traffic.
    A warmed 2-rank × 2-thread matrix-vector benchmark on the same baseline
    took a median 249 ms with checks and 13 ms without; minimum allocations
    fell from 166,088 to 78,470. Use the documented production default for
    benchmarks, log the selected mode, and make checks an explicit diagnostic
    option. Keep uniformity checking enabled in functional MPI tests. Repeat
    comparisons before attributing a few noisy samples to a scheduling change.

54. **Timing verdicts need actual sample counts and independent confirmation.**
    AirspeedVelocity's flattened statistics omit the sample count, and
    BenchmarkTools' `params.samples` is a limit rather than a measurement.
    Recover counts from raw `times` arrays. A tight spread within one process
    does not measure variation between revisions' runs: require five samples
    per revision and confirm timing regressions in fresh processes with the
    revision order reversed before counting them.

55. **CPU-only scope checks must account for MPI processor wrappers.**
    `compatible_processors` under MPI includes `MPIProcessor{ThreadProc}`, so
    `all(proc isa ThreadProc for proc in ...)` rejects an all-CPU scope.
    LU's former `use_lapack_panel` check selected the per-row task
    fallback under MPI. Test backend capability through a wrapper-aware hook
    or forwarding predicate rather than the outer processor type. Increasing
    matrix size also increases that fallback's scheduling work.

56. **MPI callers must read mutable scalar status through the dependency graph.**
    A plain `Ref` used as a Datadeps argument is updated on its owning rank;
    reading the caller's original `Ref` on other ranks can return stale status.
    Submit a final scalar read with `In(ref)` inside the region, then fetch its
    uniform task result for error checks and returned factorization status.

57. **GPU benchmark scope must cover setup, probes, and the timed run.**
    Loading a GPU extension does not enable its processors by default. Apply
    the explicit device scope around both suite construction and BenchmarkTools
    execution, so fixtures and capability probes use the same backend as the
    measured operation. CPU named cyclic grids do not preserve GPU placement;
    leave GPU fixtures on the scope-aware allocator. Use Float32 for portable
    GPU coverage (Metal and some OpenCL devices lack Float64). Require one
    backend per worker: a mixed CPU/GPU specification would otherwise measure
    GPU fixtures under the CPU method label.

58. **A reused benchmark workdir must discard old response IDs.** The plain
    worker protocol starts request IDs at one for each orchestration run.
    Reusing `BENCHMARK_WORKDIR` without removing old `response_*.json` and
    `result_*.json` lets the orchestrator accept a previous run's response
    immediately, including an error that kills a healthy new worker. Clear
    protocol outputs when starting a worker; already-loaded trials remain in
    the orchestrator's result dictionary.

59. **Buildkite dependencies do not require an actual passing test.** A test
    skipped by its condition satisfies `depends_on`, so benchmark jobs that
    require successful tests need both matching skip conditions and an outcome
    check (`buildkite-agent step get outcome`). Keep GPU+MPI dependent on the
    ordinary GPU test as well as the GPU+MPI test. Aggregate reports with one
    writer, retain only sections for the same PR head, and preserve GPU sections
    when the CPU reporter finishes later. Trigger aggregate snapshots after
    each vendor completes, with a shared writer concurrency group: an offline
    vendor queue must not withhold another vendor's finished results.

60. **Validate Buildkite matrix expansion, not just YAML syntax.** Buildkite
    replaces `{{matrix}}` once. A command containing `{{{{matrix}}}}` is valid
    YAML but passes `{{array}}` to the benchmark worker, which rejects the suite
    before sampling. Check the expanded command arguments and artifact paths
    for every pipeline; parser and dependency checks do not catch this error.

61. **Resolve optional test packages before loading dependencies or starting
    workers, and share the active test project.** Installing a GPU backend can
    downgrade LLVM and Atomix after Dagger/KernelAbstractions already loaded
    them; GPUCompiler then sees the old module with the new package graph and
    fails before any GPU test runs. `Pkg.test` uses a temporary project, so
    workers must inherit `Base.active_project()`, not the repository project.
    Install optional packages in one batch before imports and worker startup.

62. **Optional language runtimes must not initialize in unrelated test jobs.**
    Listing PythonCall/CondaPkg as unconditional test dependencies initializes
    Python during `Pkg.test` precompilation, even in GPU-only jobs. A Python
    library discovery failure then aborts Metal tests before the runner starts.
    Install those packages before imports only when the Python suite is selected;
    default CPU runs still select that suite, and GPU runs do not.

63. **Python setup failures can come from two independent upstream regressions.**
    CondaPkg scans every environment in `LOAD_PATH`, including optional global
    and stdlib environments whose `Project.toml` may not exist. CondaPkg 0.2.37
    tries to open those files and fails before Python initialization (fixed by
    JuliaPy/CondaPkg.jl#235). Once resolution succeeds, PythonCall 0.9.36 can
    still fail library discovery because conda-forge's Python 3.15 separates
    `libpython` from the interpreter (fixed by JuliaPy/PythonCall.jl#839).
    Compare unrelated PR logs and check upstream releases before changing the
    test bootstrap: installing Python only for its suite does not cause either
    failure. Both fixes together pass Dagger's Python suite on Julia 1.11.

64. **Uniform eager placement must vary by task, not by ready-arrival order.**
    MPI ignores rank-local load and capacity, so sorting every task's candidates
    identically sends all plain `@spawn` work to rank 0's first thread. Rotate
    the canonical candidates using the already-uniform thunk ID, rather than
    a scheduler cursor: ready tasks can arrive in different orders across ranks.
    With input-owner affinity, rotate only that rank's candidate prefix so
    locality survives; if a scope excludes the owner, rotate the permitted
    candidates instead. Calls without a task must retain canonical ordering
    for Datadeps and processor grids. The issue #760 reproduction changed from
    16 serial tasks on one processor to four tasks on each of four processors.

65. **A dynamic candidate scan can box even an unused backend argument.**
    Calling `processor_owner_rank(accel, proc)` from a `Vector{Processor}`
    loop boxed the isbits `MPIAcceleration` once per candidate: four candidates
    added four allocations / 64 bytes per eager placement. An out-of-line scan
    with `@nospecialize(accel)` preserves runtime hook dispatch while passing
    the already-boxed backend through the loop. Only that scan boundary now
    boxes (one allocation / 16 bytes), and the no-affinity path stays unchanged.
    This is a pass-through planning argument; do not erase types needed by
    execution or arithmetic to achieve the same effect.

66. **An MPI test timeout can leave ranks in separate process groups.**
    Timing out `test/mpi/run.jl` killed its launcher but left both Julia ranks
    running; MPICH had placed each rank in its own process group. After a
    timeout, inspect rank PIDs/PGIDs and stop those groups too. Killing only
    the wrapper's group is not sufficient for this launcher.

67. **A Distributed GPU transfer must preserve the source's ownership.**
   Sending a raw device array in a slot-creation RPC deserializes it on the
   receiver while `from_proc` still names the sender. The subsequent GPU
   `move` then activates a remote worker's context locally and asserts; some
   backends cannot serialize the array in the first place. Send an owner-side
   `Chunk` and use the backend's Chunk transport instead. The same rule
   applies when fetching a remote source for in-place copies. Pin regression
   inputs and consumers to different workers selecting device 1: a GPU scope
   alone does not guarantee that a transfer actually happens.

68. **CUDA IPC must export staging memory and return an owned copy.**
   CUDA's pooled allocations cannot be exported with `cuIpcGetMemHandle`, and
   returning an imported mapping makes the receiver alias the sender's data
   and depend on its lifetime. Use the shared `ipc_export`/`ipc_materialize`
   hooks, keep the staging token on the sender until the receiver finishes,
   and release it in `finally`. Use `CUDADRV` for closing handles too: CUDA 6
   moved the driver API into CUDACore. Test repeated transfers in both
   directions and mutate the destination to verify source independence.

69. **Two GPUs of one process are not one device, and may not even be
   peers.** Several paths assumed "same worker" meant "one kernel may touch
   both buffers": the same-worker device remainder copy, `collect`'s
   in-process `cat`, and CUDA `pointer()` (which takes ownership for the
   *active* device and throws without P2P). `CUDA.pin` is also only deduped
   per context while host registration is process-wide. Key direct device
   paths on equal memory spaces, not equal workers, read addresses without
   ownership side effects, and test with tiles on two devices of one
   process: the one-GPU-per-worker suites cannot see any of this.

70. **Reading a GPU task's result needs Dagger's stream, not the caller's.**
   GPU `execute!` returns without synchronizing: the kernels are still queued
   on Dagger's per-device stream. A plain `fetch` + `Array(x)` from the
   caller copies on the caller's own task-local stream, and is correct only
   if the array library synchronizes the previous owner when another stream
   touches a buffer. CUDA, AMDGPU and OpenCL do; oneAPI does not, and it also
   copies on the *calling task's* device. A green CUDA run therefore does not
   prove the wait exists. Before reading a tile on the host, call
   `gpu_synchronize(chunk.processor)` and copy under `with_context`, on the
   chunk's owner (as `collect` does via `_collect_host_tile`). Note that
   IntelExt's hooks sync only the calling task's stream, and each oneAPI
   `execute!` runs on a fresh task, so oneAPI's own hook is not sufficient yet.

71. **Never unregister host memory from a finalizer that can block.**
   `CUDA.pin` unregisters in a GC finalizer that takes a `ReentrantLock`
   (as does `AMDGPU.Mem.unpin`, which ROCExt's finalizer used to call). When
   that lock is contended the finalizer throws "task switch not allowed from
   inside gc finalizer", the unregistration is lost, and GC frees the memory
   anyway: the range stays registered with the driver, and a later
   allocation at that address fails to register or is DMA'd through stale
   pages. Multi-GPU GEMM segfaulted a few iterations in, with a backtrace
   that blew the stack while printing (`jl_static_show` recursion) and showed
   nothing of the cause; the "error in running finalizer" lines before it
   were the only clue. (On ROCm, 2-GPU Datadeps tests segfaulted the same
   way, inside `hipMemcpyWithStream`.) CUDAExt's and ROCExt's `pin_buffer!`
   now register buffers themselves, and their finalizers only `trylock`,
   re-arming themselves on contention (which keeps the buffer alive until a
   later GC can unregister it). Anything that must run before
   memory is freed has to be finalizer-safe: no blocking locks, no yields.

72. **Two GPUs without peer access: never let CUDA.jl stage the copy.**
   Without P2P, `copyto!` between devices goes through a fresh *pageable*
   `Vector` with a synchronous DtoH: 1.4 GB/s on PCIe-attached L40S, where
   `cuMemcpyPeerAsync` (which works without peer access; the driver pipelines
   it through its own pinned buffers) reached 22 GB/s. A 4-GPU GEMM spent
   27 s per call in copies. Check `can_access_peer` before assuming a copy
   path is cheap, and measure the copy primitive in isolation first -- it
   took one 30-line script to find a 15x. (HIP does not stage this way:
   between two RX 6900 XTs without peer access, AMDGPU's `copyto!` and
   `hipMemcpyPeerAsync` both reached ~5.5 GB/s, vs 1.4 GB/s through a
   pageable host buffer, and `copyto!` blocks until the copy is done.)

73. **One stream per device serializes every transfer with that device's
   compute.** Copies enqueued on the device stream wait behind its kernels,
   and an event recorded on the *source's* stream waits for kernels that
   merely read the data being sent (a GEMM's panel owner is busy reading the
   panel it is asked to send). Peer copies now run on a per-device copy stream
   ordered only against the allocations they touch (`BUFFER_EVENTS`, fed by
   Datadeps' `writes` option), and are host-synced so every existing
   synchronization point stays valid; 4-GPU GEMM went from ~3.9 s to ~3.1 s.
   (ROCExt does the same with HIP events and `hipMemcpyPeerAsync`.)
   Bookkeeping keyed by address outlives the allocation: a peer copy's
   "already complete, skip the next write" flag, left on short-lived packing
   buffers, swallowed the first write to the next allocation at that address,
   and a later copy out of it read stale data. Name the task the skip is for
   (`filled_by`), or keep entries to plain events.
   Two traps found on the way: Datadeps spawns copy tasks `meta`, so
   `execute!` sees `Chunk`s, not arrays (unwrap before classifying, or every
   copy looks like an opaque argument and falls back to whole-stream waits);
   and `unsafe_free!` tasks hand you a freed array (check `data.freed`).

74. **Level Zero IPC handles cannot be serialized between processes.** Intel's
   opaque `ze_ipc_mem_handle_t` includes a process-local file descriptor.
   MPI's serialized metadata copies the descriptor number, not its underlying
   resource, so `zeMemOpenIpcHandle` fails with `ZE_RESULT_ERROR_INVALID_ARGUMENT`
   on the receiving rank. A usable import needs descriptor passing, such as
   Unix-domain `SCM_RIGHTS`, before reconstructing the handle. Keep Intel MPI
   on host-staged transport until that exists; a same-node/device check or
   `DAGGER_IPC=1` cannot supply the missing descriptor. See
   https://github.com/intel/compute-runtime/issues/448.

75. **ROCm Distributed tests need a spare Julia thread for host calls.**
   AMDGPU's persistent host-call tasks poll HSA signals with
   `Libc.systemsleep` and `yield`, which can monopolize a sticky task's only
   thread (lesson 12). With one default thread the GPU suite stopped making
   progress in `hostcall_host_wait`, without an exception; with two threads
   the host-call task remained alive while the other thread continued into
   the stencil suite. Give both the driver and Distributed workers two
   default threads in ROCm CI. Increasing only the driver's thread count
   leaves remote workers exposed to the same starvation.

76. **GPU event tracking must check a chunk's acceleration ownership.**
   `root_worker_id(::MPIRef)` and MPI processors intentionally return `myid()`
   on every rank; Distributed's worker-id test therefore calls every MPI chunk
   local. A GPU copy task executes on both endpoint ranks, and recording its
   arguments then tries to unwrap the remote endpoint's placeholder and throws
   "MPIRef is not owned by this rank". Check the handle with
   `is_local(current_acceleration(), chunk.handle)` before inspecting storage.
   The existing cross-rank GPU dataflow tests exercise both endpoints; a
   Distributed-only suite cannot catch this mistake.

77. **Reduce apparent GPU ordering failures below Julia before adding fences.**
   This host's Ryzen 9600X `gfx1036` GPU with HIP 7.2 returns zeros after an
   asynchronous kernel or device copy followed by a short pause, even when
   the same stream is synchronized before reading. A native HIP C++ program
   reproduces it without Julia, KernelAbstractions, or Dagger; synchronous
   allocations and disabling SDMA do not fix it. Setting
   `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` makes that reproducer pass.
   Per-halo synchronization happened to hide the failure, but would impose
   a production cost for a host runtime problem. Use the serialization flags
   only for local correctness validation and report them with the results;
   they are not a valid configuration for performance measurements.


78. **oneMKL queue interop can fail below Dagger on Intel Xe2 GPUs.**
   With oneAPI.jl 2.9.2 and its 2025.3 support libraries, this Arc Pro B60's
   default Level Zero v2 adapter aborts with
   `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` when SYCL wraps a native Level Zero
   queue for GEMM. A standalone `oneArray` `mul!` reproduces the abort; worker
   EOF errors are its aftermath. `SYCL_UR_USE_LEVEL_ZERO_V2=0` selects the
   legacy adapter and passes that reproducer. Diagnose native BLAS failures
   separately from Dagger scheduling and record the adapter used in validation.


79. **A passing GPU suite does not certify dependency finalizers.**
   Julia logs finalizer exceptions without failing the suite or its exit code.
   AMDGPU 2.8.0's `src/cache.jl` `library_state` accepts a `destroy_handle`
   callback but its task finalizer calls undefined `destroy_handle!`, producing
   an `UndefVarError` while the two-rank numerical suite passes all assertions.
   Check stderr as well as test counts, and attribute the callback to its
   owning package before changing Dagger's task or GPU resource lifecycle.


80. **Installing an external GPU compiler is not the same as selecting it.**
   GPUCompiler 1.23 defaults `GCNCompilerTarget.backend` to `:external` only
   when `AMDGPU_LLVM_Backend_jll` is already loaded, not merely installed.
   AMDGPU 2.1 on Julia 1.11 otherwise uses the embedded LLVM backend, which
   computes wrong mixed-boundary indices on this `gfx1036` host despite
   correct scalar arguments and valid-looking LLVM IR. Loading the external
   LLVM 22 backend makes the unmodified kernel compute the correct indices.
   Load it on every ROCm test/benchmark worker before kernel compilation;
   adding it only to the driver's environment leaves remote codegen unchanged.
   Keep both comparison revisions on the same backend.

81. **Small Intel MPI smokes do not validate the benchmark workload.** On this
   Arc Pro B60 with oneAPI 2.9.2, two-rank functional tests pass all 171/174
   assertions with host staging and the legacy Level Zero adapter, and the
   MPI array benchmark passes all 28 leaves at sizes 256/1024. The linalg
   worker nevertheless segfaults on `A*A` at size 1024 with 128-element tiles,
   inside SYCL's kernel-info lookup called by oneMKL GEMM. That stack identifies
   the failing native call, not its root cause. Keep Intel MPI functional tests
   enabled separately from the benchmark jobs; a green small matmul test is
   insufficient evidence to enable the larger tiled benchmark matrix.

82. **`BasicLFUCache.get!` keeps a stale value for a present key.** A cache
   whose hit condition is stricter than key equality (`compat_entry` also
   requires `entry.procs == procs`) must *replace* the entry on a miss, not
   `get!` it: otherwise the stale entry stays forever and every later lookup
   misses and recomputes. Measured: a rank-local benchmark that registered
   extra processors after the world benchmarks had warmed the cache made
   unscoped tasks 5x slower with 10x the allocations, but only in a suite
   that registers *after* warming; a standalone script that registers first
   never shows it. When an integration branch is slower than its parts, first
   check that the baseline contains the same commits (the "regression" of
   pinned tasks turned out to be a missing earlier fix, 05e04cb), then run
   the real harness with one benchmark removed at a time.


83. **`task_tid_for_processor` decides whether `execute!(::ThreadProc)` is
   inline or a hand-over.** The scheduler starts a task pinned to its
   processor's thread only for processor types that define
   `task_tid_for_processor`; every other type starts migratable. A wrapper
   processor (`MPIProcessor`) that forwards to an inner `ThreadProc` without
   defining it makes `execute!(::ThreadProc)` miss its inline fast path and
   create, schedule and wait for a second pinned task on *every* call:
   160-190 us per task here, 7 us once pinned (a dependent chain went from
   ~410 to ~270 us per link). Also: a latency benchmark that pins its chain to
   thread 1 shares that thread with the driver planning the links and can hide
   exactly this kind of gain; pin to another thread. Sampling profilers are
   too coarse for 100-us latencies and their signals can trip the hang detector
   of long cross-rank chains; phase timers around the suspect call localize
   it in one run.

84. **Pooled task slots keep the thread decision they were created with.**
   `@reusable_tasks` pins a slot (or makes it migratable) once, when it is
   created, and reuses it for later tasks. Where the thread decision can differ
   per task (MPI pins only tasks that asked for a thread while free threads are
   on), pinned and migratable tasks need separate caches; otherwise a task that
   reuses a slot some earlier pinned task created stays bound to that thread
   (seen only as "the two tasks the rotation put on the busy processor wait",
   with seven threads free). It shows up only when a pinned warm-up precedes the
   free tasks, so test the mixed sequence, not each kind alone.

85. **A latency test must assert the thing it measures, and probes must be
   type-stable.** `fetch` returns after the result has been *delivered* to the
   other ranks, which with every thread but one busy takes ~0.45 s longer than
   running the task (the broadcast relay shares that thread): a test of "tasks
   start on a free thread" that asserted the `fetch` time failed 6 of 8 runs on
   every build, including ones without the change under suspicion. Stamp the
   time inside the task. And in a script, a non-constant global passed as a task
   argument has an unknown type, which sends every task through the slow
   protocol that broadcasts the result type: spawn in a function. A non-yielding
   loop never reaches a GC safepoint, so a collection triggered elsewhere waits
   for it. Several "causes" were ruled out one at a time before the real ones
   showed up; per-task start and fetch timestamps found them faster than any
   profile.

86. **`DTask.owner` is not a place marker; use `DTask.placed` to say where a result lives.**
   `owner >= 0` also switches `fetch` onto the failure-only path of elided tasks, which is wrong
   for tasks of a Datadeps region. Under MPI the tasks that make the blocks of a `DArray` are
   such tasks (tagged, not elidable, no owner), so every task reading a block was planned in
   full by every rank (181 us against 28 us for a `per_rank` chunk). The block grid is the same
   on every rank, so `emit_chunk_tasks!` now records each block's rank (`placed`, through
   `note_chunk_placement!`) and the elision logic reads that. When probing elision, take the
   owner of a block from the real chunk (`fetch(t; raw=true).handle.rank`), not from a guessed
   round robin: the default grid fills processors rank by rank, so on 4 ranks x 8 threads all
   four blocks of a small array sit on rank 0 and the probe measured nothing. And a chunk's
   scope only matters through `constrain(task scope, chunk scope)`: check that chain, do not
   treat any scope restriction as a reason to plan in full.

87. **A check added to the per-task path must not scale with the size of a rank-level scope.**
   `Dagger.scope(mpi_rank=r)` is a `UnionScope` with one `ExactScope` per thread of the rank, so
   `constrain`, `objectid`, `===` and `==` on it cost time proportional to the thread count. A
   `constrain(compute_scope, result_scope)` that I added to the elision check for every task made
   a foreign rank's planning cost of a rank-scoped task 69-76 us instead of 38-43 us at 16
   threads (123-130 us at 32) against ~20 us for an `ExactScope`; the result scope is always
   `nothing` there, so the call was pure cost. Do such work only where it can change the answer
   (here: only for chunk arguments that carry a restriction), and after touching elision compare
   the foreign-rank cost of an `ExactScope` task with that of a rank-scoped one (probe
   `LLM_info/bench/scope_cost.jl`, 16 and 32 threads per rank). Also: building a rank scope costs
   ~0.35 ms at 32 threads; build it once.

88. **Type the return of anything that reads from a `Dict{..,Any}` cache, and
   never extrapolate laptop timings to the cluster.** `mpi_scope_processor`
   returned an entry of a `Dict{UInt,Tuple{Any,Any}}`, so it inferred as `Any`
   and every use in `elide_task!` (field reads, `memory_space`,
   `tochunk(...; type=T)`) was a dynamic dispatch, once per planned task;
   asserting `::Union{Nothing,MPIProcessor}` on the return sites took 13-14 %
   off the per-task cost (19.3 -> 16.75 us on Fulda). The same loop costs 1.4 us
   on an M-series laptop: on the Opteron nodes *every* primitive (atomic add,
   lock, `Comm_rank`) is 7-20x slower, so the shape of a profile transfers but
   absolute numbers do not. Measure per-task costs on the cluster, with GC time
   reported separately (here it was 0 ms, so it was not the explanation). A
   scoped value is a walk of the scope chain: read it once.


89. **A collective under Dagger must be non-blocking and waited for
   cooperatively, and which MPI is installed changes the answer.** A blocking
   `MPI.Allgather`/`Barrier` parks the OS thread inside MPI (no relay runs on
   it, no GC safepoint is reached), which closes a cross-rank cycle with the
   broadcast relay. MPI.jl 0.20 wraps the blocking `Allgatherv` only, but its
   low-level `MPI.API.MPI_Iallgather`/`MPI_Iallgatherv` exist: issue the
   request and wait with `__wait_for_request` (`MPI.Test` plus `yield`).
   Measured, the same exchange is 2.6x *faster* than hand-written
   point-to-point rounds on Open MPI 5.0.10 (4 ranks: 41 vs 105 us) and 2.4x
   *slower* on MPICH 5.0.2 (216 vs 91 us); on Fulda (Open MPI) the native one
   wins from 16 KB per rank. So keep both paths testable (a `Ref` switch, an
   environment override) and run the tests on both. To get Open MPI locally,
   copy `test/mpi/env` to a scratch directory, make its `[sources]` and
   manifest paths absolute, `Pkg.add("MPIPreferences")` (the preference is only
   seen for a direct dependency) and call
   `MPIPreferences.use_system_binary(library_names=[...], mpiexec=...)`. A
   per-contribution size cannot choose the algorithm: every rank must decide
   alike.

90. **An idle poller is not free, and a longer sleep can make it worse.** A
   rank in G groups ran G relay tasks that wake on their own; each idle one
   blocks an OS thread in `Libc.systemsleep` per cycle, so with 4 threads a
   world hop went 50 -> 91 / 255 / 897 us at 8 / 32 / 128 live groups. A longer
   blocking sleep made it *worse* (a thread is held hostage for the whole
   sleep), a cooperative `sleep` helped only partly, and serving every group
   from the world's relay task was slower still, because one `MPI_Improbe` per
   group per sweep costs more than the sleepers it replaces (8 groups: 259 us).
   What works is fewer live groups: `free_group!` restores the 49 us. Measure
   with `LLM_info/bench/many_groups.jl` before redesigning a poller.

91. **`@reusable` bound variables infer as `Any`; price a scaling suspect with
   phase timers and A/B in one allocation.** Lesson 40's fix covered
   `@reusable_vector`/`@reusable_dict` but not `@reusable` itself: its
   task-local cache global is non-const, so the variable it binds (and
   everything computed from it in the block) was `Any` -- `length(P)` of a
   `@reusable ... P begin` block returned `Any`. `estimate_task_costs!`'s
   `randperm!`/`sort!` of the candidates therefore ran with dynamic dispatch,
   and a loop I first wrote inside that block measured 27 % *slower* than the
   baseline in all six rounds (there is no closure and no `Core.Box` here: the
   macro expands to a plain try/finally). The macro now asserts the cache type;
   check new scratch code with `@inferred`.
   The scaling problem itself: unscoped single-process spawns grow from 108 to
   248 us/task between 8 and 192 threads. Timers around the phases of
   `schedule_one!` put it in `estimate_task_costs!` (35 us at 8-16 threads,
   70-155 us at 64; pinned tasks skip it and stay at ~5 us): per-candidate
   pressure lookups (6 -> 25 us), cost fill (8 -> 15 us), and the candidate
   shuffle+sort (12 -> 24-58 us). Two separate Slurm jobs of the same revision
   differ by 30-40 % (11.3 vs 14.7 ms for the same fan-out), so an A/B needs
   both revisions alternating in one allocation (`LLM_info/bench/ab_fan.sbatch`),
   several rounds. Sampling profiles of the 200-us spawn path collect only ~100
   snapshots (lesson 83).

92. **A short timing window counts a rare pause far too heavily; compare steps,
   not the mean.** Otus, 8M bodies, 192 threads, 3 timed steps: the step times
   were 0.187, 0.189 and 0.766 s, because one garbage-collection pause (0.456 s)
   fell into the third step. The mean (0.381 s) looked like a scaling problem
   and a sum-of-phases gap of 0.17 s per step ("time in no phase") hid it;
   with the collection moved out of the window (`--gc_interval=64`) the steps
   are 0.19 s and the phases add up. Report the per-step list, read the GC
   counters (`gc.pauses`, `gc.total_s`) before hunting a missing phase, and
   do not attribute a speed difference between revisions before the same
   window has been checked for such a pause on both sides.

93. **Never read a pooled object after handing it over, and prove a race fix with a
   widened window.** `do_tasks` pushed each `TaskSpec` into a processor's queue
   and then read `scope`, `options` and `est_occupancy` again for its wake-up
   decision. From the push on another runner may take the spec, run it and
   recycle it (`reset_task_spec!` sets those fields to `nothing`/0), so on few
   threads the typeassert threw, and `FireTaskSpec` reported the failure under
   `first_task.thunk_id`, read after the recycle: id 0, or the id of whatever
   task had reused the spec, which failed an innocent task. A failure delivered
   to a recycled thunk then threw again in `task_failed!` (`options === nothing`)
   inside a pooled task, which is a silent hang (lesson 9). Read everything the
   rest of the function needs while you still own the object; read the id
   *before* the call that hands it over. The race hit 1 in 20 fresh 2-thread
   processes on the Fulda nodes and never on a fast laptop; a stress test that
   merely succeeds proves nothing there. Put a hook right after the hand-over
   (`Sch.AFTER_ENQUEUE_HOOK`) and sleep in it: the bug then shows in every run,
   and the test must fail on the unfixed code before it counts. Note that the
   exception was often invisible to the test's own `fetch` (reported under a
   stale id), so count the failures (`Sch.FIRE_FAILURES`) instead of waiting
   for a failed task.

94. **A cache keyed by `objectid` of a scope, and a generic `constrain`, both cost
   O(threads) on the per-task path.** `Dagger.scope(mpi_rank=r)` is a `UnionScope`
   holding a tuple of one `ExactScope` per thread. `objectid` of it hashes the
   whole tuple, so the lock-protected union-scope cache of `elide_task!` cost
   ~35 us per lookup and was probed three times per task; and
   `constrain(::UnionScope, y)` is generic (it wraps `y` and rebuilds the whole
   union), so `constrain(rank_scope, AnyScope())`, which `schedule_one!` does for
   every task, cost ~90 us and 11 KB of garbage to return its argument. Both
   showed up only as "a rank scope is 3-4x an `ExactScope`" until a profile of
   a rank that only plans (and one of the owner) named the frames. Key such caches
   by the object's address (`jl_value_ptr`; a union scope is not a bits type, the
   entry keeps it alive, the GC does not move it, `===` is pointer-equal first)
   and read them without a lock; give the identity cases their own `constrain`
   method; resolve a per-task fact once per task, not once per consumer.
   Before and after (2 ranks x 32 threads): foreign planner 57-60 -> 19-24 us,
   owner 196-260 -> 154 us per rank-scoped task. Also: a plain unscoped task is
   not cheaper to plan than a rank-scoped one (`mpi_preplace!` makes it one), and
   two equal immutable scopes are `===`, so a test cannot tell them apart by identity.

95. **Check that the A/B baseline has the same defaults, not only the same commits.**
   A 4-thread MPI A/B showed my branch 20 % slower than the cluster's `int4`
   (154-162 vs 128-135 us per plain `@spawn`), and four more jobs went into
   bisecting my own commits before a build of the actual parent commit measured
   the same 154-161 us. `int4` was the integration branch *plus* the
   free-threads-default branch: it differed from my parent in one default that
   removes a thread hand-over under MPI. Export the exact parent commit as the
   baseline first, and compare a candidate against `git merge-base` before
   suspecting the candidate. Also: `proc_states(uid)` is a read-locked dict
   lookup, so keep per-scheduler handles that the hot path needs (the queued-task
   counter) as fields of the per-processor state.

96. **Plain `@spawn` cost is the submitter's latency, so measure submit and drain
   separately before optimizing anything.** `eager_submit_internal!` schedules
   inline (`schedule_ready!` runs `schedule_one!` on the caller), and execution
   overlaps with it. Timing the submit loop and the wait for the last result
   separately (`bench/spawn_split.jl`) gave ~100-225 us per task for submission
   and ~2.5 us for the drain; a plain `Threads.@spawn` is 3-15 us. Phase timers
   on a throw-away branch (`bench/spawn_phase_timers.jl`) then split a 4-thread
   Fulda task of ~108 us into `estimate_task_costs!` ~21-28, `fire_tasks!`
   ~24 (of which launching the pooled task 8-9), options+defaults ~11, select +
   reserve ~11, edge wiring ~12, thunk+DRef ~6. Only the estimate grows with the
   thread count (50 us at 64). The Opterons are ~7x slower per primitive than a
   laptop, where the same task is 28 us, so locate costs on the cluster. A sampling
   profile of the same loop is mostly idle `psynch_cvwait` samples and says nothing.

97. **Never `git add -f` a directory that is ignored on purpose.** `LLM_info/` is
   excluded through `.git/info/exclude`; force-adding it for a throw-away
   instrumentation commit made it *tracked* on that branch, so the next
   `git checkout` of the working branch deleted all 985 files from disk (the
   copies in the commit saved them: `git checkout <tmp-branch> -- LLM_info` then
   `git reset -- LLM_info`). Stage only the source files you mean (`git add src
   test`), check `git show --stat` before leaving the branch, and never publish
   or sync a revision with `git archive` of such a commit.

98. **`Task` + `sticky = false` still starts on the creator's thread when the
   creator is bound to one; use `Dagger.set_task_migratable!`.** The lend
   service was created by the main task (thread 1) and every job it made
   inherited thread 1, so a thief with 16 idle threads ran its jobs one after
   the other, 150-480 ms late, behind the driver's planning: lending made a
   4 x 16 run *slower* (2.2-2.8 s against 2.0 s) and the symptom looked like
   protocol latency. A trace (`MPIExt.LEND_TRACE_ON[] = true`) showed jobs
   started in one millisecond and finished one task duration apart. In
   isolation: 15 busy tasks from `Task`+`schedule` on the main task used 1
   thread (1.5 s), the same from a spawned task or `Threads.@spawn` 15
   (0.1 s). Fixed running service with `set_task_migratable!`; the broadcast
   relay (`start_bcast_relay!`) is made the same way and is bound to the thread
   of whoever called `accelerate!` (not changed here).

99. **A request/grant protocol must ask per free slot, not per peer.** The
   thief registered `want = free` at a peer once and waited for the answer
   before asking again, so 15 slots that finished together shared one new
   request per round trip. Keep the demand registered at each peer equal to
   the free slots (requests carry a delta), and have the victim add it.
   Also: the first lend of a process compiles grant, job and result paths
   while 15 tasks wait (first round 2.3-2.6x ideal, warm rounds 1.2x);
   `enable_delegation!` lends one task to itself to warm them. Always report
   the first round separately from the warm ones.

100. **Measure a contended-lock change on the workload that contends, not on
   the micro-benchmark.** Spinning 2000 times before parking on `state.lock`
   (`lock_brief!`) is invisible in plain `@spawn` (79-88 us at 16 threads,
   within noise) and worth 7-9 % on a 16-thread Datadeps Cholesky (1.62-1.68 ->
   1.51-1.53 s), because there the completion threads and the submitter collide
   on the lock. Judging it on the spawn loop alone would have kept it off, as
   the first 300-spin trial did. Compare interleaved, alternating revisions in
   one allocation (lessons 91, 95), and take both numbers before deciding.

101. **A scheduler cache keyed by object address pollutes a frequency cache,
   and the address of an immutable value is not a stable fact.** Keying the
   compatible-set cache of a Datadeps union scope by address looked like the
   fix for `objectid` hashing 64 members per task, and it measured better in
   the first regions: then each region built 64 new scope objects, the
   frequency-ordered cache (256 entries) filled with entries that never hit
   again, and the new frequency-1 keys were evicted at once, so every later
   task missed (98 us per task at 64 threads, flat Datadeps Cholesky). A key
   must be a function of the *contents* (here: length, first and last member)
   with `===` as the hit test. Separately: `ccall(:jl_value_ptr, ..., (Any,), x)`
   returns the address of a temporary, the same one for every call, when the
   compiler knows `x`'s concrete immutable type (Julia 1.13: three different
   `S1(...)` values gave one address; mutable structs were fine). Whether it is
   right depends on inference, so it is not a way to ask "is this the same
   object" for a struct. `MPIExt.mpi_object_address` uses the same call; it
   works only while its argument is statically `Any`/abstract.

102. **Equal is not identical when a hit has to compare 64 members.** Datadeps
   built one `UnionScope` per processor, equal in content, so the scheduler's
   lookup ended each hit in a member-by-member `===` of 64 scopes (22 us per task
   at 64 threads on Fulda, 1.3 us on a laptop). One shared object per memory
   space (`share_equal_scope`) makes the same test a pointer comparison. When a
   cache hit costs O(n), look at how many distinct-but-equal keys the producer
   makes.

103. **A local fast path is the compatible set, not a rewrite of the decision.**
   `local_fast_place!` (src/sch/fastpath.jl) takes the cached compatible set of
   the task's scope and only then skips the general path's per-candidate work,
   so it can never place a task where the general path would not: it requires
   every member to be a `ThreadProc` of this process and a
   `DistributedAcceleration`, leaves `proclist/single/time_util/alloc_util/
   occupancy`, compute/result scopes and chunk-scope restrictions to the general
   path, and asks `iscompatible_func/arg` once (the methods dispatch on the
   processor type only). Pressure counters are resolved once per scheduler
   (`FastCounters`) and read without a lock. Anything the path cannot decide
   returns `nothing` with nothing reserved. `JULIA_DAGGER_LOCAL_FASTPATH=0`
   turns it off; compare against that, not only against an older revision
   (commit 406efc4 alone moved the Cholesky 1.60 -> 1.52 s at 64 threads).

104. **Do not run `do_tasks` on the placing thread.** Starting a pooled task per
   placed task costs the submitter 9-10 us (16 threads) to 21-32 us (64), which
   looks like the next thing to remove. `JULIA_DAGGER_FIRE_INLINE` (enqueue on
   the placing thread, tried and dropped) made plain `@spawn` 65 -> 91 us at 16
   threads, 119 -> 93 us at 64, and the 64-thread Cholesky 1.36 -> 1.58 s:
   `do_tasks` costs 24-67 us per plain task and 79-134 us per Datadeps task, of
   which the kick loop is 57-97 us (a union-scoped task is never "all pinned"),
   and the pooled task does that work off the submitter. Cut the kick loop
   before moving `do_tasks`.

105. **A one-slot memo thrashes on interleaved signatures.** `populate_defaults!`
   refilled its per-task memo (up to 24 hashed LFU lookups) whenever the signature
   changed; a task placing a Cholesky sees potrf/trsm/syrk/gemm interleaved, so
   merge options + defaults cost 10.1 us (16 threads) / 17.5 us (64) against 2 us
   for plain `@spawn`. Phase timers that split a phase in two (signature vs merge)
   found it in one run.

106. **Count the transfers before blaming the transport.** A 2-rank MPI Cholesky
   (8192^2, 256^2 tiles) was 3x slower than one rank, and the obvious suspects
   (relay, receive polling, 0.7 GB/s fresh-array receives) were all innocent:
   the run issued 5458 tile transfers where the 528 panel tiles need one each.
   `add_writer!(...; copied=true)` clears the history and records the copy as
   the only entry (and the owner), so `compute_remainder_for_arg!` saw a
   replica made by a copy as "owned by the destination" and copied the same
   unchanged tile back, and forward again, every time the *other* space read it
   (a tile read alternately from two ranks was copied on every read). The
   currency set `arg_current` already knew better; the planner only used it for
   the final write-back. Consult it first, but only for an argument that no
   other argument overlaps (a write through a view reaches `arg_current` only
   via the overlap bookkeeping), and only for entries that were recorded (a
   copy into an argument with no record starts from the origin only if nothing
   was written elsewhere). Found by counting copies per (argument, source,
   destination) in a throw-away branch and printing the copy sequence of the
   most-copied tile; the same count on any transfer-heavy slowdown takes minutes
   and rules a whole class of causes in or out. After the fix 513 transfers
   remain, 2 x 32 went 5.8 -> 2.65 s and 4 x 16 5.6 -> 3.2 s; what is left is
   per-rank planning of every task (sched phase 1.0 s against 0.56 s on one
   rank), not transfers or the broadcast relay (mean 80 us per message, ~10 %
   busy).

107. **Overlapping the plan with the run is worth more than a cheaper plan, and
   a backlog sampler shows it in one run.** Uniform (MPI) planning withheld every
   task of a region until it had planned them all, deliberately (ranks stall each
   other in the planning rendezvous while tasks run). Sampling
   `Sch.EAGER_STATE[].running_count` every 10 ms in the probe showed the
   outstanding-task count at 1 until 0.97 s (one rank) or 1.5 s (two ranks) and
   only then a backlog: planning and execution ran back to back, on one rank
   too. Streaming a large region (> 2048 tasks, `DATADEPS_UNIFORM_DEFER_MAX`)
   through the asynchronous submitter took the 8192^2 Cholesky from 1.84 to
   1.59 s (1 x 64), 2.67 to 2.12 s (2 x 32), 3.25 to 2.75 s (4 x 16). What is
   left is planning: with the tasks running, the planning thread spends 0.8 of
   ~1.5 s in `compute_remainder_for_arg!`, 0.7 s of that in ~500 blocking
   aliasing exchanges for the remote slots (1.4 ms each while the other ranks
   compute, 0.47 ms when they do not). A batched exchange for all slots a
   partition needs is the next step; measure per-section planning-thread time
   (timers around the sections of `distribute_task!`) before touching it.

108. **A "fixed" or "equal" claim names the size and configuration, and orphaned
   processes contaminate timings.** The MPI Cholesky result "same as shared
   memory" held at 4096^2 (816 tasks) and failed at 8192^2
   with two ranks (1.84 s on 1 x 64, 5.7 s on 2 x 32): the effect needs enough
   transfers to show, and nobody had measured several ranks at the larger size.
   Cite the problem size, rank x thread layout and baseline with every such
   statement. Check it against the raw log, not a summary grep: two failing-test
   lines were missed this way. Before timing, and after every `scancel`, look for
   orphaned test processes (`LLM_info/bench/stray_check.sh`): leftovers up to
   1 day 22 hours old, one at 75 % CPU, were found on four nodes after the
   "quiet" benchmark rounds. Reaping scripts must act only on the processes
   they just signalled: the first `stray_reap.sh` also SIGKILLed young
   processes of live jobs.

109. **Re-measure a "known" bottleneck on the current head before building on
   it, and check what a probe's timed section contains.** Lesson 107 put ~0.7 s
   of a 2-rank Cholesky into ~500 blocking aliasing exchanges. After the copy fix
   and streaming landed, the same `HIER_TIMING` run showed the whole `aliasing()`
   call at 29 ms of 1.95 s, and batching them (`schedule_slot_window!`) moved
   2 x 32 by nothing (it only helped 4 x 16, by 10-14 %). A phase probe that
   timed `copyto!(DA, A)` inside the measured block mixed its remote-slot-heavy
   regions into the Cholesky numbers and made the batch look like a 3x gain in
   the copy-to loop: read the per-region `HIER_TIMING` lines, or put only the
   measured operation between the barrier+`GC.gc()` and the timer. Placing a
   task ahead of the tasks before it is sound only for a scheduler that never
   reads planning state (`datadeps_schedule_is_stateless`).

110. **A result filter must not be able to hide the launcher's own failure, and
   Slurm's `--ntasks-per-node` is Open MPI's slot count.** A benchmark job asked
   for one task per node and then started two ranks with `mpiexec`; Open MPI
   refused ("There are not enough slots available"), every run ended in seconds,
   and the job printed only its own `round=... :` headers, because the output
   was piped through `grep -E "^CHOLMPI|rror"` and the refusal contains neither.
   The job looked healthy and finished in two minutes (the same shape as lesson
   32). Request at least as many tasks per node as the largest rank count you
   will launch (the threads are set separately with `-t`, and `-c` need only
   cover them), and have the wrapper print a marker such as `NO RESULT` (plus the
   last lines of raw output) when the filter matched nothing, so an empty result
   is a visible failure. Unset `-x VAR` names only warn; export them or drop them.

111. **A field typed with an abstract struct makes every property access on it a
   dynamic dispatch, and a runtime `<:` on a value's type is not a pointer
   compare.** `DTaskPair.spec::DTaskSpec` has no concrete parameters, so each
   `pair.spec.options`, `spec.fargs` (a custom `getproperty` with a `where`
   parameter) and `is_typed(spec)` in the submit path recomputed the spec's type
   parameters at run time, and `valuetype(arg) <: Thunk|DTask|Chunk` per argument
   per task went through the type-matching machinery too: ~93 % of the submit
   path's samples sat in `jl_f__compute_sparams` / `exists_subtype`, invisible in
   the source because the lines look innocent. Fixes: dispatch once onto the
   concrete type (`eager_launch!(pair) = _eager_launch!(pair.spec, pair.task)`),
   read plain fields with `getfield`, and test values with `isa` (a pointer
   compare for concrete `DTask`/`Thunk`/`ThunkID`). A custom `getproperty` that
   branches on a type parameter cannot be rewritten as a field read without
   losing the constant folding the typed path relies on, so use barriers.
   To find these: profile many repetitions, keep the stacks whose leaf frames
   are in `jl_f__compute_sparams`, `exists_subtype`, `jl_subtype`,
   `jl_apply_generic`, and attribute each to the innermost Dagger source line
   (`LLM_info/bench/prof_dyn.jl`); after the fixes that share fell to ~4 % on
   both plain `@spawn` and a Datadeps Cholesky.

112. **Size a scheduler bottleneck with Little's law, and measure kernel time under
   load, not alone.** Mean concurrency = throughput x mean task duration. A
   timeline of when each tile kernel starts and ends (`LLM_info/bench/occupancy.jl`)
   gave, for Cholesky 8192/256 on 64 threads of the Opterons: 5984 tasks in
   1.137 s = 5250 tasks/s, kernels 4.69 ms each when 64 run at once (28.1 thread-
   seconds against 17.1 sequentially: shared FPUs and memory make them 1.64x
   slower), so the expected concurrency is 24.6 and the measured one was 24.7 of
   64. The runtime was starved by task *throughput* (~190 us of serial work per
   task), while a backlog of ready tasks existed and every sampled "CPU load" or
   "idle thread" figure pointed elsewhere. Consequences: the ideal wall time is
   `K_parallel / threads` (0.44 s here), not `K_sequential / threads` (0.27 s);
   to keep T threads busy with tasks of duration d the runtime must sustain
   T/d tasks per second (13.6k/s here); and any change should be judged by the
   throughput it adds, which the same timeline reports directly. Sampling profiles
   undercount native BLAS time, and `K/(wall x threads)` is the cheap sanity check
   (87 / 79 / 52 / 24 % at 8 / 16 / 32 / 64 threads).

113. **A serial pre-pass is the cheapest place to find a dynamic dispatch, because
   `HIER_TIMING` already times it and nothing hides it.** Under MPI the hierarchical
   planner's pre-scan runs on one thread before the first task starts, so its cost is
   pure latency: 195 ms for 5984 tasks (33 us/task), identical in every rep. A profile
   limited to its frames named `type_may_alias(typeof(arg))`, `unwrap_inout`, the
   `deps` loop, `ArgumentWrapper` and the `_cached_get!` closures: `arg` came from
   `fargs[i]` and `fetch(...; raw=true)`, so it was abstract and each of those was a
   dynamic dispatch (lessons 21, 111). One function barrier per argument
   (`_collect_arg_meta!` -> `_collect_arg_value!`) halved it to 91-101 ms in 12/12
   reps. Before optimizing a per-region phase, read the per-phase times of
   `JULIA_DAGGER_HIER_TIMING=1` for the configuration you care about (1 rank MPI,
   32 threads): the profile's planner section here was dominated by warm-up and the
   copy-to region and pointed at aliasing, which the phase timer showed is 12 ms of 1.5 s.

114. **A scope that names one processor is not always a request for that thread.**
   Datadeps gives every task an `ExactScope` so that the plan can say *which rank* runs
   it, and under MPI `FREE_MPI_THREADS` read that as a promise to stay on the named thread
   ("pinned tasks and Datadeps tasks stay bound"). A ready task then waited for its planned
   thread while others sat idle: 1 rank x 32 threads ran Cholesky 8192/256 at mean
   concurrency 14.4 of 32 against 22.6 in shared memory, with identical kernel time per
   thread (0.73 s median) and a balanced total, so every per-thread total hid it. The
   discriminator was the *idle fraction per thread* (54 % vs 27 % of the wall), not the
   per-thread sum. Before blaming the planner or the runtime path, rule them out in this
   order, each with one probe: plan everything first and look at the execution window
   (`JULIA_DAGGER_UNIFORM_DEFER_MAX=1000000`), run the same DAG with 16x16 tiles so the
   kernels vanish (pure runtime throughput: MPI was *faster* than shared memory), and read
   per-thread kernel seconds (`bench/occupancy.jl`). Letting a tagged task run on any free
   thread of its rank (`FREE_DATADEPS_THREADS`, **opt-in**) took 1 x 64 from 1.33-1.53 s
   to 1.04-1.10 s. An earlier version of this lesson blamed the switch for stencil-test deadlocks at 2 ranks x 4
   threads; those runs shared a node with other jobs (lesson 115) and the full 4 x 2 suite passes with the switch on.
   It stays off because Barnes-Hut 2 x 16 steady steps are ~8 % slower with it. Also: piping a launcher such as `test/mpi/run.jl` into `head -N` closes
   its stdout after N lines and leaves the ranks hanging; redirect to a file instead.

115. **Concurrent jobs of one user are packed onto one node; a result from a shared
   node is not a result.** Six of my jobs (8 cores each, `--bind-to core`) landed on
   `compute-00` together while compute-01..07 were idle. On the 32-FPU Opterons that is
   enough contention for the MPI deadlock detector to fire in a suite that passes alone:
   the *validated baseline* "hung" and a switch-off build showed the same 7 errors as the
   switch-on build, and I spent hours on a deadlock theory (check-order race, owner tasks
   never started) that explained a loaded node, not the code. Before attributing a
   cross-rank failure to a change: look at `squeue -o "%i %j %N"` for what else shares the
   node, run the correctness suites and every A/B with `--exclusive -w compute-0N`, one per
   node, and always include the known-good baseline in the same setting. A standalone
   extract of a test section is not the suite (mine could not finish in 30 minutes even
   on the baseline); if the baseline fails in your instrument, the instrument is wrong.

116. **Planning and submission are two stages; if one thread runs both, their costs add.**
   The flat Datadeps planner (every single-owner region, i.e. all shared-memory work)
   submitted each task itself right after planning it: ~70 us of planning plus ~100 us of
   scheduler submission on the Fulda Opterons = ~157 us per task, 6,400 tasks/s, while the
   MPI path, which already handed batches to a submitter task, ran the same 5984
   near-empty tasks 1.6x faster (0.58 vs 0.95 s) despite doing strictly more work per task.
   The comparison that exposed it was *tiny kernels*: with 16x16 tiles the kernels vanish
   and wall time is pure runtime throughput, which real-sized kernels hide behind idle
   threads. When you route work through an async queue, keep every task producer on the same
   queue (copy and free tasks go through the *scoped* `task_queue`, so set that scope, do not
   just wrap the planner's own queue), flush before anything that needs a task to exist (a
   `fetch` of an unlaunched `DTask` argument), and drain in a `finally`. Real Cholesky
   8192/256 went 1.07 -> 0.92 s at 32 threads and 1.20 -> 0.93 s at 64.

