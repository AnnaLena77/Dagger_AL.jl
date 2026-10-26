# `perf/flat-async-submit`

> Submit a large flat (single-owner, i.e. shared-memory) Datadeps region through the async submitter, so planning the next tasks overlaps with turning the previous ones into scheduler thunks.

**Builds on:** `perf/task-overhead`. Changes: `src/datadeps/queue.jl` (`distribute_tasks!` wrapper, `_distribute_tasks_flat!`, `_flush_for_value_deps!`, `DATADEPS_FLAT_ASYNC_MIN`), `src/Dagger.jl` (`JULIA_DAGGER_FLAT_ASYNC_MIN`, default `0` = off).

## Why

The flat planner is the path of every region with a single owner (all shared-memory work). It hands each task to the scheduler itself right after planning it, so planning (~70 us/task) and submission (~100 us on the Fulda Opterons) add up on one thread: ~157 us per task, ~6,400 tasks/s for 5984 near-empty tasks. The hierarchical path already overlaps the two with `AsyncEnqueueQueue`; 1-rank MPI, which uses it, ran the same tasks in 0.58-0.67 s against 0.95 s for shared memory (exclusive node, 32 threads).

## What it adds

With `JULIA_DAGGER_FLAT_ASYNC_MIN=N`, a region of at least `N` tasks (not under uniform execution, more than one thread) is planned flat as before but submitted through an `AsyncEnqueueQueue`. Ordering: copy and free tasks are spawned with `Dagger.@spawn`, i.e. through the *scoped* task queue; the scope is set to the async queue for the duration (`with_options(; task_queue=aeq)`), so they queue behind the compute tasks whose thunks they depend on. A task that takes the result of an earlier task of the region (`_unlaunched_dtask_arg`) flushes the queue first, because planning fetches it. The submitter is closed (drained) in a `finally`.

## Measured effect

Fulda, exclusive nodes. Near-empty tasks (Cholesky 512/16, 5984 tasks, 32 threads): 0.989 / 0.928 -> 0.732 / 0.694 s (-26 %); laptop 4 threads 0.131 -> 0.041 s. Real Cholesky 8192/256, one process, 3 interleaved rounds x 5 samples, medians: 32 threads 1.078 / 1.092 / 1.045 -> 0.907 / 0.905 / 0.935 s (-15 %), 64 threads 1.146 / 1.251 / 1.191 -> 0.944 / 0.917 / 0.925 s (-22 %); plain `@spawn` unchanged.

## Tests

With the option forced on for every region (`JULIA_DAGGER_FLAT_ASYNC_MIN=1`, laptop): `test/datadeps.jl` 2142 passed / 2 broken (one process) and 2143 / 2 (3 extra workers), 0 failures; array core, copyto, mapreduce, broadcast, Cholesky, LU, solve, matmul, QR, stencils, streaming, task queues: 0 failures. The full Distributed suite with the option on is the open step before making it the default.

## Limits and notes

Not used under uniform execution (an MPI rank's submission runs collectives: see `AsyncEnqueueQueue`). A submission failure surfaces on the planning task at the next hand-off or at the end of the region. The default minimum is 0 (off); 1024 tasks is the suggested default once the Distributed suite is green.
