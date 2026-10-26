# `perf/datadeps-free-threads`

> Under MPI, let a Datadeps task run on any free thread of its rank, instead of waiting for the thread the plan named.

**Builds on:** `perf/task-overhead` (the free-threads default of `FREE_MPI_THREADS`). Changes: `ext/MPIExt.jl` (`mpi_thread_requested`), `src/submission.jl` (`FREE_DATADEPS_THREADS`, `free_datadeps_threads!`), `src/Dagger.jl` (`JULIA_DAGGER_FREE_DATADEPS_THREADS`, default `1`).

## Why

A Datadeps task names its processor (an `ExactScope`) because the plan has to say which rank runs it. `FREE_MPI_THREADS` treated every exact scope as a promise to stay on that thread, so under MPI a ready Datadeps task waited for its planned thread while other threads of the rank were idle. Shared memory has no such promise: its tasks carry a memory-space scope and the scheduler balances them.

Found with the occupancy timeline (`LLM_info/bench/occupancy.jl`, 1 rank x 32 threads, Cholesky 8192/256, 5984 tasks): the per-thread kernel time was balanced (31 threads used, median 0.73 s, max 0.91 s), but each thread was idle for 54 % of the wall (27 % in shared memory), mean concurrency 14.4 of 32 against 22.6. Ruled out on the way: the planner (it is not the limit: with the whole region planned first the execution window still reached only 17.6), the runtime path (tiny 16x16 kernels: 1-rank MPI 0.67 s, shared memory 0.94 s for the same 5984 tasks), `FREE_MPI_THREADS` itself (no effect), and thread imbalance in total (balanced).

## What it adds

`mpi_thread_requested(opts)` is false for a tagged (Datadeps) task when `FREE_DATADEPS_THREADS[]` is on, so such a task starts migratable and `mpi_execute_inner` runs it unbound. An explicit pin without a tag is still a promise. `JULIA_DAGGER_FREE_DATADEPS_THREADS=0` (or `Dagger.free_datadeps_threads!(false)`) restores the old behaviour; it has no effect when `FREE_MPI_THREADS` is off.

## Measured effect

Fulda, Cholesky 8192/256, interleaved, 3 rounds x 5 samples, medians, off -> on: 1 rank x 64 threads 1.326 / 1.478 / 1.529 -> 1.077 / 1.100 / 1.043 s (-26 %); 2 x 32 1.666 / 1.751 / 1.754 -> 1.577 / 1.564 / 1.521 s (-10 %); 4 x 16 1.846 / 1.881 / 1.950 -> 1.835 / 1.832 / 1.926 s (-1.5 %). 1 x 32 occupancy: wall 1.60 / 1.62 -> 1.16 / 1.12 s, mean concurrency 14.3 -> 19.9 of 32. Never worse in any round.

## Tests

`test/mpi/free_threads.jl`: a tagged exact-scope task is bound with the switch off and unbound with it on; an untagged exact scope stays bound either way. Full MPI suites (2 x 4, 4 x 2) and the Distributed suite are run on the final head.

## Limits and notes

With several ranks the gain shrinks (4 x 16: -1.5 %): there the time is in cross-rank transfers and per-rank planning, which this does not touch. A task that runs unbound may land on thread 1, which plans; `DATADEPS_AVOID_DRIVER` only steers the planner's label, not the thread an unbound task starts on.
