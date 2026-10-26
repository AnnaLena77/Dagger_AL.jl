# `perf/datadeps-avoid-driver`

> Under uniform (MPI) execution, keep the tasks of a Datadeps region that name no processor off thread 1 of every rank, the thread that plans.

**Builds on:** `perf/task-overhead`. Changes: `src/datadeps/scheduling.jl` (round-robin), `src/datadeps/hierarchical.jl` (`DATADEPS_AVOID_DRIVER`), `ext/MPIExt.jl` (`is_driver_thread_proc`).

## Why

Uniform planning pins each task to the processor the scheduler chose, round-robin over the canonical processor list, whose first entry is thread 1 of rank 0: the thread that is busy planning. A task pinned there cannot start before planning ends, and the first task of a graph is usually on its critical path (a Cholesky's first `potrf`). Occupancy timeline (`LLM_info/bench/occupancy.jl`), Cholesky 8192/256, MPI 1 rank x 64 threads: first kernel at 0.74-0.83 s, when the ~0.7 s of planning ended, against 0.06 s in shared memory, where a task's scope is the whole memory space.

## What it adds

The round-robin choice skips processors that `is_driver_thread_proc` names (thread 1) for a task without a scope of its own; a task that names a processor keeps it (removing thread 1 from the candidate list instead made tasks pinned to a tile's thread incompatible: `SchedulingException`). Bounded, so a list of only driver threads still places the task. The rule depends only on the processor, so it is the same on every rank. `JULIA_DAGGER_AVOID_DRIVER_THREAD=0` turns it off.

## Measured effect

Fulda, Cholesky 8192/256, same build, interleaved, 3 rounds, median, off -> on: 2 ranks x 32 threads 1.980 / 1.963 / 2.023 -> 1.775 / 1.828 / 1.819 s (-10 / -7 / -10 %); 4 ranks x 16 threads 2.191 / 2.230 / 2.181 -> 1.905 / 1.920 / 1.926 s (-13 / -14 / -12 %); 1 rank x 64 threads 1.768 / 1.751 -> 1.747 / 1.736 s (neutral), first kernel at 0.41 s instead of 0.83 s. Results verified in every run.

## Tests

`test/mpi/avoid_driver.jl` (2 ranks x 4 threads): with the rule no task of a 128-task region runs on thread 1, on any rank; without it thread 1 is used (so the test can tell the two apart); the work is spread over several threads.

## Limits and notes

What still delays the first task is the planner's whole-region pre-pass (prescan, aliasing, DAG, partition: ~0.27 s at 5984 tasks) that runs before anything is submitted.
