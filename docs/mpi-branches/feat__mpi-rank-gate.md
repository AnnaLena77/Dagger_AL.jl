# `feat/mpi-rank-gate` (merged into `perf/task-overhead`, commit `5dc092b`, opt-in)

> A ready queue with a concurrency limit and priorities for the tasks a rank runs on its free threads: `JULIA_DAGGER_MPI_RANK_GATE=1`, `Dagger.mpi_rank_gate!`, `Options.priority`.

## Why
Under MPI every task has occupancy 0, so all ready tasks start at once as Julia tasks (first come first served). 250 walk tiles on 8 threads put a small urgent task (the LET export) behind ~0.9 s of backlog; the application saw one task per tile 60-80 % slower under MPI.

## What it does
Wraps the call of the task's function in `mpi_run_unbound`: at most `cap - reserve` ordinary tasks run at once, `priority > 0` tasks may use the reserved slot, waiting tasks hold no thread and get slots by priority then arrival. Only tasks on a free thread (`FREE_MPI_THREADS`) are gated.

## Measured
Probe `LLM_info/bench/tiles_latency.jl`, 2 ranks x 5 threads, 160 tiles of 10 ms: urgent latency 32-211 ms without, 0-1 ms with the gate and `priority=1` (step +24 %: one thread kept free); without priority the urgent task waits for the whole backlog; reserve 0 starves the scheduler (60-212 ms). Test `test/mpi/rank_gate.jl`. Not measured in the application yet. Lesson 123.

## Main files
`ext/MPIExt.jl`, `src/acceleration.jl`, `src/options.jl`, `src/sch/Sch.jl` (priority as the second queue key), `test/mpi/rank_gate.jl`
