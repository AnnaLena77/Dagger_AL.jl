# `feat/reduce` and `feat/dynamic-for-args` (merged into `perf/task-overhead`, commits `ad00ea5`, `56da43a`, `2c85d1a`)

> `Dagger.dynamic_for(f, n, args...)` (task arguments, per-worker counts), `Dagger.dynamic_reduce(f, op, n, args...)` and `Dagger.allreduce(op, tasks)` with a fixed combination order.

## Why
The application wanted a loop that starts when its inputs exist (DTask/Chunk arguments), balance information without its own trace, and reductions that are bitwise reproducible (it compares final states byte for byte).

## What it does
`dynamic_for_async(f, n, args...)` passes `args` as ordinary task arguments, calls `f(i, args...)`, and the tasks return how many indices they took. `dynamic_reduce`: chunks of a size independent of the worker count, each folded in index order, chunk results folded in chunk order. `allreduce`: allgather then a list-order fold on every rank.

## Measured
Sum bitwise identical for 1, 2, 8 workers (`test/bulk.jl` 28/28); `allreduce` identical bits on all ranks at 3 ranks (`test/mpi/allgather.jl`); nested `dynamic_for` from tasks with all threads busy: no deadlock.

## Main files
`src/bulk.jl`, `src/acceleration.jl`, `test/bulk.jl`, `test/mpi/bulk.jl`, `test/mpi/allgather.jl`
