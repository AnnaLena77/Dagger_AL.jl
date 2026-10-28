# `fix/relay-migratable` + `fix/relay-placement` (merged into `perf/task-overhead`, commits `6205854`, `701042c`)

> The broadcast relay task is not bound to the driver's thread: migratable from 4 default threads on (`JULIA_DAGGER_MPI_RELAY=main|migratable|last` forces a placement).

## Why
The full MPI suite hung across two nodes on every build: a driver parked in a blocking `MPI.Allgather` kept the relay (started on the driver's thread) from forwarding a broadcast another rank waited for. Found with a task-backtrace watchdog and SIGTERM on the busy ranks. A migratable relay then hung the delegation test at 2 threads per rank (it competes with the tasks for the only free thread).

## What it does
`start_bcast_relay!` places the relay by thread count: migratable (`Dagger.set_task_migratable!`) with 4 or more default threads, the old placement below. With it: cooperative collectives `Dagger.allgather_value`/`allreduce_value`/`barrier` (branch `fix/relay-migratable`, commit `33decd4`), and the MPI tests no longer call blocking collectives.

## Measured
Two-node suite passes (2629/2613/2641/2632); delegation 4 x 2: cluster tag 0 of 4 hung, migratable 3 of 3, relay reverted 0 of 4; laptop MPICH old placement 0 of 5, migratable 1 of 5, pinned to the last thread 3 of 5, auto 0 of 5. `LLM_info/OPTIMIZATIONS.md` sections 40, 41; lessons 122, 124.

## Main files
`ext/MPIExt.jl`, `src/acceleration.jl`, `test/mpi/*.jl`
