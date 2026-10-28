# `feat/numa-map` (merged into `perf/task-overhead`, commits `7a98fa2`, `9ae124a`)

> `Dagger.thread_cpus()`, `threads_pinned()`, `numa_nodes()`, `numa_threads()`, `allowed_cpus()`, `pin_threads!()` (Linux; empty elsewhere).

## Why
The Barnes-Hut application needs to know which threads of a rank sit in which NUMA domain. `JULIA_EXCLUSIVE=1` pins thread i to CPU i-1 and ignores the process CPU mask (under `numactl --cpunodebind=2,3` the threads landed on CPUs 0-15), so several ranks per node must not use it.

## What it does
`thread_cpus` samples `sched_getcpu` on every default thread, `numa_nodes` reads `/sys/devices/system/node`, `numa_threads` maps domain => Julia thread ids, `pin_threads!` pins the default-pool threads onto the CPUs `sched_getaffinity` allows. **A scope over a domain's threads is NOT provided:** with free MPI threads a union-scoped task runs on any free thread (tasks scoped to threads 2, 3 ran on 3, 4, 5).

## Main files
`src/numa.jl`, `test/numa.jl`; probe `LLM_info/bench/numa_probe.jl`
