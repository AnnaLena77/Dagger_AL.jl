# `perf/mpi-space-processors-cache`

> The sorted processor list of an MPI memory space is built once, not on every call.

**Builds on:** integration/mpi-all (0f1f488)

## Why

`processors(::MPIMemorySpace)` built its list on every call and sorted it by `short_name` strings (the order must be deterministic: callers take `first(processors(space))` and every rank must pick the same). Each comparison allocates strings, so the cost grew faster than the thread count, and the call runs several times per task in Datadeps planning, copies and frees. Found by an allocation profile of 1000 independent Datadeps tasks: `short_name` made a third of the MPI backend's extra allocations. Stock Dagger 0.22.7 has the same code (BUGS B12).

## What it changes

`ext/MPIExt.jl`: the sorted list is built once per space (processors are fixed once the acceleration is up) and kept in a table that is read without a lock and replaced as a whole; each call returns a copy, since callers may modify it.

## Measured effect

| | before | after |
|---|---|---|
| `processors(space)`, 4 threads (laptop) | 1.2 us, 63 allocations | 28 ns, 2 |
| `processors(space)`, 16 threads (laptop) | 23 us, 1086 allocations | 31 ns, 2 |
| 1000 independent Datadeps tasks, 16 threads (laptop) | 243.9 us, 2783 allocations per task | 146.1 us, 601 (shared memory 67.8 us) |
| **Fulda, MPI Cholesky 4096^2 in 256^2 tiles, 1 rank x 64 (interleaved A/B)** | **0.76-0.79 s** | **0.36-0.38 s** (shared memory 0.35-0.40 s) |
| **Fulda, same, 2 ranks x 32 (interleaved A/B)** | **0.79-0.89 s** | **0.48-0.50 s** |

## Tests

Targeted MPI subset (spawn, elision, free threads, Datadeps thread placement, struct transport, rank-local graphs) locally with the switches off (the local loop meant to switch them on did not, see lesson 101); full MPI suite on Fulda with free threads off and on, and with free threads + elision (0 failures each). Probe: `LLM_info/bench/task_overhead.jl`, allocation profile `chol_profile.jl` (not in git).

## Main files

`ext/MPIExt.jl`
