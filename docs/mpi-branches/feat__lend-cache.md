# `feat/lend-cache` (merged into `perf/task-overhead`, commit `4657028`, opt-in)

> Lending: a thief-side argument cache and a size cap: `enable_delegation!(; slots, cache_bytes, cache_min_bytes, max_bytes)`, `delegation_cache_stats()`, `lend_cache_clear!()`.

## Why
A lent task serialized all its arguments every time: 250 walk tiles sharing one tree and one LET would ship them 250 times. (A task with a chunk argument can be lent: arguments are resolved to values first.)

## What it does
An argument of at least `cache_min_bytes` travels once per (owner, thief) pair, later tasks send a `LendRef`. The owner models each thief's cache and sends put/evict/clear operations the thief applies in order (one communicator and tag: ordered; encode + send under one lock). Keyed by object identity: cached arguments must not be written while lendable tasks run. `max_bytes` keeps tasks with large not-yet-held arguments at home.

## Measured
3 ranks, MPICH, one flooded owner, 12 tasks sharing 400 KB: 2 puts + 10 references, 0.8 MB sent against 4.8 MB; tests for a write after clearing, eviction, the cap (`test/mpi/delegation.jl`). Not measured in the application yet.

## Main files
`ext/MPIExtDelegation.jl`, `src/acceleration.jl`, `test/mpi/delegation.jl`
