# `perf/interval-tree-balance`

> Keep the interval tree that Datadeps uses for memory overlap balanced, so spans arriving in address order no longer build a linked list. Shared memory and MPI alike.

**Builds on:** `perf/task-overhead`; the change is in `src/utils/interval_tree.jl` and independent of MPI.

## Why

Aliasing information, the remainders of a copy and the lookup of overlapping arguments are kept in an interval tree. It was an unbalanced binary search tree on the span start. Spans arrive in increasing address order whenever arrays or tiles are allocated one after another, and the tree then had depth n: inserting 16000 increasing spans cost 33.6 us each against 0.1 us for the same spans shuffled. A profile of a Datadeps region over 20000 distinct arrays put 85 % of its planning samples in the tree; the Cholesky planner spent about 40 % of its samples there.

## What it adds

A treap: node priorities keep it balanced (expected depth O(log n)), `insert!` rotates, `delete!` looks on both sides for equal starts (rotations can separate them). Priorities come from a per-tree counter (splitmix64), not from the span, so every MPI rank that performs the same inserts builds the same shape.

## Measured effect

Laptop, 4 threads: Datadeps with independent arrays 73.5 -> 30.3 us/task. Fulda, 64 threads, interleaved against the parent, 2 rounds: 1760-1908 -> 4366-4434 tasks/s (568 / 524 -> 229 / 226 us per task); independent `@spawn`, layered dataflow and Datadeps chains unchanged (within noise); Cholesky 8192/256 neutral in shared memory (1.115 / 1.183 vs 1.194 s) and under MPI (2 x 32 1.970 / 1.995 vs 1.958 / 2.005 s, 4 x 16 2.349 / 2.361 vs 2.387 / 2.355 s): its trees only hold 1024 tiles.

## Tests

`test/interval_tree.jl` (1274 assertions): invariants (search order, heap order, `max_end`), exact deletes with equal starts against a list, overlap queries and span subtraction against brute force, same operations give the same shape. It fails on the old tree.

## Main files

`src/utils/interval_tree.jl`, `test/interval_tree.jl`
