# `feat/mpi-owner-planning`

> Owner-based planning for the MPI backend: tasks are planned by the rank that owns their output, remote inputs are named `(tile, version)`; fine-grained work is generated locally (design stage, nothing implemented yet).

**Builds on:** `perf/task-overhead` (`f27fa67`, includes `@light`). Not based on the older design-only branch `feat/datadeps-owner-computes` (`5e2ccd7`), whose levels 0-2 were overtaken by later measurements (AGENTS.md lessons 106, 107, 109, 117): aliasing exchanges and the result broadcast are not what limits 2 x 32 any more; per-rank planning of every task is.

## Why

Every rank plans every task (SPMD). Planning costs ~100-125 us per task on Fulda and does not shrink with more ranks (Amdahl), so 2 x 32 is far slower than 1 x 64 and the overhead grows with the rank count. Even a replicated *enumeration* of the task stream (~0.3 us per task, estimate) caps P x T at E x d (about 30 threads for 10 us tasks, ~15000 for 5 ms tiles). HPX does not plan foreign tasks and creates only the tasks of tiles a locality owns.

## Idea

1. **Names and owners.** A global name `(tile, version)` and a changeable owner table; the owner of the output tile plans and runs the task. Ranks learn versions by counting writes while walking the program (cheap), not by alias analysis.
2. **Two levels.** Coarse blocks are walked by every rank (owner table, transfers by last writer). The owner of a block generates its fine tasks locally as `@light` tasks; no other rank sees them.
3. **Elasticity.** Ranks join or leave only at region boundaries: `Dagger.rebalance` moves tiles and the owner table is replaced. MPI itself is static (`COMM_WORLD`); dynamic membership needs `MPI_Comm_spawn` or connect/accept and must be tested on Fulda first.
4. Additive and opt-in; the existing Datadeps/MPI path stays unchanged.

## First step (measure before building)

Benchmark-only prototype under `LLM_info/bench/` (nothing in `src/`): tiled Cholesky 8192/256 with a last-writer table, MPI Isend/Irecv and light tasks on 1 x 64, 2 x 32, 4 x 16 (one node and several), against the current MPI path, Distributed and HPX. First measure the real enumeration rate E.

## Acceptance (to be refined with the prototype)

2 x 32 and 4 x 16 within ~15 % of 1 x 64; foreign-task cost near zero; full MPI suite (2 and 4 ranks, uniformity checks) and `test/allocations.jl` green; Distributed unchanged; interleaved A/B in one allocation.

## Rules

Local branch only until validated; never push to the PR head `fix/mpi-rank-local-tasks`.

## Measured: the replicated walk is cheap (2026-11-01)

`LLM_info/bench/enum_rate.jl` (single process, no Dagger/MPI) walks a whole tiled-Cholesky task stream with the minimum owner-computes bookkeeping (owner of the output tile, holder bitmask per tile, version bump on write, transfer counted when the owner lacks the current version).
* **Laptop 2.0 ns/task, Fulda (compute-09) 36 ns/task = 28 M tasks/s**, independent of the problem size (6k to 2.8 M tasks) and of the rank count (1 to 16). Transfers stay small (nt=32: 496 / 992 / 2792 for 2 / 4 / 16 ranks over 5984 tasks).
* The limit P x T <= E x d with E = 28 M/s: **~280 threads for 10 us tasks, ~28000 for 1 ms, ~140000 for 5 ms tiles.** The earlier estimate (0.3 us/task) was 8x too pessimistic on Fulda.
* For comparison: planning costs ~100-125 us per task today (3000x more), a foreign task ~19-24 us.
* **Caveat:** this is the floor of the walk itself. A real front-end (argument handling, closure, DTask/Chunk references) for tasks a rank does not own must also be skipped or kept at ~tens of ns; otherwise the front-end, not the walk, is the limit. That is the first thing the prototype has to show.
