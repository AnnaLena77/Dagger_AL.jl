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

## Prototype status (2026-11-01, `LLM_info/bench/tileflow/`, benchmark-only)

TileFlow (`tf!(sp, f, TIn(id), TRW(id))`) implements the design for whole tiles: replicated version/holder table, owner = rank of the first written tile, deterministic transfers (sender = lowest holder, per-pair sequence tags), own tasks as `@light` tasks, one progress task polling MPI requests.
* **Correct** for Cholesky, LU (no pivoting), GEMM and a Jacobi stencil against sequential references (errors 7e-18 .. 4e-16) on 1, 2, 4 ranks locally and on 2 and 4 Fulda nodes with 1 and 2 ranks per node (up to 8 ranks, real network transfers).
* **Cholesky 8192/256, one Fulda node, round 1** (TileFlow vs the current Dagger MPI path, same allocation): 1 x 64 1.42-1.51 vs 1.33-1.57 s; 2 x 32 1.05-1.48 vs 1.53-1.60 s; 4 x 16 1.43-2.06 vs 1.75-1.95 s.
* **Walk time** (all ranks walk all tasks): 0.04 s at 1 x 64 and 2 x 32; 0.3 s at 4 x 16 was garbage collection of the receive buffers allocated in the walker (AGENTS.md lesson 138); a buffer pool removes it.

### Measured A/B, one Fulda node, Cholesky 8192/256 (job 9789: 3 rounds x 3 TileFlow repetitions vs 5 Dagger repetitions, same allocation, pooled buffers)

| layout | TileFlow min / median / max (n=9) | Dagger MPI min / median / max (n=15) |
|---|---|---|
| 1 x 64 | 1.02 / 1.43 / 1.47 s | 1.27 / 1.50 / 1.58 s |
| 2 x 32 | 0.97 / 1.10 / 1.39 s | 1.42 / 1.55 / 1.77 s |
| 4 x 16 | 1.20 / 1.61 / 2.03 s | 1.67 / 1.80 / 1.99 s |

Equal at 1 x 64, 29 % faster at 2 x 32, 11 % at 4 x 16 (medians). Whole tiles only; the first repetition of each process includes the cold buffer pool. Across nodes (round 1 of job 9790, 8192/256): 2 x 1 x 64 TileFlow 0.89-1.18 s vs Dagger 1.28-1.36 s; 4 x 1 x 64 1.00-1.55 s vs 1.72-1.85 s; Dagger's own path gets slower with more nodes. At 16384/512 the Dagger path fails (`BUGS.md` B26).

### Where the Dagger MPI path spends its time (job 9800, Fulda, Cholesky 8192/256, `JULIA_DAGGER_HIER_TIMING=1`, warm regions of 5984 tasks, 2026-11-01)

| 1 x 64 | ms | share |
|---|---|---|
| phase 4 schedule | 710-1220 | 85-90 % |
| phase 1 prescan | 82 | 7 % |
| phase 1 aliasing | 13 | 1 % |
| dag + partition + epilogue | ~28 | 2 % |
| planning total | 833-1345 | |
| measured run | 1186-1549 | |

Planning is almost the whole run; per task 120-200 us in phase 4. Aliasing is not the problem. At 2 x 32: phase 4 830-910 ms, epilogue 155-210 ms, 497 data moves (87-110 ms). Next: split phase 4 (`distribute_task!`) into task build / processor choice / data moves / launch / cross-rank agreement before choosing between a tile path and cheaper launching.

### Current standing (same node type, same code, Cholesky 8192/256, 1 x 64, memory interleaved where stated)

Distributed (1 process x 64 threads) 0.90 s; Dagger MPI 1.48 s; TileFlow prototype 0.59 s. Dagger MPI is still slower than Distributed; the prototype is not in `src/`.
