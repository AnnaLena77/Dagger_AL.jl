# MPI backend work for Dagger.jl: branch guide

This directory describes the branches of the fork `AnnaLena77/Dagger.jl` that make Dagger's MPI backend (SPMD: every rank runs
the same program and plans every task) cheaper and scalable, developed for a Barnes-Hut code that is to run on 100+ nodes.
Each branch has its own page here, named after the branch with `/` replaced by `__`.
**Use `integration/mpi-all`: it contains everything** (the last two branches, `feat/mpi-struct-raw-transport` and `perf/sched-targeted-kicks`, were merged after targeted tests; the complete `test/mpi/cpu.jl` passed on `feat/mpi-struct-raw-transport`, not yet on the merged head). The single branches exist so that each change can be reviewed, tested and
proposed upstream on its own.

All measurements are from the Fulda cluster (AMD Opteron 6272, 64 cores/node, Open MPI, Julia 1.13) unless a page says otherwise.
Every Opteron primitive is 7-20x slower than on a laptop, so numbers do not transfer between the two.

## 1. How the branches fit together

```
master
 |-- fix/mpi-space-hash            independent: Open MPI hash fix
 |-- perf/sched-cost-estimate      independent: cheap cost estimate (no MPI needed)
 |-- bench/tasks-suite             independent: the `tasks` benchmark suite
 |-- fix/mpi-rank-local-tasks      independent, superseded by feat/mpi-comm-self
 `-- perf/mpi-constants            scheduler constants: placement sort, no wake-up of non-stealers, stale cache entry
      `-- feat/mpi-owner-elision   owner elision + free MPI threads        (switches, off by default)
           `-- feat/mpi-comm-self  rank-local graphs: with_rank_local, per_rank
                `-- perf/mpi-pin-runner       MPI tasks start pinned to their thread
                     `-- fix/mpi-reftid-leak  ref-id table leak
                          `-- feat/mpi-local-datadeps       Datadeps inside a rank-local graph
                               `-- feat/mpi-elide-scoped-chunks   elision for scope-restricted chunks; MPI_FEATURES.md
                                    `-- feat/mpi-elide-darray-blocks   elision for DArray blocks
                                         `-- perf/mpi-rank-scope-cost   fix of my own scope-chain regression
                                              `-- feat/mpi-elision-stats    Dagger.elision_stats()
                                                   `-- feat/mpi-group-graphs     with_group, per_group, elision in groups
                                                        `-- perf/mpi-elide-typed-proc   -13..-14 % planning cost per foreign task
                                                             `-- feat/mpi-allgather
                                                                  |-- feat/mpi-struct-raw-transport   structs of bit arrays travel as raw buffers
                                                                  `-- perf/sched-targeted-kicks       wake only runners that can take the task
                                                                       (both merged: integration/mpi-all)
```

The stacked part is linear: each branch contains all branches above it. The four independent branches are merged into the
stack by cherry-pick (`feat/mpi-elide-scoped-chunks` carries copies of the cost-estimate and hash-fix commits).

## 2. Branch overview

| Branch | What it gives | Key number |
|---|---|---|
| `fix/mpi-space-hash` | Hash of `MPIMemorySpace` without the raw communicator handle | 19 / 47 spurious "Non-uniform value" failures on Open MPI -> 0 |
| `perf/mpi-constants` | Placement order cached per candidate set; no wake-up of runners that cannot steal; stale cache entry replaced | unscoped spawn of 64 tasks at 256 processors 1258 -> 14 ms (90x) |
| `perf/sched-cost-estimate` | Cost estimate of a task with many candidate processors made cheap (no MPI) | 553 -> 331 us per task at 64 threads |
| `bench/tasks-suite` | Benchmark suite `tasks`: the runtime's own per-task cost | provides the numbers below |
| `fix/mpi-rank-local-tasks` | Early rank-local approach (superseded) | - |
| `feat/mpi-owner-elision` | A rank that neither runs a task nor owns its data does not build it; failure-only status; free MPI threads | foreign task ~170 -> ~20 us; pinned-to-other-rank 256 tasks 66.9 -> 4.8 ms |
| `feat/mpi-comm-self` | `Dagger.with_rank_local`, `Dagger.per_rank`: graphs only this rank plans, on `COMM_SELF` | a local task costs what a shared pinned task costs |
| `perf/mpi-pin-runner` | MPI tasks start pinned to their processor's thread | dependent chain link 458 -> 284 us |
| `fix/mpi-reftid-leak` | Ref-id counters of plain tasks are dropped when done | 300 plain tasks leave <= 20 table entries |
| `feat/mpi-local-datadeps` | `spawn_datadeps()` inside a rank-local graph without breaking world ids | correctness (failed before) |
| `feat/mpi-elide-scoped-chunks` | Elision for chunk arguments with a scope restriction; `MPI_FEATURES.md` | 181 -> ~28 us per foreign task reading a restricted chunk |
| `feat/mpi-elide-darray-blocks` | Elision for tasks reading `DArray` blocks (`DTask.placed`) | 181 -> 35.9 us |
| `perf/mpi-rank-scope-cost` | Fix of a regression: scope chain only for restricted chunks | rank-scoped foreign task 123-130 -> ~60 us at 32 threads |
| `feat/mpi-elision-stats` | `Dagger.elision_stats()` / `reset_elision_stats!()` | shows why a task was planned in full |
| `feat/mpi-group-graphs` | `with_group(f, comm)`, `per_group`, `free_group!`, elision inside groups | 64 ranks: shared graph 108.6 ms vs group graph 30.4 ms per step |
| `perf/mpi-elide-typed-proc` | Typed processor in `elide_task!`, one scoped lookup in `take_ref_id!` | 19.3 -> 16.75 us per foreign task (-13 %) |
| `feat/mpi-allgather` | `Dagger.allgather(tasks)` (concrete `Vector`, `allgather(tasks, T)`), Datadeps-in-group test, docs, AGENTS lessons | application: `tree.summarize` 66 -> 20 ms at 104 ranks; cluster: gather+fetch 408 -> 171 ms at 96 ranks, 64 KB per rank |
| `feat/mpi-struct-raw-transport` | A struct holding bit arrays is sent as raw buffers automatically (other fields in the header, shared arrays and nested structs handled) | 40 MB struct between two nodes 580 -> 210 ms; one node 580 -> 174 ms |
| `fix/mpi-raw-transport-views` | Array wrappers (`SubArray`, reshaped/permuted views) are never sent as raw parts of their parent (bug of `feat/mpi-struct-raw-transport`) | `copyto!(::DArray, ::Matrix)` 2048^2, 2 ranks: 2.1-2.9 s -> 0.05-0.1 s; 8192^2 no longer takes minutes |
| `perf/mpi-tochunk-kw` | MPI task results built without rebuilding their keyword arguments (stock defect) | `tochunk` 1067 -> 450 ns; plain `@spawn` 36.5 -> 26.4 us per task |
| `perf/mpi-space-processors-cache` | Sorted processor list of an MPI memory space built once, not per call (stock defect) | MPI Cholesky 4096^2, 1 x 64: 0.76-0.79 -> 0.36-0.38 s (= shared memory); 2 x 32: 0.79-0.89 -> 0.48-0.50 s |
| `fix/mpi-elide-stub-processor` | With elision, a spectator's placeholder names the processor the owner runs the task on (uniform `chunk.processor`) | full suite with free threads + elision: 62-119 placement failures -> 0 |
| `perf/sched-targeted-kicks` | After an enqueue wake only parked runners that can take the task, at most one per task (no kick for `ExactScope`) | one process, 64 threads: unscoped 3.3k -> 5.6k tasks/s (+69 %), pinned 5.5k -> 7.2k (+32 %) |
| `perf/mpi-rank-scope-planning` | Rank scope planned as cheaply as an `ExactScope`: address-keyed lock-free scope cache, one lookup per task, `constrain(::UnionScope, ::AnyScope)` is the identity, `preferred_processor_count` | foreign rank, rank-scoped task 57-60 -> 19-24 us; owner of a rank-scoped task 196-260 -> 154 us (32 threads) |

At application level (Barnes-Hut, 1M bodies, 16 ranks x 32 threads, measured by the application team): stock Dagger 0.95 s per step,
`integration/mpi-all` without switches 0.65 s, with owner elision 0.49 s; 4M bodies, 26 ranks: 2.14 s -> 1.10 s.

## 3. Switches and environment variables

Set them on **every** rank (e.g. `mpiexec -x VAR`).

| Variable | Function | Default | Branch |
|---|---|---|---|
| `JULIA_DAGGER_ELIDE_FOREIGN_TASKS=1` | `Dagger.elide_foreign_tasks!(true)`: owner elision | off | `feat/mpi-owner-elision` |
| `JULIA_DAGGER_FREE_MPI_THREADS=1` | `Dagger.free_mpi_threads!(true)`: tasks that name no thread run on any free thread of their rank | off | `feat/mpi-owner-elision` |
| `JULIA_DAGGER_MPI_ALLGATHER=native` | `Dagger.allgather` uses the library's `MPI_Iallgatherv` instead of the hand-written rounds | rounds | `feat/mpi-allgather` |

Uniformity checking (`Dagger.check_uniformity!(true)`) switches elision off; keep it off for production runs and on in functional tests.

## 4. API added

| Call | Meaning |
|---|---|
| `Dagger.with_rank_local(f)` | Tasks spawned in `f` form a graph only this rank plans and runs (`COMM_SELF`) |
| `Dagger.per_rank(f, T)` | SPMD: every rank runs `f` as a local graph; returns one shared chunk per rank |
| `Dagger.with_group(f, comm)` | Tasks spawned in `f` form a graph on the communicator `comm` that only its members plan and run |
| `Dagger.per_group(f, comm, T)` | `per_rank` on a group |
| `Dagger.free_group!(comm)` | Release a group's bookkeeping and relay |
| `Dagger.allgather(tasks)` | Every rank gets the values of tasks owned by all ranks, in one exchange |
| `Dagger.elision_stats()` / `reset_elision_stats!()` | Counters: why a task was or was not elided |
| `Dagger.elide_foreign_tasks!(::Bool)`, `Dagger.free_mpi_threads!(::Bool)` | The two switches in code |

Usage, rules and pitfalls are in `MPI_FEATURES.md` at the repository root (on `integration/mpi-all`).

## 5. Validation status

* The complete `test/mpi/cpu.jl` passes at 2 ranks x 2 threads on Fulda (Open MPI): on `integration/mpi-all` at `a3d4650` 1263 and 1270
  assertions, 43 minutes; on `feat/mpi-struct-raw-transport` (which adds `allgather.jl`, the group-Datadeps test and `struct_transport.jl`)
  1315 and 1323 assertions on the two ranks, no failure or error, 44 minutes. The "Hit probable hang on recv" lines in these runs (22, then 152) are
  10-second timers of waits during first-call kernel compilation (A4), never the 120-second abort. The merged head with `perf/sched-targeted-kicks`
  has had the targeted tests only (798 assertions at 3 ranks).
* `allgather.jl` and `groups.jl` pass at 2, 3, 4 (and for allgather 7) ranks on MPICH 5.0.2 and Open MPI 5.0.10 (local).
* Not tested: GPU chunks in rank-local graphs and groups, MPI x GPU in general (no hardware here).
* Open issues are listed in `AGENTS.md` (lessons) and the maintainer notes; the stencil first-call compile cost (minutes for 3D/4D, a
  per-tile-signature specialization explosion) is a property of the stencil code, not of MPI scheduling.

## 6. Branches that are not pushed

* `feat/mpi-rank-local`: superseded by `feat/mpi-comm-self`.
* `perf/mpi-relay-interactive`: experiment, not recommended. `JULIA_DAGGER_MPI_RELAY_THREAD=spare` runs the broadcast relay on the spare
  thread that has no Dagger processor. It fixes late delivery of results (0.34-0.44 s -> 0.05-0.08 s when all but one thread are busy with
  non-yielding code) but takes the only free thread, so it was net worse at 3 ranks x 4 threads with 3 busy threads.

## 6b. Branches added 2026-10-24 .. 28 (all merged into `perf/task-overhead` unless marked; one note each)

| Note | What |
|---|---|
| `perf__datadeps-free-threads.md`, `perf__flat-async-submit.md`, `perf__datadeps-avoid-driver.md` | opt-in free Datadeps threads, async submission of flat regions, driver-thread rule |
| `feat__datadeps-elide.md` | Datadeps elision (opt-in, -8..-10 % Cholesky 4 x 16); `feat__datadeps-elide-foreign.md` is the shelved first prototype |
| `perf__mpi-malloc-tune.md` | `mallopt` at MPI start (-3..-16 % multi-rank Cholesky) |
| `fix__relay-placement.md` | relay not on the driver'"'"'s thread (migratable from 4 threads), cooperative collectives; fixes a two-node suite stall |
| `feat__mpi-rank-gate.md` | opt-in ready queue with a concurrency limit and priorities for MPI tasks (`priority=1`) |
| `feat__lend-cache.md` | opt-in lending: thief-side argument cache and size cap |
| `feat__numa-map.md` | thread-to-CPU / NUMA map, `pin_threads!` |
| `feat__reduce.md` | `dynamic_for` with arguments and counts, `dynamic_reduce`, `allreduce` |
| `shelved__slot-header-prefetch.md`, `shelved__task-priority.md` | measured, no gain (kept as records) |

## 7. Glossary (Dagger-internal terms)

* **Elision**: a rank skips building a task it neither runs nor needs data for; it only takes the same ids and stamps a placeholder.
* **Stub / placeholder chunk**: what a non-owner holds for the result of a task that ran elsewhere.
* **Owner**: the rank that runs a task / holds a chunk. **Participates**: the rank owns one of the task's arguments.
* **Scope** (`ExactScope`, `Dagger.scope(mpi_rank=r)`): where a task may run. A rank scope is a union of the rank's processors.
* **Communicator**: the group of ranks a graph lives on (`COMM_WORLD`, `COMM_SELF`, or a group built with `MPI.Comm_split`).
* **Relay**: a per-rank task that receives and forwards the small broadcast messages (task status, result types).
* **Datadeps**: Dagger's region with data-dependency tracking (`spawn_datadeps`).
* **Uniform ids**: every rank must take the same task and reference ids in the same order; most of the rules in `MPI_FEATURES.md` follow from this.
