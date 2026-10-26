# `feat/datadeps-elide-foreign` (shelved, not merged)

> Let a rank skip the Datadeps tasks it does not run: `JULIA_DAGGER_ELIDE_DATADEPS=1` (off by default).

## Why

Owner elision (about 20-25 us for a foreign task) did not apply to Datadeps tasks, which carry a tag and syncdeps, so a rank planned in full every task it did not run. The elision counters show the tag is the only blocking condition (`not_failure_only` = 912 of the 912 tasks that reached the check in a 4096/128 Cholesky).

## What it did

`elide_task!` accepted tagged tasks with syncdeps behind the switch (tag and syncdeps conditions only). Failures of an elided task are not reported to the other ranks, so a rank waiting for data of a failed producer would hang.

## Measured effect: none

Cholesky 8192/256, interleaved: 2 ranks x 32 threads median 1.99 -> 1.95 s (-2 %), 4 ranks x 16 threads 2.29 -> 2.32 s (+1 %); a 4096/128 run at 2 x 8 threads got slower (1.46 -> 2.05 s, one run) although the counters confirm that 512 tasks were elided on rank 0 and 400 on rank 1. Results were correct. Planning of foreign tasks is not what holds the ranks back; the failure semantics were not worth the risk. The branch is kept as a record.

## Main files

`ext/MPIExt.jl` (`elide_task!`, `mpi_failure_only`), `src/submission.jl`, `src/Dagger.jl`
