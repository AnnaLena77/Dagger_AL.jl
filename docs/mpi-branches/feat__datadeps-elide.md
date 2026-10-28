# `feat/datadeps-elide` (merged into `perf/task-overhead`, commits `10855ef`, `2dccac9`, `2945b67`, opt-in `JULIA_DAGGER_ELIDE_DATADEPS=1`)

> A rank skips the Datadeps tasks it does not run (implies owner elision; needs uniformity checks off). Replaces the shelved prototype `feat/datadeps-elide-foreign`.

## Measured (re-measured 2026-10-27; the earlier "no gain" was masked by other bottlenecks)
Cholesky 8192/256, interleaved, exclusive nodes: one node 4 x 16 medians 1.893 / 1.955 / 1.906 / 1.923 -> 1.699 / 1.717 / 1.686 / 1.712 s (-10 %); two nodes x 2 ranks x 16 threads 1.608 / 1.698 / 1.618 / 1.603 -> 1.473 / 1.541 / 1.522 / 1.459 s (-8.5 %); tiny tiles no effect. Full MPI suites with it on pass except tests asserting the old behaviour (gated now). A failing task still reaches every rank (failure-only status path; test `test/mpi/elide.jl` "Datadeps elision"). Stays opt-in. `OPTIMIZATIONS.md` 38.

## Main files
`ext/MPIExt.jl`, `src/submission.jl`, `src/Dagger.jl`, `test/mpi/elide.jl`
