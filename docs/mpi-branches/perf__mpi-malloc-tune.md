# `perf/mpi-malloc-tune` (merged into `perf/task-overhead`, commit `0a17013`)

> Keep freed tile memory in the heap when the MPI acceleration starts (`mallopt`, glibc only): `JULIA_DAGGER_MALLOC_TUNE=0` turns it off.

## Why
Datadeps slots and received tiles are 512 KB libc buffers allocated and freed per region; glibc maps each fresh and trims the heap top, so every use pays page faults (~0.7 GB/s into fresh memory).

## Measured
Interleaved A/B (same settings as environment variables): 2 nodes x 1 rank x 32 threads, Cholesky 8192/256, medians 1.563 / 1.457 / 1.486 / 1.393 -> 1.312 / 1.351 / 1.366 / 1.353 s (-16 / -7 / -8 / -3 %); one node 4 x 16 -3 / -3.5 / -2 %. Never worse. `OPTIMIZATIONS.md` 39, 41.

## Main files
`src/memory-spaces.jl` (`tune_malloc!`), `ext/MPIExt.jl`
