# `shelved/slot-header-prefetch` (shelved, not merged)

> One header message per rank pair for the remote slots of a Datadeps window (`JULIA_DAGGER_SLOT_HEADER_PREFETCH=0` turns it off).

## Why
Under MPI each remote slot costs one blocking owner-to-destination header exchange (`mpi_endpoint_alloc`); on Fulda this was 0.15-0.43 s of a 1.2 s schedule phase at 4 x 16 (8 ms at one rank), Cholesky 8192/256.

## What it did
`schedule_slot_window!` collects (argument, destination) pairs; `prefetch_slot_headers!` (MPIExt) exchanges their headers, all sends posted before the first receive; `generate_slot!` passes the header through the scoped value `SLOT_HEADER`. Leaf dense slots only. Tests in `test/mpi/slot_window.jl`.

## Measured effect: none
Interleaved A/B, exclusive nodes: 4 x 16 medians 1.86-1.93 -> 1.87-1.98 s, 2 x 32 1.68-1.77 -> 1.72 s. The saved slot time reappears as waiting elsewhere in the pipelined schedule. See `LLM_info/OPTIMIZATIONS.md` section 35.

## Main files
`ext/MPIExt.jl`, `src/datadeps/aliasing.jl`, `src/datadeps/hierarchical.jl`, `test/mpi/slot_window.jl`
