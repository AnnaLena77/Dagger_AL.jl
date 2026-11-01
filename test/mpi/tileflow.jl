# TileFlow over MPI: Cholesky, LU, GEMM, a hopping chain and a failing kernel, each compared with the sequential result (rank 0 gathers).
include(joinpath(@__DIR__, "..", "tileflow_defs.jl"))
@testset "TileFlow over MPI ($(MPI.Comm_size(comm)) ranks)" begin
    tileflow_checks(MPI.Comm_rank(comm), MPI.Comm_size(comm))
end
