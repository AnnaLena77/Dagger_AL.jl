# A failure broadcast needs a valid MPI tag (BH entry 53: MPI_ERR_TAG with Open MPI, tag_ub 8388607, rank-local ids start above it)
@testset "failure broadcast tag" begin
    ext = Base.get_extension(Dagger, :MPIExt)
    @test ext.mpi_failure_tag(5) == UInt32(5)
    @test ext.mpi_failure_tag(ext.MPI_LOCAL_ID_BASE) === nothing            # rank-local graph: nobody else waits
    @test ext.mpi_failure_tag(ext.MPI_LOCAL_ID_BASE + 12345) === nothing
    @test ext.mpi_failure_tag(Int(MPI.tag_ub()) + 1) === nothing || Int(MPI.tag_ub()) + 1 >= ext.MPI_LOCAL_ID_BASE
    @test ext.mpi_failure_tag(ext.MPI_GROUP_ID_BASE + (1 << 24) + 7) isa UInt32   # group graph ids still map to their tag
end
