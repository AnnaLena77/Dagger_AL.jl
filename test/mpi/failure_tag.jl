# A failure broadcast needs a valid MPI tag (BH entry 53: MPI_ERR_TAG with Open MPI, tag_ub 8388607, rank-local ids start above it)
@testset "failure broadcast tag" begin
    ext = Base.get_extension(Dagger, :MPIExt)
    @test ext.mpi_failure_tag(5) == UInt32(5)
    @test ext.mpi_failure_tag(ext.MPI_LOCAL_ID_BASE) === nothing            # rank-local graph: nobody else waits
    @test ext.mpi_failure_tag(ext.MPI_LOCAL_ID_BASE + 12345) === nothing
    @test ext.mpi_failure_tag(Int(MPI.tag_ub()) + 1) === nothing || Int(MPI.tag_ub()) + 1 >= ext.MPI_LOCAL_ID_BASE
    @test ext.mpi_failure_tag(ext.MPI_GROUP_ID_BASE + (1 << 24) + 7) isa UInt32   # group graph ids still map to their tag
end

f_nothing_fits() = 1
@testset "SchedulingException names the task and the candidates" begin
    other = (MPI.Comm_rank(MPI.COMM_WORLD) + 1) % 2
    msg = Dagger.with_rank_local() do
        t = Dagger.@spawn scope=Dagger.scope(mpi_rank=other) f_nothing_fits()
        try fetch(t); "no error" catch e; sprint(showerror, e) end
    end
    @test occursin("No processors available", msg)
    @test occursin("f_nothing_fits", msg)
    @test occursin("0 candidate processor(s)", msg)
end
