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

@testset "union-scope caches: equal scope objects share one entry" begin
    ext = Base.get_extension(Dagger, :MPIExt)
    me = MPI.Comm_rank(MPI.COMM_WORLD)
    accel = ext.MPIAcceleration(MPI.COMM_WORLD)
    # scopes are rebuilt or copied on the way to the planner (equal, not identical): they must not each get a cache entry
    # (the cache used to be keyed by an address that is a fresh box per call for a statically typed immutable)
    sc0 = Dagger.scope(mpi_rank=me)
    ext.mpi_union_scope_processor(sc0); ext.mpi_union_scope_ordered(accel, sc0)
    n1 = length(ext.MPI_UNION_SCOPE_CACHE.table); n2 = length(ext.MPI_UNION_ORDER_CACHE.table)
    for _ in 1:20
        sc = Dagger.scope(mpi_rank=me)
        @test sc == sc0
        ext.mpi_union_scope_processor(sc); ext.mpi_union_scope_ordered(accel, sc)
    end
    @test length(ext.MPI_UNION_SCOPE_CACHE.table) == n1
    @test length(ext.MPI_UNION_ORDER_CACHE.table) == n2
end
