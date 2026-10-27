@testset "NUMA map" begin
    nodes = Dagger.numa_nodes()
    cpus = Dagger.thread_cpus()
    if Sys.islinux()
        @test length(cpus) == Threads.maxthreadid()
        @test all(c -> c >= 0, [cpus[t] for t in 1:Threads.nthreads() if Threads.threadpool(t) === :default])
        if !isempty(nodes)
            map = Dagger.numa_threads(; cpus, nodes)
            # every known thread lands in exactly one domain
            tids = reduce(vcat, values(map); init=Int[])
            @test length(tids) == length(unique(tids))
            @test issubset(tids, 1:Threads.maxthreadid())
        end
        @test Dagger.threads_pinned() isa Bool
    else
        @test isempty(cpus) && isempty(nodes)
        @test !Dagger.threads_pinned()
    end
    # the pure part: a fake map
    fake_nodes = Dict(0 => [0, 1, 2, 3], 1 => [4, 5, 6, 7])
    fake_cpus = [0, 5, 2, -1, 7]            # threads 1..5
    @test Dagger.numa_threads(; cpus=fake_cpus, nodes=fake_nodes) == Dict(0 => [1, 3], 1 => [2, 5])
end
