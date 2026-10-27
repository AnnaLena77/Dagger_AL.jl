# The rank gate (`Dagger.mpi_rank_gate!`): a concurrency limit and priorities for the tasks a rank runs on its free threads.
rg_active = Threads.Atomic{Int}(0)
rg_max = Threads.Atomic{Int}(0)
function rg_tile(i::Int, ms::Float64)
    a = Threads.atomic_add!(rg_active, 1) + 1
    Threads.atomic_max!(rg_max, a)
    t0 = time_ns(); while (time_ns() - t0) < ms * 1e6; end      # a kernel: occupies its thread, never yields
    Threads.atomic_sub!(rg_active, 1)
    return i
end
rg_urgent(x::Int) = x + 1
rg_maxseen() = rg_max[]

@testset "Rank gate" begin
    nthr = Threads.nthreads(:default)
    if nranks > 1 && nthr >= 3
        Dagger.check_uniformity!(false)
        sc = Dagger.scope(mpi_rank=nranks - 1)
        try
            Dagger.mpi_rank_gate!(true; cap=nthr, reserve=1)
            # warm up with the same shape (queued tiles, an urgent task): the first call of each path compiles for a while
            wt = [Dagger.@spawn(scope=sc, return_type=Int, rg_tile(i, 2.0)) for i in 1:2 * nthr]
            sleep(0.01)
            wu = Dagger.@spawn scope=sc return_type=Int priority=1 rg_urgent(1)
            fetch(wu); foreach(fetch, wt)
            fetch(Dagger.@spawn(scope=sc, return_type=Int, rg_maxseen()))
            Threads.atomic_xchg!(rg_max, 0)
            n = 6 * nthr
            tiles = [Dagger.@spawn(scope=sc, return_type=Int, rg_tile(i, 15.0)) for i in 1:n]
            sleep(0.05)                                                    # the tiles are queued; now an urgent one
            t0 = time()
            u = Dagger.@spawn scope=sc return_type=Int priority=1 rg_urgent(1)
            ur = fetch(u)
            urgent_latency = time() - t0          # taken before the first `@test` of the testset compiles inside the window
            @test ur == 2
            @test fetch.(tiles) == collect(1:n)
            # the backlog is n * 15 ms / (nthr - 1) s: the urgent task did not wait for it
            @test urgent_latency < 0.5 * (n * 0.015 / (nthr - 1))
            # at most cap - reserve ordinary tasks ran at once, on the rank that owns them
            mx = fetch(Dagger.@spawn(scope=sc, return_type=Int, rg_maxseen()))
            @test mx <= nthr - 1
            @test mx >= min(2, nthr - 1)                                   # and the gate did let several run together
        finally
            Dagger.mpi_rank_gate!(false)
            Dagger.check_uniformity!(true)
        end
    else
        @test_skip false
    end
    Dagger.barrier(; comm)
end
