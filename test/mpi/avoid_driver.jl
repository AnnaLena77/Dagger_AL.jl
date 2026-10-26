# Uniform Datadeps planning keeps unscoped tasks off the driver thread (thread 1 of every rank): a task pinned to the
# thread that is busy planning cannot start before planning ends, and the first task of a graph is usually on its
# critical path (`DATADEPS_AVOID_DRIVER`).
tid_of!(a::Vector{Float64}) = (a[1] += 1; Threads.threadid())

@testset "Datadeps avoids the driver thread" begin
    if Threads.nthreads() >= 3
        old = Dagger.DATADEPS_AVOID_DRIVER[]
        try
            for avoid in (true, false)
                Dagger.DATADEPS_AVOID_DRIVER[] = avoid
                arrs = [zeros(1) for _ in 1:128]
                tasks = Dagger.DTask[]
                Dagger.spawn_datadeps() do
                    for a in arrs
                        push!(tasks, Dagger.@spawn tid_of!(Dagger.InOut(a)))
                    end
                end
                tids = Int[fetch(t) for t in tasks]      # the result of a task reaches every rank
                @test length(tids) == 128
                avoid && @test !(1 in tids)              # none on the driver thread, on any rank
                avoid || @test 1 in tids                 # the default order does use it (the test can tell the two apart)
                @test length(unique(tids)) >= 2          # and the work is spread over threads
            end
        finally
            Dagger.DATADEPS_AVOID_DRIVER[] = old
        end
    end
end
MPI.Barrier(comm)
