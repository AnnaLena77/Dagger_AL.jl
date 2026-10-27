# Slot-window batching (`schedule_slot_window!`): the aliasing info of the remote slots of a window of
# tasks is exchanged at once. Same results as one exchange per slot, fewer exchanges, and a task that
# takes an in-region value still works inside a window.
slot_add!(A, B) = (A .+= B; nothing)
slot_scale(x) = 2x

function slot_region(window::Int, nblocks::Int)
    old = Dagger.DATADEPS_SLOT_WINDOW[]
    Dagger.DATADEPS_SLOT_WINDOW[] = window
    try
        A = zeros(Blocks(4, 4), Float64, 4nblocks, 4nblocks)
        B = zeros(Blocks(4, 4), Float64, 4nblocks, 4nblocks)
        copyto!(A, [Float64(i + 10j) for i in 1:4nblocks, j in 1:4nblocks])
        copyto!(B, ones(4nblocks, 4nblocks))
        fut = nothing
        Dagger.spawn_datadeps() do
            for i in 1:nblocks, j in 1:nblocks
                Dagger.@spawn slot_add!(InOut(A.chunks[i, j]), In(B.chunks[i, j]))
            end
            # a task consuming the value of an earlier in-region task
            t = Dagger.@spawn slot_scale(3)
            Dagger.@spawn slot_add!(InOut(A.chunks[1, 1]), In(B.chunks[1, 1]))
            fut = Dagger.@spawn slot_scale(t)
        end
        return collect(A), fetch(fut)
    finally
        Dagger.DATADEPS_SLOT_WINDOW[] = old
    end
end

@testset "Slot window batching" begin
    if MPI.Comm_size(comm) >= 2
        # `@opcounter` creates its counter in the module that uses it: the per-slot broadcast is
        # counted in MPIExt, the window's exchange in Dagger
        mpiext = Base.get_extension(Dagger, :MPIExt)
        cat = :aliasing_bcast_send_yield
        for c in (cat, :slot_window_ainfos)
            c in Dagger.OPCOUNTER_CATEGORIES || push!(Dagger.OPCOUNTER_CATEGORIES, c)
        end
        sent(c) = Dagger.allreduce_value(+, Dagger.opcounter(c === cat ? mpiext : Dagger, c); comm)
        slot_region(1, 2)   # warm up
        slot_region(32, 2)
        before1 = sent(cat)
        res1, v1 = slot_region(1, 4)
        per_slot = sent(cat) - before1
        before32 = sent(cat); batched0 = sent(:slot_window_ainfos)
        res32, v32 = slot_region(32, 4)
        windowed = sent(cat) - before32
        @test res32 == res1
        @test v32 == v1 == 12
        @test sent(:slot_window_ainfos) > batched0          # the window did exchange slots
        @test windowed < per_slot                           # ... and that replaced per-slot exchanges
        expected = [Float64(i + 10j) + 1 + (i <= 4 && j <= 4 ? 1 : 0) for i in 1:16, j in 1:16]
        @test res32 == expected
    end
end
Dagger.barrier(; comm)
