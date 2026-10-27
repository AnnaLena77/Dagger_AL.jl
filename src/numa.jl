# Where do Julia's threads run? A thread-to-CPU map and the NUMA domain of each CPU (Linux only), for placing
# work by locality. Julia does not pin its threads: the map means something only when the job pins them
# (`JULIA_EXCLUSIVE=1`, or the launcher: `numactl`, Open MPI `--bind-to`).

"""
    thread_cpus() -> Vector{Int}

The CPU (Linux numbering, `sched_getcpu`) each Julia thread of the default pool is running on right now,
indexed by `Threads.threadid()`; `-1` for a thread that could not be asked. Empty off Linux. Sampled with a
task pinned to each thread, so it costs one scheduling round trip per thread; call it once at startup.
Use [`threads_pinned`](@ref) to find out whether the answer can be trusted.
"""
function thread_cpus()
    Sys.islinux() || return Int[]
    cpus = fill(-1, Threads.maxthreadid())
    tasks = Task[]
    for tid in 1:Threads.maxthreadid()
        Threads.threadpool(tid) === :default || continue
        t = Task(() -> (cpus[Threads.threadid()] = Int(ccall(:sched_getcpu, Cint, ()))))
        set_task_tid!(t, tid)
        schedule(t)
        push!(tasks, t)
    end
    foreach(wait, tasks)
    return cpus
end

"""
    threads_pinned(; samples=5) -> Bool

Whether every default-pool thread stayed on one CPU across `samples` samples of [`thread_cpus`](@ref) taken
while the threads are made busy: `true` means the map is stable (the job pins its threads), `false` that the
operating system may move them and a NUMA map would be a guess. `false` off Linux.
"""
function threads_pinned(; samples::Int=5)
    Sys.islinux() || return false
    first = thread_cpus()
    for _ in 2:samples
        t0 = time_ns()
        @sync for _ in 1:Threads.nthreads()
            Threads.@spawn (while time_ns() - t0 < 20_000_000; end)    # keep the threads busy so they get moved if they can
        end
        thread_cpus() == first || return false
    end
    return true
end

"""
    numa_nodes() -> Dict{Int,Vector{Int}}

CPUs of every NUMA domain, from `/sys/devices/system/node` (`Dict(0 => [0, 1, ..., 7], 1 => [8, ...], ...)`). Empty when
the system does not say (off Linux, or a container without `/sys`).
"""
function numa_nodes()
    nodes = Dict{Int,Vector{Int}}()
    base = "/sys/devices/system/node"
    isdir(base) || return nodes
    for entry in readdir(base)
        m = match(r"^node(\d+)$", entry)
        m === nothing && continue
        list = try strip(read(joinpath(base, entry, "cpulist"), String)) catch; continue end
        cpus = Int[]
        for part in split(list, ',')
            isempty(part) && continue
            if occursin('-', part)
                a, b = parse.(Int, split(part, '-'))
                append!(cpus, a:b)
            else
                push!(cpus, parse(Int, part))
            end
        end
        nodes[parse(Int, m[1])] = cpus
    end
    return nodes
end

"""
    numa_threads(; cpus=thread_cpus(), nodes=numa_nodes()) -> Dict{Int,Vector{Int}}

The Julia thread ids that run on each NUMA domain, `domain => [tids...]`, from a thread-to-CPU map and the CPU lists of the
domains. Only meaningful when the threads are pinned ([`threads_pinned`](@ref)). Threads whose CPU is unknown are left out.
"""
function numa_threads(; cpus::Vector{Int}=thread_cpus(), nodes::Dict{Int,Vector{Int}}=numa_nodes())
    out = Dict{Int,Vector{Int}}()
    for (domain, domain_cpus) in nodes
        tids = [tid for tid in eachindex(cpus) if cpus[tid] >= 0 && cpus[tid] in domain_cpus]
        isempty(tids) || (out[domain] = tids)
    end
    return out
end
