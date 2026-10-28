# `shelved/task-priority` (shelved; the `Options.priority` part is merged, commits `8c59674`, `54fd611`)

> `Options.priority` as the second key of the processor queue, and a lookahead variant of the Cholesky submission order: neither shortened the Cholesky panel chain.

## Why / measured
The potrf chain advanced at 42-49 ms per panel against ~10 ms of kernel time with half the threads idle. Priority (panel tasks first): 64 threads 1.026 / 1.047 -> 1.095 / 1.079 s, 32 threads 0.964 / 0.962 -> 0.991 / 0.956 s. Lookahead (next panel before the trailing updates): 1.080 / 1.110 -> 1.116 / 1.096 s and 0.959 / 0.948 -> 0.958 / 0.969 s. The chain is paced by task throughput (Little's law). The priority option stays because the rank gate uses it. `OPTIMIZATIONS.md` 36, lesson 118.
