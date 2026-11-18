## profile_breakdown

| phase | profiler_seconds | profiler_share_pct | wallclock_seconds | wallclock_share_pct |
|---|---|---|---|---|
| data_loading | 13.4491 | 61.8010 | 13.4566 | 61.7027 |
| forward | 4.7919 | 22.0196 | 4.8049 | 22.0322 |
| backward | 3.0967 | 14.2298 | 3.1111 | 14.2655 |
| optimizer_step | 0.4243 | 1.9497 | 0.4361 | 1.9995 |

- One epoch of cnn, batch 128, 0 workers on cpu.
- Largest share: data_loading (61.8% by torch.profiler).
- profiler_* = torch.profiler record_function ranges; wallclock_* = independent perf_counter cross-check. Top operators: results/profile_top_ops.txt.
