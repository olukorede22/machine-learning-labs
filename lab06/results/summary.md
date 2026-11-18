# Lab 6 results summary

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

## workers_study

| num_workers | epoch1_s | epoch_s | samples_per_s | data_wait_s | compute_s | data_wait_pct |
|---|---|---|---|---|---|---|
| 0.0000 | 43.6730 | 37.8100 | 1454.6250 | 27.0630 | 10.7470 | 71.5760 |
| 1.0000 | 15.2010 | 25.8390 | 2128.5600 | 1.9990 | 23.8400 | 7.7360 |
| 2.0000 | 18.1980 | 22.6480 | 2428.4540 | 0.5790 | 22.0690 | 2.5570 |
| 4.0000 | 16.8370 | 16.1010 | 3416.0130 | 0.4410 | 15.6590 | 2.7420 |
| 8.0000 | 14.9800 | 25.1520 | 2186.7050 | 3.6200 | 21.5320 | 14.3920 |

- With 0 workers the loop waits for data 72% of the time, so data loading is the bottleneck there.
- Best throughput: 4 workers, 3416 samples/s (2.35x the 0-worker speed).
- More workers than CPU cores, or more than the model can consume, only add process start-up and inter-process copy overhead (see epoch1_s).

## batch_size_study

| batch_size | samples_per_s | peak_mem_mb | mem_delta_mb | final_train_loss | final_val_acc | test_acc |
|---|---|---|---|---|---|---|
| 32.0000 | 2406.3525 | 633.1597 | 97.2349 | 0.2209 | 0.8998 | 0.9010 |
| 64.0000 | 2843.7803 | 632.8771 | 99.7827 | 0.2386 | 0.9044 | 0.9008 |
| 128.0000 | 3004.9750 | 634.4950 | 102.0682 | 0.2625 | 0.8970 | 0.8919 |
| 256.0000 | 2871.1574 | 636.3996 | 103.1700 | 0.2936 | 0.8918 | 0.8888 |
| 512.0000 | 2906.8395 | 659.5092 | 127.3160 | 0.3261 | 0.8816 | 0.8782 |

- Throughput 2406 -> 2907 samples/s from batch 32 to 512.
- The learning rate is the same for every batch size, so accuracy differences mix the effect of batch size with fewer optimizer steps per epoch.
- peak_mem_mb = CUDA allocator peak on GPU, or peak process RSS on CPU; mem_delta_mb = peak minus the value at the start of the run.

## scheduler_optimizer_study

| run | final_lr | best_val_acc | final_val_acc | test_acc | final_train_loss | total_time_s |
|---|---|---|---|---|---|---|
| adam + constant lr | 0.0010 | 0.9068 | 0.9032 | 0.8973 | 0.2079 | 326.9411 |
| adam + StepLR | 0.0001 | 0.9066 | 0.9066 | 0.9020 | 0.2194 | 352.5515 |
| adam + CosineAnnealingLR | 0.0000 | 0.9080 | 0.9066 | 0.9022 | 0.2139 | 652.1035 |
| adam + ReduceLROnPlateau | 0.0010 | 0.9068 | 0.9032 | 0.8973 | 0.2079 | 409.8912 |
| sgd+momentum + CosineAnnealingLR | 0.0012 | 0.9144 | 0.9144 | 0.9142 | 0.1575 | 305.7337 |

- Best test accuracy in a 10-epoch budget: sgd+momentum + CosineAnnealingLR (0.9142).
- Adam vs SGD+momentum, both with cosine decay: 0.9022 vs 0.9142.
- Adam lr 0.001, SGD lr 0.05 (momentum 0.9); differences of a few tenths of a percent from one seed are within noise.

## grad_accumulation_check

| run | final_train_loss | final_val_acc | test_acc | peak_mem_mb |
|---|---|---|---|---|
| batch 256 (direct) | 0.2936 | 0.8918 | 0.8888 | 631.3370 |
| batch 64 x accumulate 4 | 0.2938 | 0.8918 | 0.8884 | 633.7987 |
| batch 64 (no accumulation) | 0.2386 | 0.9044 | 0.9008 | 634.0813 |

- Per-epoch curves, direct batch 256 vs 64x4: max |train-loss difference| = 1.82e-04, max |val-accuracy difference| = 0.0004.
- Same seed, same data order, and each micro-batch weighted by its share of the group, so the two runs match up to floating-point rounding.
- Accumulation trades speed for memory: the micro-batch needs less activation memory.

## resume_check

| epoch | ref_train_loss | resumed_train_loss | ref_val_acc | resumed_val_acc | abs_loss_diff | abs_acc_diff |
|---|---|---|---|---|---|---|
| 1.0000 | 0.5295 | 0.5295 | 0.8472 | 0.8472 | 0.0000 | 0.0000 |
| 2.0000 | 0.3508 | 0.3508 | 0.8686 | 0.8686 | 0.0000 | 0.0000 |
| 3.0000 | 0.3040 | 0.3040 | 0.8840 | 0.8840 | 0.0000 | 0.0000 |
| 4.0000 | 0.2789 | 0.2789 | 0.8928 | 0.8928 | 0.0000 | 0.0000 |
| 5.0000 | 0.2607 | 0.2607 | 0.8996 | 0.8996 | 0.0000 | 0.0000 |
| 6.0000 | 0.2496 | 0.2496 | 0.8990 | 0.8990 | 0.0000 | 0.0000 |

- Training was interrupted after epoch 3 and resumed from checkpoint last.pt.
- After resuming (epochs 4-6): max |train-loss difference| vs the uninterrupted run = 0.00e+00, max |val-accuracy difference| = 0.0000.
- The checkpoint stores model, optimizer, scheduler and all RNG states (including the data shuffling generator), so the curve continues as if nothing happened. On a GPU, tiny differences from non-deterministic kernels are normal.
