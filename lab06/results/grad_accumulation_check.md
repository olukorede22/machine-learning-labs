## grad_accumulation_check

| run | final_train_loss | final_val_acc | test_acc | peak_mem_mb |
|---|---|---|---|---|
| batch 256 (direct) | 0.2936 | 0.8918 | 0.8888 | 631.3370 |
| batch 64 x accumulate 4 | 0.2938 | 0.8918 | 0.8884 | 633.7987 |
| batch 64 (no accumulation) | 0.2386 | 0.9044 | 0.9008 | 634.0813 |

- Per-epoch curves, direct batch 256 vs 64x4: max |train-loss difference| = 1.82e-04, max |val-accuracy difference| = 0.0004.
- Same seed, same data order, and each micro-batch weighted by its share of the group, so the two runs match up to floating-point rounding.
- Accumulation trades speed for memory: the micro-batch needs less activation memory.
