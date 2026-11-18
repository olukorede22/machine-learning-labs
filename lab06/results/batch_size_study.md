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
