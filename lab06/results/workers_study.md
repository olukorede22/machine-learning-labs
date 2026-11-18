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
