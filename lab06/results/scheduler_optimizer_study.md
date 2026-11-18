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
