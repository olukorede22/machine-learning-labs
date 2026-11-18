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
