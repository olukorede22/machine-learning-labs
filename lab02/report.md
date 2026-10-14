# Lab 2: A neural network from scratch

## 1. Goal
(What you built and what you compared: widths 16/64, learning rates, gradient correctness, forward vs backward cost.)

## 2. Method
- Data: `load_digits`, inputs / 16, one-hot labels, 70/30 stratified split, `random_state=42` (1257 train / 540 test).
- Network: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax; shapes and parameter counts (see derivation).
- Init: weights N(0, 0.01) (state whether you read 0.01 as std or variance), zero biases.
- Gradients: derived by hand (summarise the key steps in words/equations, no code).
- Gradient check: 10 random weights, central difference with eps=1e-5, relative error |a-n|/(|a|+|n|).
- Training: shuffle each epoch, batch 32, 50 epochs, SGD; lr in {0.001, 0.01, 0.1}, W in {16, 64}.
- Timing: batch of 32, 1 warm-up, median of 100 runs, forward and backward timed separately.
- Machine: (CPU, RAM, OS), versions in `results/versions.txt`.

## 3. Results
**Sanity checks:** output shape (batch,10); max |row sum - 1| = ; untrained loss = (ln 10 = 2.3026).

**Gradient check** (`results/grad_check.csv`): pass counts per width / init std.

**Final test accuracy** (`results/final_accuracy*.csv`):
| W | lr=0.001 | lr=0.01 | lr=0.1 |
|---|---|---|---|
| 16 | | | |
| 64 | | | |

Loss curves: `results/loss_curves.png` (and `loss_curves_std0.1.png`).

**Timing** (`results/timing.csv`):
| W | forward (us) | backward (us) | backward/forward |
|---|---|---|---|
| 16 | | | |
| 64 | | | |

## 4. Three conclusions (own words)
1.
2.
3.
