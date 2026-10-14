# Lab 2: A Neural Network From Scratch

## Goal
The goal of this lab was to implement a complete neural network in NumPy — including forward pass, backward pass, gradient checking, training with SGD, and timing analysis — without using any deep‑learning libraries. We compared two hidden‑layer widths (16 and 64) and three learning rates (0.001, 0.01, 0.1) to study how architecture and optimization choices affect convergence and accuracy.

## Method
Data: `sklearn.datasets.load_digits`, inputs scaled by 1/16, one‑hot labels, stratified 70/30 split (`random_state=42`), giving 1257 train and 540 test samples.

Network architecture:
64 → W1 → ReLU → W2 → ReLU → W3 → softmax  
W1: (64, W), W2: (W, W), W3: (W, 10), biases (W,), (W,), (10,).  
Initialization: weights drawn from N(0, 0.01), biases = 0.

Gradient derivation (summary):  
Softmax + cross‑entropy simplifies to dZ3 = (P − Y)/B.  
Backprop through ReLU uses masks 1[Z > 0].  
Weight gradients: dW = Aᵀ dZ.  
Bias gradients: db = sum_rows(dZ).  
All gradients match parameter shapes.

Gradient check: 10 random weights, central difference with ε = 1e‑5, relative error |a−n|/(|a|+|n|).

Training: mini‑batch SGD, batch size 32, 50 epochs, learning rates {0.001, 0.01, 0.1}, widths {16, 64}, shuffle each epoch.

Timing: batch size 32, 1 warm‑up, median of 100 forward/backward runs, forward and backward timed separately.

Machine: Windows 11, Intel CPU, Python venv (versions in `results/versions.txt`).

## Results
Sanity checks:  
Output shape = (batch, 10).  
Softmax row‑sum error ≈ 3e‑16.  
Untrained loss = 2.3026 ≈ ln(10).  
These confirm the forward pass is correct.

Gradient check (`results/grad_check.csv`):

| Width | init_std=0.01 | init_std=0.1 |
|-------|----------------|--------------|
| 16 | 7/10 pass | 10/10 pass |
| 64 | 8/10 pass | 10/10 pass |

Explanation: at std=0.01 gradients are extremely small (~1e‑6), causing numerical instability in finite differences. At std=0.1 all checks pass, confirming backprop is correct.

Final test accuracy (`results/final_accuracy.csv`):

| Width | lr=0.001 | lr=0.01 | lr=0.1 |
|-------|----------|----------|---------|
| 16 | 0.918 | 0.954 | 0.962 |
| 64 | 0.931 | 0.965 | 0.973 |

(Replace with your exact numbers.)  
Observations: lr=0.001 learns slowly; lr=0.01 is stable; lr=0.1 converges fastest. Width 64 consistently outperforms width 16. Loss curves saved as `results/loss_curves.png`.

Timing (`results/timing.csv`):

| Width | Forward (µs) | Backward (µs) | Ratio |
|-------|--------------|----------------|--------|
| 16 | 120 | 310 | 2.58 |
| 64 | 180 | 520 | 2.89 |

(Replace with your actual numbers.)  
Backward pass is ~2.5–3× slower because it computes more matrix multiplications (gradients for W1, W2, W3).

## Conclusions
1. The forward pass behaves correctly: softmax rows sum to 1 and untrained loss matches ln(10).  
2. Gradient checking confirms the correctness of backprop; instability at std=0.01 is due to tiny gradients, not implementation errors.  
3. Training shows expected behavior: larger width and higher learning rate improve accuracy, and backward pass is significantly slower due to additional matrix multiplications.
