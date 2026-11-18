# Individual Project Proposal: Compact Traffic-Sign Recognition with Knowledge Distillation

**Student:** Olukorede Odeinde  **Student ID:** 74363  **Course:** ML Systems Engineering

## Problem
Driver-assistance systems and embedded cameras must recognise traffic signs quickly on limited
hardware. Large image classifiers are accurate but too big and slow for such devices. This project
builds a small traffic-sign classifier that keeps most of the accuracy of a large model while
meeting strict size and latency limits.

## Dataset
German Traffic Sign Recognition Benchmark (GTSRB): 43 sign classes, about 39,200 training images and
12,630 official test images, with strong variation in lighting, size and blur. Images are resized to
32x32 and normalised. The training images are split into train and validation sets, and the official
test set is used only for the final numbers. Classes are imbalanced (some have about 200 images,
others about 2,000). Augmentation uses small shifts, rotations and colour jitter, and no horizontal
flips, because flipping changes the meaning of signs such as "turn left" and "turn right".

## Success metric
- Top-1 accuracy on the official test set: the student reaches at least 95% and stays within 2
  percentage points of the teacher.
- Macro-F1 (to account for class imbalance), reported alongside accuracy.
- A student trained **without** distillation is the baseline, to show what distillation adds.

## Deployment constraints
- **Latency:** at most 50 ms per image (batch size 1, CPU, median of 100 runs after warm-up).
- **Model size:** the saved student model is at most 5 MB.

Only the student is deployed. The teacher is used for training and does not have to meet the limits.

## Planned optimization: knowledge distillation
- **Teacher:** ResNet-18 fine-tuned on GTSRB (about 11 million parameters, about 45 MB).
- **Student:** a small CNN with 3-4 convolution blocks (about 0.2-0.5 million parameters,
  roughly 1-2 MB at 32-bit).
- **Loss:** alpha x cross-entropy on the true labels + (1 - alpha) x T^2 x KL divergence between the
  teacher's and student's softened outputs. Starting values are T = 4 and alpha = 0.3, tuned on the
  validation set.
- **Stretch goal (if time allows):** post-training int8 quantization of the student, to further reduce
  size and latency.

## Plan
1. Data pipeline and baseline student (reusing the YAML + MLflow training setup from Lab 6).
2. Train and evaluate the teacher.
3. Train the distilled student and tune T and alpha.
4. Measure accuracy, macro-F1, size and latency against the limits and write up the results.

## Risks
Class imbalance and the German-only signs limit how well the model generalises to other countries.
If the student misses the 95% target, I will widen the student slightly while keeping it under the
size and latency limits.
