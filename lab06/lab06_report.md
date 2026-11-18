# Lab 6 Report: Training pipelines, profiling and MLflow

**Student:** Olukorede Odeinde  **Student ID:** 74363

All runs use the CNN from Lab 3 on Fashion-MNIST, on CPU, with seed 42. Settings are in `config.yaml`, and every run is logged to MLflow. Tables and graphs are in `results/`.

## 1. Profiling (`results/profile_breakdown.png`)
One epoch (CNN, batch 128, 0 workers) split into phases: data loading 61.8%, forward 22.0%, backward 14.2%, optimizer step 2.0%. A second timing method (wall-clock) agrees within 0.1 percentage points. The bottleneck is data loading, not computation, because with 0 workers the main process must prepare every batch itself before the model can use it.

## 2. DataLoader workers (`results/workers_study.png`)
With 0 workers the loop waited for data 71.6% of the time, at 1,455 samples/s. Adding workers cut the wait to about 2-8%. The best setting was 4 workers at 3,416 samples/s, which is 2.35x faster than 0 workers. With 8 workers throughput fell to 2,187 samples/s, because more workers than the CPU can use only add process start-up and data-copying overhead and compete with training for cores. The laptop was also running other programs, so individual timings are noisy; the overall trend is clear, but the exact ranking of 1, 2 and 8 workers is not reliable.

## 3. Batch size (`results/batch_size_study.png`)
Throughput rose from 2,406 samples/s at batch 32 to about 3,000 at batch 128, then stayed flat, because the CPU is already fully used. Peak memory barely changed (633 to 660 MB). On CPU this measures the whole process, which is mostly fixed overhead, so a GPU would show a bigger effect. Accuracy fell as the batch grew (test 0.9010 at batch 32 vs 0.8782 at batch 512). The learning rate was the same for every batch size, so larger batches made fewer updates per epoch; the accuracy difference mixes batch size with the number of update steps.

## 4. Schedulers and optimizers (10 epochs)
| Run | Test accuracy |
|---|---|
| Adam, constant LR | 0.8973 |
| Adam + StepLR | 0.9020 |
| Adam + CosineAnnealingLR | 0.9022 |
| Adam + ReduceLROnPlateau | 0.8973 |
| SGD+momentum + CosineAnnealingLR | 0.9142 |

Decaying the learning rate helped Adam a little (about +0.5 points). ReduceLROnPlateau gave exactly the same result as constant LR, because validation accuracy never stalled long enough for it to lower the rate (final LR stayed 0.001). SGD with momentum and cosine decay was best, 1.2 points above Adam with cosine. This is one seed, and differences of a few tenths of a percent are within noise, but this gap is larger than that. The Cosine run took longer (652 s vs about 330 s) most likely because the laptop was busy with something else, not because cosine is slower.

## 5. Gradient accumulation
Batch 64 with 4 accumulation steps matched a direct batch of 256: the largest difference in training loss was 1.8e-4 and in validation accuracy 0.0004. Each micro-batch's loss is weighted by its share of the group so the gradients match. Batch 64 without accumulation behaves differently (0.9008 test), so the equivalence is with batch 256. Memory was the same (632-634 MB) in my measurement, because on CPU the process memory is dominated by fixed overhead. The benefit would show on a GPU, where smaller micro-batches need less activation memory. The cost of accumulating is more sequential steps.

## 6. Checkpoint resume
Training was stopped after epoch 3 and resumed from `last.pt`. Epochs 4-6 matched the uninterrupted run exactly (loss difference 0.0, accuracy difference 0.0). The checkpoint stores the model, optimizer, scheduler and all random-number states, including the data-shuffling generator, so the run continues as if it was never interrupted.

## Files
`lab06.py`, `config.yaml`, `requirements.txt`, `results/` (tables, graphs, `summary.md`), `project_proposal.md`.