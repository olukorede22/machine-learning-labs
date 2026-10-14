# Lab 1: Environment and first system measurements

## 1. Goal
(2-3 sentences: what you set out to measure and why — e.g. compare a linear model and a tree ensemble on accuracy and system cost, then judge deployment fit.)

## 2. Method
- Environment: Python 3.11.8, pinned `requirements.txt`, versions in `results/versions.txt`.
- Data: Breast Cancer Wisconsin, 70/30 stratified split, `random_state=42`, seeds set to 42.
- Models: LogisticRegression(max_iter=1000), RandomForest(n_estimators=100).
- Training time: median of 5 runs after 1 warm-up. Latency: median of 100 single-sample predictions (after warm-up).
- Size: `joblib.dump`. Memory: peak RSS via memory_profiler (10 ms sampling).
- Machine: (CPU, RAM, OS).

## 3. Results
| Model | Test acc. | Train time (s) | Latency (ms) | Size (bytes / KB) | Peak RSS train / infer (MB) |
|---|---|---|---|---|---|
| LogReg | | | | | |
| RandomForest | | | | | |

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogReg | | | | |
| RandomForest | | | | |

## 4. Three conclusions (own words)
1.
2.
3.
