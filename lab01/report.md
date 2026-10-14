# Lab 1: Environment and first system measurements

## 1. Goal
The goal of this lab is to evaluate two baseline machine learning models — Logistic Regression and Random Forest — not only on predictive accuracy but also on their system costs. These measurements help determine which deployment environments (Cloud, Edge, Mobile, TinyML) each model is suitable for.

## 2. Method
- Environment: Python 3.11.8, pinned `requirements.txt`, versions in `results/versions.txt`.
- Data: Breast Cancer Wisconsin dataset, 70/30 stratified split, `random_state=42`, seeds set to 42.
- Models: LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=100).
- Training time: median of 5 runs after 1 warm-up.
- Latency: median of 100 single-sample predictions (after warm-up).
- Size: measured via `joblib.dump`.
- Memory: peak RSS via `memory_profiler` (10 ms sampling).
- Machine: Windows 10, ~16 GB RAM (RSS baseline ~106 MB).

## 3. Results

### Baseline accuracy + system costs
| Model | Test acc. | Train time (s) | Latency (ms) | Size (bytes / KB) | Peak RSS train / infer (MB) |
|---|---|---|---|---|---|
| LogReg | 0.9415 | 0.4052 | 0.0700 | 1055 B / 1.03 KB | 106.9 / 106.9 |
| RandomForest | 0.9357 | 0.1921 | 2.9803 | 290889 B / 284.07 KB | 107.9 / 107.9 |

### Deployment budget comparison
| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| LogReg | ✔ fits | ✔ fits | ✔ fits | ✔ fits |
| RandomForest | ✔ fits | ✔ fits | ✔ fits | ✘ too large (284 KB > 100 KB) |

## 4. Three conclusions (own words)
1. Logistic Regression is extremely lightweight: tiny model size, very fast inference, and low memory usage. It fits all deployment environments, including TinyML.
2. Random Forest offers slightly higher training speed but produces a much larger model. Although its latency is still low, its size prevents deployment on TinyML devices.
3. Both models easily meet Cloud, Edge, and Mobile constraints, but only Logistic Regression is small enough for ultra‑constrained environments.
