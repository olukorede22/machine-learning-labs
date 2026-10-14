"""Task 3: training time, single-sample latency, model size, peak memory."""
import csv
import statistics
import time
from pathlib import Path

import joblib
import psutil
from memory_profiler import memory_usage

from common import set_seeds, get_split, make_models

RES = Path(__file__).resolve().parents[1] / "results"
TRAIN_RUNS, INFER_RUNS = 5, 100


def time_training(build, X, y):
    build().fit(X, y)  # 1 warm-up
    times = []
    for _ in range(TRAIN_RUNS):
        m = build()
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_inference(model, x1):
    model.predict(x1)  # warm-up
    times = []
    for _ in range(INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x1)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000  # ms


def peak_rss_mb(fn):
    """Peak RSS (MiB) of this process while fn runs, sampled every 10 ms."""
    return max(memory_usage((fn, (), {}), interval=0.01, max_usage=True, include_children=False), 0)


def main():
    set_seeds()
    RES.mkdir(exist_ok=True)
    Xtr, Xte, ytr, yte = get_split()
    x1 = Xte[:1]
    rows = []
    for name, build in make_models().items():
        base_mb = psutil.Process().memory_info().rss / 2**20
        train_s = time_training(build, Xtr, ytr)
        model = build().fit(Xtr, ytr)

        # size: dump to the required path, record bytes/KB, keep a named copy
        path = RES / "model.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size
        path.replace(RES / f"model_{name}.joblib")

        lat_ms = time_inference(model, x1)
        peak_train = peak_rss_mb(lambda: build().fit(Xtr, ytr))
        peak_inf = peak_rss_mb(lambda: [model.predict(x1) for _ in range(INFER_RUNS)])
        rows.append({
            "model": name,
            "train_time_s_median": f"{train_s:.4f}",
            "infer_latency_ms_median": f"{lat_ms:.4f}",
            "model_size_bytes": size_b,
            "model_size_KB": f"{size_b / 1024:.2f}",
            "baseline_rss_MB": f"{base_mb:.1f}",
            "peak_rss_train_MB": f"{peak_train:.1f}",
            "peak_rss_infer_MB": f"{peak_inf:.1f}",
        })
    with open(RES / "system_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
