"""Task 7: time forward and backward separately, batch 32, 100 runs, median (after warm-up)."""
import csv
import statistics
import time
from pathlib import Path
from nn import MLP, set_seeds, get_data

RES = Path(__file__).resolve().parents[1] / "results"
RUNS = 100


def main():
    set_seeds()
    RES.mkdir(exist_ok=True)
    Xtr, _, _, _, Ytr, _ = get_data()
    X, Y = Xtr[:32], Ytr[:32]
    rows = []
    for w in (16, 64):
        m = MLP(w)
        m.forward(X); m.backward(Y)          # warm-up
        fwd, bwd = [], []
        for _ in range(RUNS):
            t0 = time.perf_counter(); m.forward(X); t1 = time.perf_counter()
            m.backward(Y); t2 = time.perf_counter()    # uses the cache from the forward just run
            fwd.append(t1 - t0); bwd.append(t2 - t1)
        f, b = statistics.median(fwd) * 1e6, statistics.median(bwd) * 1e6
        rows.append({"width": w, "forward_us": f"{f:.1f}", "backward_us": f"{b:.1f}",
                     "backward/forward": f"{b / f:.2f}"})
        print(rows[-1])
    with open(RES / "timing.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
