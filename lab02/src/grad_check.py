"""Task 5: numerical gradient check on 10 random weights (central difference, eps=1e-5).

Run at the manual's N(0, 0.01) init and ALSO at init_std=0.1. With std=0.01 the
first-layer gradients are ~1e-6, so float rounding in the finite difference can
dominate the relative error; this is a property of the check, not a backprop bug.
"""
import csv
from pathlib import Path
import numpy as np
from nn import MLP, set_seeds, get_data, cross_entropy

RES = Path(__file__).resolve().parents[1] / "results"
EPS = 1e-5


def check(width, init_std, n=10, seed=42):
    rng = np.random.default_rng(seed)
    Xtr, _, _, _, Ytr, _ = get_data()
    X, Y = Xtr[:32], Ytr[:32]
    m = MLP(width, init_std=init_std)
    m.forward(X)
    g = m.backward(Y)
    rows = []
    wnames = ["W1", "W2", "W3"]
    while len(rows) < n:
        name = wnames[rng.integers(3)]
        W = getattr(m, name)
        idx = tuple(rng.integers(s) for s in W.shape)
        old = W[idx]
        W[idx] = old + EPS; lp = cross_entropy(m.forward(X), Y)
        W[idx] = old - EPS; lm = cross_entropy(m.forward(X), Y)
        W[idx] = old
        num, ana = (lp - lm) / (2 * EPS), g[name][idx]
        if abs(ana) < 1e-12 and abs(num) < 1e-12:   # dead ReLU path: uninformative, resample
            continue
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        rows.append({"width": width, "init_std": init_std, "weight": f"{name}{list(idx)}",
                     "analytic": f"{ana:.6e}", "numerical": f"{num:.6e}", "rel_error": f"{rel:.3e}",
                     "pass": rel < 1e-6})
    return rows


def main():
    set_seeds()
    RES.mkdir(exist_ok=True)
    allrows = []
    for std in (0.01, 0.1):
        for w in (16, 64):
            rows = check(w, std)
            npass = sum(r["pass"] for r in rows)
            print(f"W={w} init_std={std}: {npass}/10 weights pass (need >= 9)")
            allrows += rows
    with open(RES / "grad_check.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(allrows[0]))
        wr.writeheader(); wr.writerows(allrows)


if __name__ == "__main__":
    main()
