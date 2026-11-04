"""Task 4: engine vs PyTorch autograd vs numerical gradients on ((a*b + c)**2).relu().sum()."""
import csv
from pathlib import Path
import numpy as np
import torch
from engine import Value, vsum, graph_summary

RES = Path(__file__).resolve().parents[1] / "results"
TOL, EPS = 1e-6, 1e-6


def num_grad(f, arrs):
    out = []
    for k in range(len(arrs)):
        g = np.zeros_like(arrs[k])
        for i in range(len(arrs[k])):
            hi = [x.copy() for x in arrs]; lo = [x.copy() for x in arrs]
            hi[k][i] += EPS; lo[k][i] -= EPS
            g[i] = (f(*hi) - f(*lo)) / (2 * EPS)
        out.append(g)
    return out


CASES = {
    # name: (engine fn on lists of Value, torch fn, numpy fn)
    "((a*b + c)**2).relu().sum()  [required]": (
        lambda a, b, c: vsum([((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c)]),
        lambda a, b, c: ((a * b + c) ** 2).relu().sum(),
        lambda a, b, c: float(np.sum(np.maximum((a * b + c) ** 2, 0)))),
    "(a*b - c).relu().sum()  [relu active and inactive]": (
        lambda a, b, c: vsum([(x * y - z).relu() for x, y, z in zip(a, b, c)]),
        lambda a, b, c: (a * b - c).relu().sum(),
        lambda a, b, c: float(np.sum(np.maximum(a * b - c, 0)))),
    "(a*a + a*b + a).sum()  [a reused: gradient accumulation]": (
        lambda a, b, c: vsum([x * x + x * y + x for x, y in zip(a, b)]),
        lambda a, b, c: (a * a + a * b + a).sum(),
        lambda a, b, c: float(np.sum(a * a + a * b + a))),
}


def main():
    RES.mkdir(exist_ok=True)
    rows = []
    for seed in (42, 7):
        rng = np.random.default_rng(seed)
        arrs = [rng.normal(size=8) for _ in range(3)]
        for name, (fe, ft, fn) in CASES.items():
            vals = [[Value(float(x)) for x in a] for a in arrs]
            out = fe(*vals)
            out.backward()
            eng = np.concatenate([[v.grad for v in vs] for vs in vals])

            ts = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrs]
            ft(*ts).backward()
            tor = np.concatenate([(t.grad.numpy() if t.grad is not None else np.zeros(8)) for t in ts])

            num = np.concatenate(num_grad(fn, arrs))
            d_et, d_en, d_tn = (float(np.max(np.abs(eng - tor))), float(np.max(np.abs(eng - num))),
                                float(np.max(np.abs(tor - num))))
            rows.append({"case": name, "seed": seed, "forward_value": f"{out.data:.10f}",
                         "max|engine-torch|": f"{d_et:.3e}", "max|engine-numerical|": f"{d_en:.3e}",
                         "max|torch-numerical|": f"{d_tn:.3e}", "pass(<1e-6)": d_et < TOL and d_en < TOL})
            print(rows[-1])
    with open(RES / "gradient_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # evidence that the forward pass built a graph (first element of the required expression)
    a, b, c = Value(2.0), Value(3.0), Value(-1.0)
    g = graph_summary(((a * b + c) ** 2).relu())
    (RES / "graph_summary.txt").write_text(f"((a*b + c)**2).relu() for scalars -> {g}\n")
    print(g)


if __name__ == "__main__":
    main()
