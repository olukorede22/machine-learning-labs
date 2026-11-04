"""Task 6: PyTorch graph inspection, eager vs torch.jit.trace vs torch.compile, tensor metadata,
and the cost of non-contiguous views. torch.compile is attempted; if unsupported (e.g. Windows with
torch 2.3.1, or no C++ compiler) the failure is recorded and tracing is used, as the manual allows."""
import csv
import statistics
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
RES = Path(__file__).resolve().parents[1] / "results"
torch.manual_seed(42)


def median_ms(fn, runs=100, warmup=1):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def walk_grad_fn(fn):
    seen, names, stack = set(), [], [fn]
    while stack:
        f = stack.pop()
        if f is None or id(f) in seen:
            continue
        seen.add(id(f)); names.append(type(f).__name__)
        stack.extend(nf for nf, _ in f.next_functions)
    return names


def inspect_graph_and_tensors():
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
    x = torch.randn(1000, 2)
    out = model(x)
    loss = nn.BCELoss()(out.squeeze(1), torch.randint(0, 2, (1000,)).float())
    names = walk_grad_fn(loss.grad_fn)
    traced = torch.jit.trace(model, x)
    txt = ["AUTOGRAD GRAPH (loss.grad_fn, output -> inputs):", *[f"  {i}: {n}" for i, n in enumerate(names)],
           f"  total nodes: {len(names)}", "", "TORCHSCRIPT IR (jit.trace):", str(traced.graph)]
    (RES / "graph_inspection.txt").write_text("\n".join(txt))
    print(f"autograd graph: {len(names)} nodes ->", " <- ".join(names))

    big = torch.randn(2048, 2048)
    tensors = {"input x (1000x2)": x, "Linear0.weight": model[0].weight, "Linear0.weight.t()": model[0].weight.t(),
               "output": out, "big (2048x2048)": big, "big.t() (view)": big.t(), "big[:, ::2] (strided)": big[:, ::2]}
    with open(RES / "tensor_inspection.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tensor", "shape", "dtype", "stride", "is_contiguous", "device"])
        for k, t in tensors.items():
            w.writerow([k, tuple(t.shape), str(t.dtype), t.stride(), t.is_contiguous(), str(t.device)])
            print(f"  {k:24s} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} contiguous={t.is_contiguous()}")


def chain(x):
    y = torch.sigmoid(x * 1.5 + 0.5) * torch.tanh(x)
    return y + x.abs().sqrt()


def workloads():
    tiny = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()).eval()
    wide = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).eval()
    for m in (tiny, wide):
        for p in m.parameters():
            p.requires_grad_(False)
    return {"tiny MLP 2-8-1 (batch 1000)": (tiny, torch.randn(1000, 2)),
            "wide MLP 512-512-512-10 (batch 256)": (wide, torch.randn(256, 512)),
            "elementwise chain (1M floats)": (chain, torch.randn(1_000_000))}


def eager_vs_compiled():
    rows = []
    for name, (fn, x) in workloads().items():
        with torch.no_grad():
            ref = fn(x)
            eager = median_ms(lambda: fn(x), 100, 5)
            rows.append({"workload": name, "mode": "eager", "status": "ok", "first_call_ms": "",
                         "median_ms": f"{eager:.4f}", "speedup_vs_eager": "1.00", "max_abs_diff": "0",
                         "threads": torch.get_num_threads()})
            # --- jit.trace ---
            try:
                t0 = time.perf_counter(); tr = torch.jit.trace(fn, x); first = (time.perf_counter() - t0) * 1000
                m = median_ms(lambda: tr(x), 100, 5)       # extra warm-up: profiling executor optimises on early calls
                rows.append({"workload": name, "mode": "jit.trace", "status": "ok", "first_call_ms": f"{first:.1f}",
                             "median_ms": f"{m:.4f}", "speedup_vs_eager": f"{eager / m:.2f}",
                             "max_abs_diff": f"{(tr(x) - ref).abs().max().item():.2e}", "threads": torch.get_num_threads()})
            except Exception as e:
                rows.append({"workload": name, "mode": "jit.trace", "status": f"FAILED {type(e).__name__}: {str(e)[:100]}"})
            # --- torch.compile (may be unsupported) ---
            try:
                t0 = time.perf_counter(); c = torch.compile(fn); c(x); first = (time.perf_counter() - t0) * 1000
                m = median_ms(lambda: c(x), 100, 5)
                rows.append({"workload": name, "mode": "torch.compile", "status": "ok", "first_call_ms": f"{first:.1f}",
                             "median_ms": f"{m:.4f}", "speedup_vs_eager": f"{eager / m:.2f}",
                             "max_abs_diff": f"{(c(x) - ref).abs().max().item():.2e}", "threads": torch.get_num_threads()})
            except Exception as e:
                rows.append({"workload": name, "mode": "torch.compile",
                             "status": f"UNSUPPORTED {type(e).__name__}: {str(e)[:100]}"})
    cols = ["workload", "mode", "status", "first_call_ms", "median_ms", "speedup_vs_eager", "max_abs_diff", "threads"]
    with open(RES / "eager_vs_compiled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


def noncontiguous():
    x, w = torch.randn(2048, 2048), torch.randn(2048, 64)
    xt, half = x.t(), torch.randn(2048, 1024)
    cases = [
        ("x*2+1", "contiguous 2048x2048", x, lambda: x * 2 + 1),
        ("x*2+1", "x.t() view (non-contiguous)", xt, lambda: xt * 2 + 1),
        ("x*2+1", "x.t().contiguous() then op (copy included)", xt, lambda: xt.contiguous() * 2 + 1),
        ("half*2+1", "contiguous 2048x1024", half, lambda: half * 2 + 1),
        ("half*2+1", "x[:, ::2] strided slice 2048x1024", x[:, ::2], lambda: x[:, ::2] * 2 + 1),
        ("sum(dim=0)", "contiguous", x, lambda: x.sum(0)),
        ("sum(dim=0)", "x.t() view (non-contiguous)", xt, lambda: xt.sum(0)),
        ("matmul @ w", "contiguous x", x, lambda: x @ w),
        ("matmul @ w", "x.t() view (BLAS handles transpose)", xt, lambda: xt @ w),
    ]
    rows, base = [], {}
    for op, desc, t, fn in cases:
        ms = median_ms(fn, 20, 2)
        base.setdefault(op, ms)
        rows.append({"operation": op, "input": desc, "is_contiguous": t.is_contiguous(),
                     "median_ms": f"{ms:.3f}", "slowdown_vs_contiguous": f"{ms / base[op]:.2f}"})
        print(rows[-1])
    with open(RES / "noncontiguous.csv", "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)


def main():
    RES.mkdir(exist_ok=True)
    print("torch", torch.__version__, "threads", torch.get_num_threads())
    inspect_graph_and_tensors()
    eager_vs_compiled()
    noncontiguous()


if __name__ == "__main__":
    main()
