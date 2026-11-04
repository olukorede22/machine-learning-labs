"""Task 5: tiny 2->8->1 sigmoid/BCE model on make_moons: Value engine vs PyTorch (same init, float64)."""
import csv
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from moons_common import get_data, init_params, train_value, accuracy_value, EPOCHS, LR, SEED

RES = Path(__file__).resolve().parents[1] / "results"
RUNS = 5


def train_torch(Xtr, ytr, epochs=EPOCHS, lr=LR):
    p = init_params(SEED)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(p["W1"])); model[0].bias.copy_(torch.from_numpy(p["b1"]))
        model[2].weight.copy_(torch.from_numpy(p["W2"])); model[2].bias.copy_(torch.from_numpy(p["b2"]))
    X, y = torch.tensor(Xtr, dtype=torch.float64), torch.tensor(ytr, dtype=torch.float64)
    opt, lossf, losses = torch.optim.SGD(model.parameters(), lr=lr), nn.BCELoss(), []
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X).squeeze(1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def timed(fn):
    fn(1)                                   # 1 warm-up (1 epoch)
    ts, last = [], None
    for _ in range(RUNS):
        t0 = time.perf_counter(); last = fn(EPOCHS); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), last


def main():
    RES.mkdir(exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr, Xte, ytr, yte = get_data()
    Xl, yl, Xtel, ytel = Xtr.tolist(), ytr.astype(float).tolist(), Xte.tolist(), yte.astype(float).tolist()

    t_val, (mv, loss_v) = timed(lambda e: train_value(Xl, yl, epochs=e))
    acc_v = accuracy_value(mv, Xtel, ytel)
    t_tor, (mt, loss_t) = timed(lambda e: train_torch(Xtr, ytr, epochs=e))
    with torch.no_grad():
        acc_t = float(((mt(torch.tensor(Xte, dtype=torch.float64)).squeeze(1) > 0.5).numpy() == (yte > 0.5)).mean())

    max_diff = float(np.max(np.abs(np.array(loss_v) - np.array(loss_t))))
    rows = [
        {"implementation": "Value engine (pure Python, scalar)", "epochs": EPOCHS, "median_train_s": f"{t_val:.4f}",
         "ms_per_epoch": f"{1000 * t_val / EPOCHS:.3f}", "final_loss": f"{loss_v[-1]:.6f}", "test_accuracy": f"{acc_v:.4f}"},
        {"implementation": "PyTorch (vectorised, float64)", "epochs": EPOCHS, "median_train_s": f"{t_tor:.4f}",
         "ms_per_epoch": f"{1000 * t_tor / EPOCHS:.3f}", "final_loss": f"{loss_t[-1]:.6f}", "test_accuracy": f"{acc_t:.4f}"},
    ]
    with open(RES / "training_speed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = (f"slowdown (engine / torch) = {t_val / t_tor:.1f}x ; max |loss_engine - loss_torch| over "
               f"{EPOCHS} epochs = {max_diff:.3e}")
    (RES / "training_speed_summary.txt").write_text(summary + "\n")
    print(rows[0]); print(rows[1]); print(summary)

    plt.figure(figsize=(6, 4))
    plt.plot(loss_v, label="Value engine"); plt.plot(loss_t, "--", label="PyTorch")
    plt.xlabel("epoch"); plt.ylabel("train BCE loss"); plt.legend(); plt.grid(alpha=.3)
    plt.title("make_moons, 2-8-1: engine vs PyTorch"); plt.tight_layout()
    plt.savefig(RES / "loss_engine_vs_torch.png", dpi=150)


if __name__ == "__main__":
    main()
