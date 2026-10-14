"""Task 6: mini-batch SGD, batch 32, 50 epochs, lr in [0.001,0.01,0.1], width in [16,64]."""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nn import MLP, set_seeds, get_data, cross_entropy

RES = Path(__file__).resolve().parents[1] / "results"
LRS, WIDTHS, EPOCHS, BS = [0.001, 0.01, 0.1], [16, 64], 50, 32
# Manual says N(0, 0.01). Default = 0.01 as std; pass a value to compare, e.g. `python train.py 0.1`.
INIT_STD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
SUFFIX = "" if INIT_STD == 0.01 else f"_std{INIT_STD}"


def run(width, lr, data):
    Xtr, Xte, ytr, yte, Ytr, Yte = data
    set_seeds()
    m = MLP(width, init_std=INIT_STD)
    rng = np.random.default_rng(42)
    hist = []
    for ep in range(1, EPOCHS + 1):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), BS):
            b = perm[i:i + BS]
            m.forward(Xtr[b])
            m.step(m.backward(Ytr[b]), lr)
        tr_loss = cross_entropy(m.forward(Xtr), Ytr)
        te_loss = cross_entropy(m.forward(Xte), Yte)
        te_acc = float((m.predict(Xte) == yte).mean())
        hist.append((tr_loss, te_loss, te_acc))
        print(f"W={width} lr={lr} epoch {ep:02d}/{EPOCHS} train_loss={tr_loss:.4f} "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")
    return hist


def main():
    RES.mkdir(exist_ok=True)
    data = get_data()
    results = {(w, lr): run(w, lr, data) for w in WIDTHS for lr in LRS}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        for lr in LRS:
            ax.plot(range(1, EPOCHS + 1), [h[0] for h in results[(w, lr)]], label=f"lr={lr}")
        ax.set(title=f"W={w}, init std={INIT_STD}", xlabel="epoch", ylabel="train cross-entropy loss")
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(RES / f"loss_curves{SUFFIX}.png", dpi=150)

    with open(RES / f"final_accuracy{SUFFIX}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["width", "lr", "final_train_loss", "final_test_loss", "final_test_accuracy"])
        for (w, lr), h in results.items():
            wr.writerow([w, lr, f"{h[-1][0]:.4f}", f"{h[-1][1]:.4f}", f"{h[-1][2]:.4f}"])


if __name__ == "__main__":
    main()
