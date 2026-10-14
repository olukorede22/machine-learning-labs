"""Task 2: train both baselines, write results/baseline_accuracy.csv (4 decimals)."""
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score
from common import set_seeds, get_split, make_models

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    set_seeds()
    RES.mkdir(exist_ok=True)
    Xtr, Xte, ytr, yte = get_split()
    rows = []
    for name, build in make_models().items():
        m = build().fit(Xtr, ytr)
        rows.append({"model": name, "test_accuracy": f"{accuracy_score(yte, m.predict(Xte)):.4f}"})
    with open(RES / "baseline_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
