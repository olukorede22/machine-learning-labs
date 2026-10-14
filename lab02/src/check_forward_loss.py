"""Tasks 2-4: shapes, forward pass (rows sum to 1), untrained loss ~ ln(10)."""
import numpy as np
from nn import MLP, set_seeds, get_data, cross_entropy


def main():
    set_seeds()
    Xtr, Xte, ytr, yte, Ytr, Yte = get_data()
    print(f"train {Xtr.shape}, test {Xte.shape}, ln(10)={np.log(10):.4f}")
    for w in (16, 64):
        m = MLP(w)
        shapes = {n: getattr(m, n).shape for n in MLP.PARAMS}
        P = m.forward(Xtr[:32])
        row_err = np.abs(P.sum(1) - 1).max()
        loss = cross_entropy(m.forward(Xtr), Ytr)
        print(f"W={w}: shapes={shapes}")
        print(f"  output {P.shape}, max |rowsum-1|={row_err:.2e} (<1e-6: {row_err < 1e-6}), "
              f"untrained loss={loss:.4f}")
        assert P.shape == (32, 10) and row_err < 1e-6


if __name__ == "__main__":
    main()
