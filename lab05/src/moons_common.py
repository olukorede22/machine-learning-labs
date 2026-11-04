"""Shared data, initial weights and the pure-Python (Value) training loop for Task 5."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from engine import Value

SEED, HIDDEN, EPOCHS, LR = 42, 8, 50, 0.5


def get_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def init_params(seed=SEED):
    """Identical starting weights for the Value engine and PyTorch (2 -> 8 -> 1)."""
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.5, (HIDDEN, 2)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, 0.5, (1, HIDDEN)), "b2": np.zeros(1)}


class ValueMLP:
    """2 -> 8 -> ReLU -> 1 -> sigmoid, built from scalar Value nodes."""

    def __init__(self, p):
        self.W1 = [[Value(float(w)) for w in row] for row in p["W1"]]
        self.b1 = [Value(float(b)) for b in p["b1"]]
        self.W2 = [Value(float(w)) for w in p["W2"][0]]
        self.b2 = Value(float(p["b2"][0]))

    def params(self):
        return [w for row in self.W1 for w in row] + self.b1 + self.W2 + [self.b2]

    def forward(self, x):
        out = self.b2
        for j in range(HIDDEN):
            h = (self.W1[j][0] * x[0] + self.W1[j][1] * x[1] + self.b1[j]).relu()
            out = out + self.W2[j] * h
        return out.sigmoid()


def train_value(Xtr, ytr, epochs=EPOCHS, lr=LR, seed=SEED):
    """Full-batch gradient descent on mean BCE. Xtr/ytr are plain Python lists of floats."""
    m = ValueMLP(init_params(seed))
    params = m.params()
    n = len(Xtr)
    # constant input/label nodes are created ONCE (they are leaves; only the weights are trained)
    data = [((Value(x[0]), Value(x[1])), Value(y), Value(1.0 - y)) for x, y in zip(Xtr, ytr)]
    losses = []
    for _ in range(epochs):
        for p in params:
            p.grad = 0.0
        total = None
        for x, y, ny in data:
            p = m.forward(x)
            l = -(y * p.log() + ny * (1.0 - p).log())                  # binary cross-entropy
            total = l if total is None else total + l
        loss = total * (1.0 / n)
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
        losses.append(loss.data)
    return m, losses


def accuracy_value(m, X, y):
    return sum((m.forward(x).data > 0.5) == (t > 0.5) for x, t in zip(X, y)) / len(y)
