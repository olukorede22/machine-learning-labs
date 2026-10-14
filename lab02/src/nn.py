"""Two-hidden-layer MLP in NumPy: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax.

Shapes: W1 (64,W), b1 (W,), W2 (W,W), b2 (W,), W3 (W,10), b3 (10,).
Activations are row-major: X is (batch, features).
"""
import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def get_data():
    """Task 1: scale by 1/16, one-hot labels, 70/30 stratified split."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    return Xtr, Xte, ytr, yte, one_hot(ytr), one_hot(yte)


def one_hot(y, k=10):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class MLP:
    def __init__(self, width, init_std=0.01, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, init_std, (64, width)); self.b1 = np.zeros(width)
        self.W2 = rng.normal(0, init_std, (width, width)); self.b2 = np.zeros(width)
        self.W3 = rng.normal(0, init_std, (width, 10)); self.b3 = np.zeros(10)

    PARAMS = ("W1", "b1", "W2", "b2", "W3", "b3")

    # ---- Task 3: forward ----
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1;  self.A1 = np.maximum(0, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2;  self.A2 = np.maximum(0, self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.P = softmax(self.Z3)
        return self.P

    # ---- Task 5: backward (mean CE over batch) ----
    def backward(self, Y):
        B = Y.shape[0]
        dZ3 = (self.P - Y) / B                       # softmax + CE
        g = {"W3": self.A2.T @ dZ3, "b3": dZ3.sum(0)}
        dZ2 = (dZ3 @ self.W3.T) * (self.Z2 > 0)      # ReLU mask
        g["W2"], g["b2"] = self.A1.T @ dZ2, dZ2.sum(0)
        dZ1 = (dZ2 @ self.W2.T) * (self.Z1 > 0)
        g["W1"], g["b1"] = self.X.T @ dZ1, dZ1.sum(0)
        return g

    def step(self, g, lr):
        for n in self.PARAMS:
            setattr(self, n, getattr(self, n) - lr * g[n])

    def predict(self, X):
        return self.forward(X).argmax(1)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # numerical stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, Y, eps=1e-12):
    """Task 4: mean categorical cross-entropy."""
    return float(-np.mean(np.sum(Y * np.log(P + eps), axis=1)))
