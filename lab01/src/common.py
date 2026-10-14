"""Shared helpers for Lab 1: seeding and the reproducible data split."""
import random
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for Lab 1
        pass


def get_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def make_models():
    return {
        "logreg": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
