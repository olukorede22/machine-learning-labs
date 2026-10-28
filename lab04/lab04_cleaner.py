"""Fitted cleaning transformer for Lab 4 (kept in its own module so joblib can reload it)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING = "?"


def is_missing(s, token=MISSING):
    return s.isna() | (s.astype(str).str.strip() == token)


class TrainFittedCleaner(BaseEstimator, TransformerMixin):
    """'?' -> NaN, categorical mode / numeric median imputation, then clip numeric
    columns to the 1st/99th percentile. All statistics come from fit(X_train)."""

    def __init__(self, numeric_cols, categorical_cols, lower_q=0.01, upper_q=0.99,
                 missing_token=MISSING):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.missing_token = missing_token

    def _prep(self, X):
        X = X.copy()
        for c in self.categorical_cols:
            s = X[c].astype(object)
            X[c] = s.where(~is_missing(s, self.missing_token), np.nan)
        for c in self.numeric_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def fit(self, X, y=None):
        Xp = self._prep(X)
        self.fill_values_ = {c: Xp[c].dropna().mode().sort_values().iloc[0]
                             for c in self.categorical_cols}
        self.fill_values_.update({c: float(Xp[c].median()) for c in self.numeric_cols})
        imp = Xp.fillna(self.fill_values_)
        self.clip_bounds_ = {c: (float(imp[c].quantile(self.lower_q)),
                                 float(imp[c].quantile(self.upper_q)))
                             for c in self.numeric_cols}
        return self

    def transform_with_report(self, X):
        Xp = self._prep(X)
        report = {"imputed": {c: int(Xp[c].isna().sum()) for c in Xp.columns
                              if Xp[c].isna().any()}, "clipped": {}}
        Xp = Xp.fillna(self.fill_values_)
        for c, (lo, hi) in self.clip_bounds_.items():
            n = int(((Xp[c] < lo) | (Xp[c] > hi)).sum())
            if n:
                report["clipped"][c] = n
            Xp[c] = Xp[c].clip(lo, hi)
        return Xp, report

    def transform(self, X):
        return self.transform_with_report(X)[0]
