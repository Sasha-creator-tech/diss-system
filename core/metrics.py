"""Класифікація за порогами та метрики якості."""
from __future__ import annotations

import numpy as np


def classify(values: np.ndarray, thresholds) -> np.ndarray:
    """Клас 1..m за порогами t1<...<t_{m-1}: v < t1 -> 1, t1 <= v < t2 -> 2, ..., v >= t_{m-1} -> m.

    Пороги сортуються (як у legacy).
    """
    t = np.sort(np.asarray(thresholds, dtype=float))
    return np.searchsorted(t, np.asarray(values, dtype=float), side="right") + 1


def ssd(pred: np.ndarray, real: np.ndarray) -> float:
    """Сума квадратів відхилень ||A_n - Â_n||^2."""
    return float(np.sum((np.asarray(pred) - np.asarray(real)) ** 2))


def mse(pred, real) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(real)) ** 2))


def mad(pred, real) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(real))))


def accuracy(pred_cls, real_cls) -> float:
    return float(np.mean(np.asarray(pred_cls) == np.asarray(real_cls)))


def confusion_matrix(real_cls, pred_cls, n_classes: int) -> np.ndarray:
    """Матриця [реальний, обчислений] для класів 1..n_classes (табл. 7)."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for r, p in zip(real_cls, pred_cls):
        M[int(r) - 1, int(p) - 1] += 1
    return M
