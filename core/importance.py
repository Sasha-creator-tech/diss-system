"""Ранги (індекси важливості) вхідних концептів I(C_i) — як legacy/med_experiment.py, розділ 9.

1. A_min, A_max виходу — перебором усіх бінарних комбінацій входів.
2. Для кожного входу C_i: початковий вектор з одиницею лише в C_i,
   I(C_i) = (A_out - A_min) / (A_max - A_min).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fcm


@dataclass
class ImportanceResult:
    values: np.ndarray          # I(C_i) для входів (у порядку inputs)
    inputs: list[int]           # індекси вхідних концептів
    a_min: float
    a_max: float
    combo_min: tuple
    combo_max: tuple


def concept_importance(W: np.ndarray, out_idx: int = -1, iterations: int = 30, exact: bool = True) -> ImportanceResult:
    n = W.shape[0]
    out = out_idx % n
    lo, hi, c_lo, c_hi = fcm.output_bounds_bruteforce(W, out, iterations, exact)
    inputs = [i for i in range(n) if i != out]
    A0 = np.zeros((len(inputs), n))
    for k, i in enumerate(inputs):
        A0[k, i] = 1.0
    raw = fcm.simulate(A0, W, iterations, exact)[:, out]
    vals = np.zeros(len(inputs)) if hi - lo == 0 else (raw - lo) / (hi - lo)
    return ImportanceResult(vals, inputs, lo, hi, c_lo, c_hi)
