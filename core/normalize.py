"""Нормалізація даних: нечітка перфектність (стаття EMP-PAVLOV, формула 8).

A = 1 / (1 + ((u - b) / c)^2),  u = (x - x_min) / (x_max - x_min)

Правила (як у legacy/med_experiment.py):
    'a' — чим більше, тим краще  (b = 1)
    'b' — чим менше, тим краще   (b = 0)
    'c' — золота середина        (b = 0.5)
"""
from __future__ import annotations

import numpy as np

DEFAULT_C = 0.389

RULE_TO_B = {"a": 1.0, "b": 0.0, "c": 0.5}


def linear_unit(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """u = (x - x_min)/(x_max - x_min); якщо x_max == x_min — повертає одиниці (як у legacy)."""
    x = np.asarray(x, dtype=float)
    if x_max - x_min == 0:
        return np.ones_like(x)
    return (x - x_min) / (x_max - x_min)


def perfection(u: np.ndarray, b: float, c: float = DEFAULT_C) -> np.ndarray:
    """Функція нечіткої перфектності, формула (8)."""
    return 1.0 / (1.0 + ((np.asarray(u, dtype=float) - b) / c) ** 2)


def rule_to_b(rule: str | float) -> float:
    """Перетворює правило 'a'/'b'/'c' (або число) на параметр b."""
    if isinstance(rule, str):
        return RULE_TO_B[rule]
    return float(rule)


def normalize_table(
    data: np.ndarray,
    rules: list,
    c: float = DEFAULT_C,
    output_cols: tuple[int, ...] = (),
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Нормалізує таблицю спостережень (табл. 1 -> табл. 2 статті).

    data        — (N, n) сирі значення.
    rules       — правило для кожного вхідного стовпця ('a'/'b'/'c' або число b).
                  Для стовпців з output_cols правило ігнорується.
    output_cols — стовпці-виходи (діагноз): лінійне min-max, без перфектності (як у legacy).
    bounds      — (x_min, x_max) по стовпцях; за замовчуванням — min/max з даних (як у legacy).
    """
    data = np.asarray(data, dtype=float)
    x_min, x_max = bounds if bounds is not None else (data.min(axis=0), data.max(axis=0))
    out = np.zeros_like(data)
    for j in range(data.shape[1]):
        u = linear_unit(data[:, j], x_min[j], x_max[j])
        if j in output_cols:
            out[:, j] = u
        else:
            out[:, j] = perfection(u, rule_to_b(rules[j]), c)
    return out
