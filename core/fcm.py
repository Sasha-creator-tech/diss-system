"""Моделювання НКК: рекурентне співвідношення (формула 4) і межі виходу (формула 7).

A^{k+1} = A^k + (A^k - A^{k-1}) · W,   A^{-1} = 0  =>  A^1 = A^0 + A^0 · W.

Реалізація повторює legacy/med_experiment.py::simulate_fcm_raw (фіксована кількість ітерацій).
"""
from __future__ import annotations

import itertools

import numpy as np


def simulate(A0: np.ndarray, W: np.ndarray, iterations: int = 30, exact: bool = True) -> np.ndarray:
    """Стан НКК після `iterations` кроків для кожного початкового вектора.

    A0    — (N, n) або (n,) початкові вектори.
    exact — True: побітово ідентично legacy (np.dot по рядках);
            False: швидке матричне множення (розбіжність ~1e-11).
    """
    A0 = np.asarray(A0, dtype=float)
    single = A0.ndim == 1
    A = np.atleast_2d(A0).copy()
    A_prev = np.zeros_like(A)
    for _ in range(iterations):
        delta = A - A_prev
        if exact:
            step = np.stack([np.dot(row, W) for row in delta])
        else:
            step = delta @ W
        A_prev, A = A, A + step
    return A[0] if single else A


def output_bounds_by_sign(W: np.ndarray, out_idx: int = -1, iterations: int = 30, exact: bool = True):
    """Межі (A_min, A_max) вихідного концепту за знаками ПРЯМИХ дуг до виходу.

    A_max: входи з w_{i,out} > 0 = 1; A_min: входи з w_{i,out} < 0 = 1; решта 0.
    Як legacy::calculate_bounds (використовується в прогнозі та налаштуванні).
    """
    n = W.shape[0]
    out = out_idx % n
    a_min = np.zeros(n)
    a_max = np.zeros(n)
    for i in range(n):
        if i == out:
            continue
        if W[i, out] > 0:
            a_max[i] = 1.0
        elif W[i, out] < 0:
            a_min[i] = 1.0
    lo = simulate(a_min[None, :], W, iterations, exact)[0, out]
    hi = simulate(a_max[None, :], W, iterations, exact)[0, out]
    return lo, hi


def output_bounds_bruteforce(W: np.ndarray, out_idx: int = -1, iterations: int = 30, exact: bool = True):
    """Межі (A_min, A_max) виходу перебором усіх бінарних комбінацій входів (2^(n-1)).

    Як legacy, розділ 9 (ранги важливості). Повертає (lo, hi, combo_lo, combo_hi).
    """
    n = W.shape[0]
    out = out_idx % n
    inputs = [i for i in range(n) if i != out]
    combos = np.array(list(itertools.product([0, 1], repeat=len(inputs))), dtype=float)
    A0 = np.zeros((len(combos), n))
    A0[:, inputs] = combos
    vals = simulate(A0, W, iterations, exact)[:, out]
    k_lo, k_hi = int(np.argmin(vals)), int(np.argmax(vals))
    return vals[k_lo], vals[k_hi], tuple(combos[k_lo].astype(int)), tuple(combos[k_hi].astype(int))


def run_until_stable(
    x0: np.ndarray,
    W: np.ndarray,
    max_iter: int = 50,
    eps: float | None = 1e-3,
    blowup: float | None = None,
    record: bool = False,
):
    """Один вектор: ітерації формули (4) до стаціонарності (шкала [-1, 1], статті 1, 2, 4).

    Цикл як у legacy/transport_*.py та legacy/saaty_tuning.py:
        x_next = x + (x - x_prev)·W
        eps    — зупинка, коли max|x_next - x| < eps (None = рівно max_iter кроків);
        blowup — аварійна зупинка, коли max|x_next| > blowup (saaty_tuning: 1e5).
    Повертає (final, history), history = [x0, x1, ...] якщо record=True, інакше None.
    """
    x = np.asarray(x0, dtype=float).copy()
    x_prev = np.zeros_like(x)
    hist = [x.copy()] if record else None
    for _ in range(max_iter):
        x_next = x + np.dot(x - x_prev, W)
        if record:
            hist.append(x_next.copy())
        if blowup is not None and np.max(np.abs(x_next)) > blowup:
            x = x_next
            break
        if eps is not None and np.max(np.abs(x_next - x)) < eps:
            x = x_next
            break
        x_prev, x = x, x_next
    return x, (np.array(hist) if record else None)


def normalize_maxabs(x: np.ndarray) -> np.ndarray:
    """Нормування стаціонарного вектора на max|x| по всіх концептах (як у legacy, шкала [-1, 1])."""
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def predict_output(
    A_input: np.ndarray, W: np.ndarray, out_idx: int = -1, iterations: int = 30, exact: bool = True
) -> np.ndarray:
    """Нормований прогноз виходу в [0, 1] (формула 7, межі — за знаками прямих дуг, з обрізанням)."""
    lo, hi = output_bounds_by_sign(W, out_idx, iterations, exact)
    raw = simulate(A_input, W, iterations, exact)[:, out_idx]
    if hi - lo == 0:
        return np.clip(raw, 0, 1)
    return np.clip((raw - lo) / (hi - lo), 0, 1)
