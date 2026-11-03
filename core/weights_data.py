"""Вилучення ваг дуг НКК з таблиці спостережень (стаття EMP-PAVLOV, розділ 3, формули 10–15).

Реалізація повторює legacy/med_experiment.py::calculate_similarity_and_weights.
"""
from __future__ import annotations

import numpy as np


def distances_and_signs(A: np.ndarray):
    """Відстані d_ij(p), знаки sign_ij та подібність S_ij (формули 10–12).

    Знак обирається за меншою сумою відстаней: пряма |A_i - A_j| чи обернена |A_i - (1 - A_j)|.
    Повертає (dist[i, j, p], signs[i, j], S[i, j]).
    """
    N, m = A.shape
    dist = np.zeros((m, m, N))
    signs = np.zeros((m, m))
    S = np.zeros((m, m))
    for j in range(m):
        for i in range(m):
            if i == j:
                continue
            d_direct = np.abs(A[:, i] - A[:, j])
            d_inverse = np.abs(A[:, i] - (1 - A[:, j]))
            if np.sum(d_direct) <= np.sum(d_inverse):
                dist[i, j, :] = d_direct
                signs[i, j] = 1
            else:
                dist[i, j, :] = d_inverse
                signs[i, j] = -1
            S[i, j] = signs[i, j] * (1 - np.mean(dist[i, j, :]))
    return dist, signs, S


def extract_weights(
    A: np.ndarray,
    significance: float = 0.7,
    weight_threshold: float = 0.0,
) -> np.ndarray:
    """Матриця ваг W з нормалізованих даних A (N, n) (формули 13–15).

    significance     — поріг значущості |S_ij| > 0.7 (строго, як у legacy).
    weight_threshold — додатковий поріг відсікання |w_ij| < threshold -> 0 (legacy: 0.09).
                       Після відсікання ваги НЕ перенормовуються (як у legacy).
    """
    N, m = A.shape
    dist, signs, S = distances_and_signs(A)
    W = np.zeros((m, m))
    for j in range(m):
        infl = [i for i in range(m) if i != j and abs(S[i, j]) > significance]
        if not infl:
            continue
        d_min = np.min(dist[infl, j, :], axis=0)  # формула (13)
        contribs = {}
        total = 0.0
        for i in infl:
            contrib = 1 - np.mean(np.abs(d_min - dist[i, j, :]))
            val = max(0, contrib - significance)
            contribs[i] = val
            total += val
        for i in infl:
            if total > 0:
                w = signs[i, j] * (contribs[i] / total)  # формула (15)
                W[i, j] = w if abs(w) >= weight_threshold else 0.0
    return W


def apply_structure(W: np.ndarray, no_outgoing: list[int] = (), no_incoming: list[int] = ()) -> np.ndarray:
    """Структурні обмеження: обнуляє рядки (концепт ні на що не впливає) та стовпці (на концепт ніщо не впливає).

    У legacy для ІХС: no_outgoing=[C13] (діагноз), no_incoming=[C1] (вік).
    """
    W = W.copy()
    for i in no_outgoing:
        W[i, :] = 0.0
    for j in no_incoming:
        W[:, j] = 0.0
    return W
