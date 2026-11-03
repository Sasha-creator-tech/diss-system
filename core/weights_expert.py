"""Метод найменшого впливу (шкала Сааті) для проєктування НКК за експертними оцінками.

Повторює legacy/saaty_main.py:
  1. Дуги ділять на групи: додатні та від'ємні.
  2. У групі знаходять найменший за модулем вплив (еталон).
  3. Ранг Сааті кожної дуги r = round(|w| / |w_min|), обмежений [1, 9] (Python round — банківське округлення).
  4. w_min = 1 / Σr  (формула 3),  w = w_min · r  (формула 2),  потім ділення на max|w| у групі.
     Отже, у підсумку w = знак · r / max(r).
  5. Інтервали для налаштування: w ± 0.2, обмежені [0.02, 0.98] за модулем, округлені до 0.01.

На практиці експерт задає ранги напряму (`weights_from_ranks`); `ranks_from_matrix` — відтворення
прикладу статті, де ранги виводяться з відомої матриці.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SAATY_LINGUISTIC = {
    1: "Такий самий (Equal)",
    2: "Між 1 і 3",
    3: "Трохи сильніший (Moderate)",
    4: "Між 3 і 5",
    5: "Сильніший (Strong)",
    6: "Між 5 і 7",
    7: "Значно сильніший (Very Strong)",
    8: "Між 7 і 9",
    9: "Абсолютно сильніший (Extreme)",
}


@dataclass
class Link:
    src: int
    tgt: int
    sign: int          # +1 / -1
    rank: int          # 1..9 (шкала Сааті)

    @property
    def id(self) -> str:
        return f"w{self.src + 1},{self.tgt + 1}"


def saaty_rank(ratio: float) -> int:
    """Ранг Сааті з відношення сил впливу: int(round(ratio)) в межах [1, 9]."""
    return int(min(9, max(1, int(round(ratio)))))


def ranks_from_matrix(W: np.ndarray) -> list[Link]:
    """Ранги з відомої матриці (як legacy::extract_influences + розрахунок рангів). Порядок: рядок за рядком."""
    links = []
    for sign in (1, -1):
        group = [(i, j, W[i, j]) for i in range(W.shape[0]) for j in range(W.shape[1])
                 if (W[i, j] > 0 if sign > 0 else W[i, j] < 0)]
        if not group:
            continue
        w_min = min(abs(v) for _, _, v in group)
        links += [Link(i, j, sign, saaty_rank(abs(v) / w_min)) for i, j, v in group]
    return links


def weights_from_ranks(links: list[Link], n: int):
    """Ваги з рангів (формули 2–3 + нормування на максимум у групі).

    Повертає (W, table): table — DataFrame з проміжними величинами для звіту.
    """
    W = np.zeros((n, n))
    rows = []
    for sign in (1, -1):
        group = [l for l in links if l.sign == sign]
        if not group:
            continue
        total = sum(l.rank for l in group)
        w_min = 1.0 / total
        raw = [w_min * l.rank for l in group]
        w_max = max(raw)
        for l, r in zip(group, raw):
            w = sign * r / w_max
            W[l.src, l.tgt] = w
            rows.append({"Дуга": l.id, "Знак": "+" if sign > 0 else "−", "Ранг Сааті": l.rank,
                         "Оцінка": SAATY_LINGUISTIC[l.rank], "Σ рангів групи": total,
                         "w_min = 1/Σ": round(w_min, 4), "w = w_min·r": round(sign * r, 4),
                         "w (норм.)": round(w, 4)})
    return W, pd.DataFrame(rows)


def weight_intervals(W: np.ndarray, variation: float = 0.2, abs_bounds=(0.02, 0.98),
                     interval_decimals: int = 2, init_decimals: int | None = 4) -> list[dict]:
    """Структура для налаштування: [{'src','tgt','init','min','max'}, ...] (рядок за рядком).

    init_decimals=4 — як legacy (структура копіювалась у saaty_tuning.py з друку '{w:.4f}').
    """
    lo_abs, hi_abs = abs_bounds
    out = []
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            if w == 0:
                continue
            if w > 0:
                lo, hi = max(lo_abs, w - variation), min(hi_abs, w + variation)
            else:
                lo, hi = max(-hi_abs, w - variation), min(-lo_abs, w + variation)
            init = round(float(w), init_decimals) if init_decimals is not None else float(w)
            out.append({"src": i, "tgt": j, "init": init,
                        "min": round(lo, interval_decimals), "max": round(hi, interval_decimals)})
    return out
