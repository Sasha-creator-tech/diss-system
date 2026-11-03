"""Шкала значень концептів: [0, 1] ('unit') або [-1, 1] ('sym').

Кожен метод рахує у своїй «рідній» шкалі, як у статтях і legacy-коді:
  • модель за даними (перфектність, формули 8–15) — [0, 1];
  • експертна НКК (статті 1, 2, 4)                 — [-1, 1].
Шкала проєкту визначає, у якій шкалі користувач ВВОДИТЬ і БАЧИТЬ значення концептів.
Перетворення лінійне (мінімум → мінімум, максимум → максимум), тож результати моделі від вибору
шкали не змінюються:   x_unit = (x_sym + 1) / 2,   x_sym = 2·x_unit − 1.
Ваги дуг завжди в [-1, 1] (знак = напрям впливу) і не перетворюються.
"""
from __future__ import annotations

import numpy as np

UNIT, SYM = "unit", "sym"
RANGES = {UNIT: (0.0, 1.0), SYM: (-1.0, 1.0)}
LABELS = {UNIT: "[0, 1]", SYM: "[−1, 1]"}


def convert(x, src: str, dst: str):
    """Перетворення значень концептів між шкалами."""
    if src == dst:
        return x
    x = np.asarray(x, dtype=float) if not np.isscalar(x) else float(x)
    if src == UNIT and dst == SYM:
        return 2.0 * x - 1.0
    if src == SYM and dst == UNIT:
        return (x + 1.0) / 2.0
    raise ValueError(f"Невідома шкала: {src!r} → {dst!r}")


def lo(scale: str) -> float:
    return RANGES[scale][0]


def hi(scale: str) -> float:
    return RANGES[scale][1]
