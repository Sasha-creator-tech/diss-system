"""Наскрізне відтворення legacy/med_experiment.py (стаття EMP-PAVLOV, табл. 5–7).

Швидкий тест (exact=False, ~10 с) і повний побітово-точний (pytest -m slow, ~1 хв).
"""
import copy

import numpy as np
import pytest

from core.experiment import run_experiment
from core.reports import table5_weights, table6_comparison

# Табл. 7 статті: рядки — реальні d1..d6, стовпці — обчислені
CONFUSION_BEFORE = [[11, 0, 0, 0, 0, 0], [4, 7, 0, 0, 0, 0], [0, 4, 5, 2, 0, 0],
                    [0, 0, 3, 8, 0, 0], [0, 0, 0, 5, 6, 0], [0, 0, 0, 0, 8, 2]]
CONFUSION_AFTER = [[11, 0, 0, 0, 0, 0], [2, 9, 0, 0, 0, 0], [0, 0, 10, 1, 0, 0],
                   [0, 0, 0, 11, 0, 0], [0, 0, 0, 0, 11, 0], [0, 0, 0, 0, 0, 10]]
THRESHOLDS_AFTER = [0.1293, 0.2853, 0.4537, 0.5784, 0.6874]


def _check(r, golden):
    assert np.round(r.tuning.thresholds, 4).tolist() == THRESHOLDS_AFTER
    assert r.before.confusion.tolist() == CONFUSION_BEFORE
    assert r.after.confusion.tolist() == CONFUSION_AFTER
    assert (r.before.matches, r.after.matches) == (39, 62)
    t5 = table5_weights(r)
    assert t5["До налаштування"].tolist() == golden["t5"]["До налаштування"].tolist()
    assert t5["Після налаштування"].tolist() == golden["t5"]["Після налаштування"].tolist()
    assert table6_comparison(r).astype(str).values.tolist() == golden["t6"].values.tolist()


def test_ihd_fast(ihd_cfg, golden):
    cfg = copy.deepcopy(ihd_cfg)
    cfg.exact = False
    _check(run_experiment(cfg, with_importance=False), golden)


@pytest.mark.slow
def test_ihd_exact_with_importance(ihd_cfg, golden):
    r = run_experiment(copy.deepcopy(ihd_cfg), with_importance=True)
    _check(r, golden)
    imp = r.importance
    assert round(imp.a_min, 3) == -155.963 and round(imp.a_max, 3) == 38.284
    assert np.round(imp.values, 3).tolist() == [0.801, 0.732, 0.701, 1.0, 0.656, 0.774,
                                                0.803, 0.772, 0.721, 0.591, 0.770, 0.709]
