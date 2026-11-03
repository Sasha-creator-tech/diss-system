"""Автопідприємство (УДС, шкала [-1, 1]): відтворення legacy/transport_*.py та legacy/saaty_*.py.

Допуск 1e-12: на різних процесорах (x86 / ARM Mac) останній біт float може відрізнятися.

Еталони для ГА-частин отримано з копій legacy-скриптів із доданим seed
(tests/golden/transport/legacy_*_seeded.py), бо оригінали seed не задавали.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import gds, weights_expert

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "tests" / "golden" / "transport"
CFG = ROOT / "examples" / "transport" / "config.yaml"
TOL = 1e-12


@pytest.fixture(scope="module")
def cfg():
    return gds.load_gds_config(CFG)


def test_forecast_history_exact(cfg):
    f = gds.run_forecast(cfg)
    assert np.allclose(f.history, np.load(G / "forecast_hist.npy"), rtol=0, atol=TOL)
    # нормований стаціонарний стан наслідків ≈ статті 4: [0.70, 0.70, 0.27, 1.00, 0.89]
    assert np.round(f.normalized[cfg.effect_idx], 2).tolist() == [0.70, 0.70, 0.27, 1.00, 0.89]


def test_saaty_structure_exact(cfg):
    rep = gds.run_expert(cfg, tune=False)
    assert rep.structure == json.loads((G / "saaty_structure.json").read_text())
    ranks = dict(zip(rep.saaty_table["Дуга"], rep.saaty_table["Ранг Сааті"]))
    assert ranks["w6,9"] == 9 and ranks["w3,6"] == 8 and ranks["w4,8"] == 4


def test_saaty_ranks_input_equals_matrix_route(cfg):
    links = weights_expert.ranks_from_matrix(cfg.W)
    W1, _ = weights_expert.weights_from_ranks(links, cfg.n)
    assert np.allclose(np.sign(W1), np.sign(cfg.W))
    assert np.isclose(np.abs(W1).max(), 1.0)


def test_diagnosis_matches_seeded_legacy(cfg):
    rep = gds.run_diagnosis(cfg)
    ref = pd.read_csv(G / "diag_seeded.csv")
    cols = [c for c in ref.columns if c != "Scenario"]
    assert np.allclose(rep.table[cols].to_numpy(float), ref[cols].to_numpy(float), rtol=0, atol=TOL)
    s1 = json.loads((G / "diag_seeded_s1.json").read_text())
    r = rep.results["S1"]
    assert np.allclose(r.cause_history, np.array(s1["hist"]), rtol=0, atol=TOL)
    assert np.allclose(r.error_history, s1["error_curve"], rtol=0, atol=TOL)
    # таблиця наслідків детермінована і збігається зі статтею 1 (табл. 2)
    s1_effects = rep.table.loc[0, ["hat_x6", "hat_x7", "hat_x8", "hat_x9", "hat_x10"]].to_numpy(float)
    assert np.allclose(s1_effects, [-0.206, -0.378, 0.341, -0.492, -0.438], atol=6e-4)


def test_diagnosis_without_reference_runs(cfg):
    c = copy.deepcopy(cfg)
    c.diagnosis = {**c.diagnosis, "n_runs": 5, "select_by_reference": False}
    rep = gds.run_diagnosis(c, scenarios={"S3": c.diagnosis["scenarios"]["S3"]})
    assert rep.results["S3"].error < 0.05


@pytest.mark.slow
def test_saaty_tuning_matches_seeded_legacy(cfg):
    rep = gds.run_expert(cfg, tune=True)
    ref = json.loads((G / "tune_seeded.json").read_text())
    assert np.allclose(rep.tuning.genes_after, ref["solution"], rtol=0, atol=TOL)
    assert np.allclose(rep.tuning.history_mse, ref["hist"], rtol=0, atol=TOL)
    assert np.isclose(rep.tuning.mse_after, ref["mse_after"], rtol=0, atol=TOL)
    assert np.isclose(rep.tuning.mad_after, ref["mad_after"], rtol=0, atol=TOL)
    assert round(rep.tuning.mse_before, 5) == 0.11843
