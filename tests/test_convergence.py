"""Перевірка збіжності: правила (1)–(6) і функції табл. 1 огляду Nápoles et al. (2026)."""
import numpy as np
import pytest

from core import convergence as cv
from core import experiment, fcm, gds, weights_expert
from core.config import load_config
from core.tuning import mimo_result

P = cv.DEFAULT_PARAMS


def test_activations_table1():
    x = np.array([[-2.0, 0.0, 0.3, 0.7]])
    W = np.zeros((4, 4))
    assert cv.activate("bivalent", x, W, P).tolist() == [[0, 0, 1, 1]]
    assert cv.activate("step", x, W, P).tolist() == [[0, 0, 0, 1]]          # v = 0.5
    assert cv.activate("trivalent", x, W, P).tolist() == [[-1, 0, 1, 1]]
    assert cv.activate("sigmoid", x, W, P)[0, 1] == pytest.approx(0.5)
    assert cv.activate("tanh", x, W, P)[0, 3] == pytest.approx(np.tanh(0.7))
    assert np.linalg.norm(cv.activate("rescaled", x, W, P)) == pytest.approx(1.0)
    W2 = np.array([[0, 0.5], [-0.25, 0]])                                  # Σ_j|w_j0| = 0.25, Σ_j|w_j1| = 0.5
    assert cv.activate("expnorm", np.array([[-0.2, 0.25]]), W2, P)[0].tolist() == pytest.approx([-0.8, 0.5])


def test_rule_equivalences():
    rng = np.random.default_rng(0)
    W = rng.uniform(-1, 1, (6, 6)) * (rng.random((6, 6)) < 0.5)
    np.fill_diagonal(W, 0)
    A0 = rng.random((5, 6))
    p = {**P, "k1": 1.0, "k2": 0.5, "wii": 0.5}
    # (5) — частковий випадок (3) з k1 = 1, k2 = w_ii
    a = cv.simulate(W, A0, cv.Variant("selfloop", "sigmoid"), p, 20)
    b = cv.simulate(W, A0, cv.Variant("extended", "sigmoid"), p, 20)
    assert np.allclose(a, b)
    # (6) з φ = 1 — класичне правило (1)
    a = cv.simulate(W, A0, cv.Variant("quasi", "tanh"), {**P, "phi": 1.0}, 20)
    b = cv.simulate(W, A0, cv.Variant("kosko", "tanh"), P, 20)
    assert np.allclose(a, b)
    # базова формула без функції = fcm.simulate (рекурентна формула статей)
    a = cv.simulate(W, A0, cv.Variant("native", "none"), P, 30)[-1]
    assert np.allclose(a, fcm.simulate(A0, W, 30, exact=False))


def test_dynamics_states():
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    d = cv.classify(cv.simulate(swap, [[1.0, 0.0]], cv.Variant("kosko", "none"), P, 40))
    assert d.status[0] == cv.CYCLE and d.period[0] == 2
    d = cv.classify(cv.simulate(2 * swap, [[1.0, 0.5]], cv.Variant("native", "none"), P, 60))
    assert d.status[0] == cv.DIVERGED
    small = 0.2 * swap
    A0 = np.random.default_rng(1).random((20, 2))
    d = cv.classify(cv.simulate(small, A0, cv.Variant("kosko", "sigmoid"), P, 50))
    assert (d.status == cv.FIXED).all() and d.unique_fixed_point
    d = cv.classify(cv.simulate(small, A0, cv.Variant("quasi", "sigmoid"), P, 80))
    assert (d.status == cv.FIXED).all() and not d.unique_fixed_point          # теорема 10


def test_conditions():
    W = 0.2 * np.array([[0.0, 1.0], [1.0, 0.0]])
    c = {x.name: x for x in cv.conditions(W, cv.Variant("kosko", "sigmoid"), P)}
    assert c["теорема 3: ‖W‖_F < 4/λ"].holds and c["стиск q = L·‖M‖₂ < 1"].holds
    assert cv.conditions(W, cv.Variant("native", "none"), P)[0].holds              # ρ = 0.2
    assert not cv.conditions(5 * W, cv.Variant("native", "none"), P)[0].holds      # ρ = 1


def test_baseline_reproduces_ihd_initial_model():
    cfg = load_config("examples/ihd/config.yaml")
    cfg.exact = False
    raw, A_norm, A_input, real_class, *_, W0 = experiment.initial_model(cfg)
    ref = experiment.evaluate(cfg, W0, cfg.initial_thresholds, A_input, A_norm[:, -1], real_class)
    sc = cv.empirical_scorer(W0, A_norm[:, -1], real_class, cfg.initial_thresholds)
    rep = cv.compare(W0, A_input, [cv.Variant("native", "none")], T=cfg.iterations, scorer=sc)[0]
    assert rep.quality["з"]["збіги"] == ref.matches == 39
    assert np.allclose(rep.quality["norm"], ref.pred)


def test_baseline_reproduces_transport_tuning_mse():
    cfg = gds.load_gds_config("examples/transport/config.yaml")
    df = gds.load_training(cfg)
    ids = cfg.concept_ids
    X = df[[ids[i] for i in cfg.cause_idx]].to_numpy(float)
    Y = df[[ids[i] for i in cfg.effect_idx]].to_numpy(float)
    W_s, _ = weights_expert.weights_from_ranks(weights_expert.ranks_from_matrix(cfg.W), cfg.n)
    st = weights_expert.weight_intervals(W_s, 0.2)
    ref = mimo_result(st, [c["init"] for c in st], [], X, Y, cfg.cause_idx, cfg.effect_idx, cfg.n)
    A0 = np.zeros((len(X), cfg.n))
    A0[:, cfg.cause_idx] = X
    reps = cv.compare(ref.W_before, A0, cv.all_variants(), T=50, scorer=cv.gds_scorer(Y, cfg.effect_idx))
    assert reps[0].variant.is_baseline
    assert reps[0].quality["з"]["MSE"] == pytest.approx(ref.mse_before, abs=1e-12)
    assert len(reps) == 5 * 8                                    # усі правила × усі функції
