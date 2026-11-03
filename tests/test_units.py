"""Модульні тести ядра (швидкі)."""
import numpy as np

from core import fcm, metrics, normalize, weights_data
from core.experiment import prepare, split


def legacy_simulate(A_initial, W, iterations=30):
    """Копія legacy::simulate_fcm_raw для порівняння."""
    N, n = A_initial.shape
    out = np.zeros((N, n))
    for p in range(N):
        a_prev = np.zeros(n)
        a = A_initial[p, :].copy()
        for _ in range(iterations):
            nxt = a + np.dot(a - a_prev, W)
            a_prev = a.copy()
            a = nxt.copy()
        out[p, :] = a
    return out


def test_perfection_rules():
    u = np.array([0.0, 0.5, 1.0])
    assert np.allclose(normalize.perfection(u, 1.0)[2], 1.0)   # 'a': максимум у u=1
    assert np.allclose(normalize.perfection(u, 0.0)[0], 1.0)   # 'b': максимум у u=0
    assert normalize.rule_to_b("c") == 0.5


def test_classify_boundaries():
    t = [0.2, 0.35, 0.5, 0.65, 0.8]
    vals = [0.0, 0.2, 0.349, 0.35, 0.8, 1.0]
    assert metrics.classify(vals, t).tolist() == [1, 2, 2, 3, 6, 6]
    assert metrics.classify([0.3], [0.35, 0.2, 0.5, 0.65, 0.8]).tolist() == [2]  # пороги сортуються


def test_simulate_exact_matches_legacy():
    rng = np.random.default_rng(0)
    W = rng.uniform(-0.4, 0.4, (13, 13))
    A = rng.uniform(0, 1, (20, 13))
    assert np.array_equal(fcm.simulate(A, W, exact=True), legacy_simulate(A, W))
    assert np.allclose(fcm.simulate(A, W, exact=False), legacy_simulate(A, W), rtol=1e-9)


def test_normalization_matches_golden(ihd_cfg, golden):
    _, A_norm, _, _ = prepare(ihd_cfg)
    for i in range(12):
        got = [f"{v:.3f}" for v in A_norm[:, i]]
        assert got == golden["t4"][f"A{i + 1}"].tolist(), f"A{i + 1}"


def test_initial_weights_match_golden(ihd_cfg, golden):
    _, A_norm, _, _ = prepare(ihd_cfg)
    train_idx, _ = split(len(A_norm), ihd_cfg)
    W = weights_data.extract_weights(A_norm[train_idx], 0.7, 0.09)
    W = weights_data.apply_structure(W, no_outgoing=[12], no_incoming=[0])
    for _, row in golden["t5"].iterrows():
        i, j = (int(x) - 1 for x in row["Зв'язок"][2:].split(","))
        assert round(W[i, j], 3) == row["До налаштування"], row["Зв'язок"]
    assert np.count_nonzero(W) == len(golden["t5"])


def test_model_io_roundtrip(ihd_cfg):
    import copy
    from core import model_io
    from core.experiment import run_experiment
    cfg = copy.deepcopy(ihd_cfg)
    cfg.exact = False
    cfg.ga.num_generations = 5
    r = run_experiment(cfg, with_importance=False)
    m = model_io.from_json(model_io.to_json(model_io.EmpiricalModel.from_result(r)))
    pred, cls = m.predict(r.raw[:, :-1])
    assert np.allclose(pred, r.after.pred) and np.array_equal(cls, r.after.classes)
