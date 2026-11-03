"""Повний цикл емпіричного моделювання (стаття EMP-PAVLOV) — як legacy/med_experiment.py.

Дані -> нормалізація -> ваги з даних -> налаштування ГА [W | T] -> оцінка -> ранги.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import fcm, metrics, normalize, weights_data
from .config import ExperimentConfig
from .importance import ImportanceResult, concept_importance
from .tuning import TuningResult, build_chromosome, initial_population, tune


@dataclass
class Evaluation:
    pred: np.ndarray                  # нормований прогноз виходу [0, 1]
    classes: np.ndarray | None        # обчислені класи (None для числового виходу)
    mad: float
    mse: float
    matches: int | None
    confusion: np.ndarray | None
    r2: float | None = None           # коефіцієнт детермінації (числовий вихід)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    raw: np.ndarray
    A_norm: np.ndarray
    real_class: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    W_initial: np.ndarray
    tuning: TuningResult
    before: Evaluation    # уся вибірка, початкові W і пороги
    after: Evaluation     # уся вибірка, налаштовані W і пороги
    test_before: Evaluation
    test_after: Evaluation
    importance: ImportanceResult | None


def evaluate(cfg: ExperimentConfig, W, thresholds, A_input, real_output, real_class) -> Evaluation:
    pred = fcm.predict_output(A_input, W, -1, cfg.iterations, cfg.exact)
    if cfg.is_numeric or not thresholds:
        ss_tot = float(np.sum((real_output - np.mean(real_output)) ** 2))
        r2 = 1.0 - float(np.sum((pred - real_output) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        return Evaluation(pred, None, metrics.mad(pred, real_output), metrics.mse(pred, real_output), None, None, r2)
    cls = metrics.classify(pred, thresholds)
    return Evaluation(
        pred, cls, metrics.mad(pred, real_output), metrics.mse(pred, real_output),
        int(np.sum(cls == real_class)), metrics.confusion_matrix(real_class, cls, cfg.n_classes),
    )


def prepare(cfg: ExperimentConfig):
    """Нормалізація та вхідна матриця (вихід = 0 на старті), як legacy розділ 2."""
    raw = cfg.matrix()
    n = raw.shape[1]
    rules = [c.rule for c in cfg.concepts] + [None]
    A_norm = normalize.normalize_table(raw, rules, cfg.perfection_c, output_cols=(n - 1,))
    A_input = A_norm.copy()
    A_input[:, -1] = 0.0
    real_class = np.zeros(len(raw), dtype=int) if cfg.is_numeric else raw[:, -1].astype(int)
    return raw, A_norm, A_input, real_class


def split(N: int, cfg: ExperimentConfig):
    """np.random.seed(seed) + permutation — ВАЖЛИВО: споживає глобальний RNG так само, як legacy."""
    np.random.seed(cfg.seed)
    order = np.random.permutation(N)
    if cfg.train_fraction >= 1.0:
        return order[:], order[:]                      # legacy: train = test = усі
    k = int(round(N * cfg.train_fraction))
    return order[:k], order[k:]


def initial_model(cfg: ExperimentConfig):
    """Нормалізація, поділ вибірки та ваги з даних (формули 8, 10–15) — етап «проєктування» без ГА.

    Повертає (raw, A_norm, A_input, real_class, train_idx, test_idx, W0).
    Увага: split() перевстановлює глобальний seed — порядок викликів як у legacy.
    """
    raw, A_norm, A_input, real_class = prepare(cfg)
    n = raw.shape[1]
    train_idx, test_idx = split(len(raw), cfg)
    W0 = weights_data.extract_weights(A_norm[train_idx], cfg.significance, cfg.weight_threshold)
    ids = cfg.concept_ids
    W0 = weights_data.apply_structure(
        W0,
        no_outgoing=[n - 1] if cfg.no_outgoing_output else [],
        no_incoming=[ids.index(c) for c in cfg.no_incoming],
    )
    return raw, A_norm, A_input, real_class, train_idx, test_idx, W0


def run_experiment(
    cfg: ExperimentConfig,
    with_importance: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> ExperimentResult:
    raw, A_norm, A_input, real_class, train_idx, test_idx, W0 = initial_model(cfg)
    real_output = A_norm[:, -1]

    spec = build_chromosome(W0, cfg.initial_thresholds, cfg.weight_delta, tuple(cfg.weight_abs_bounds),
                            tuple(cfg.threshold_bounds), cfg.tune_thresholds)
    pop = initial_population(spec, cfg.ga.sol_per_pop, tuple(cfg.threshold_bounds))
    res = tune(W0, A_input[train_idx], real_output[train_idx], real_class[train_idx], spec, cfg.ga,
               fixed_thresholds=cfg.initial_thresholds, fitness=cfg.fitness, iterations=cfg.iterations,
               exact=cfg.exact, monotone_history=cfg.monotone_history, population=pop, progress=progress)

    return assemble_result(cfg, (raw, A_norm, A_input, real_class, train_idx, test_idx, W0), res, with_importance)


def assemble_result(cfg: ExperimentConfig, init, res: TuningResult, with_importance: bool = False) -> ExperimentResult:
    """Оцінки до/після для відомого результату налаштування (без ГА)."""
    raw, A_norm, A_input, real_class, train_idx, test_idx, W0 = init
    real_output = A_norm[:, -1]
    ev = lambda W, t, idx: evaluate(cfg, W, t, A_input[idx], real_output[idx], real_class[idx])
    all_idx = np.arange(len(raw))
    return ExperimentResult(
        cfg, raw, A_norm, real_class, train_idx, test_idx, W0, res,
        before=ev(W0, cfg.initial_thresholds, all_idx),
        after=ev(res.W, res.thresholds, all_idx),
        test_before=ev(W0, cfg.initial_thresholds, test_idx),
        test_after=ev(res.W, res.thresholds, test_idx),
        importance=concept_importance(res.W, -1, cfg.iterations, cfg.exact) if with_importance else None,
    )


def rebuild_result(cfg: ExperimentConfig, W, thresholds, history_mse=(), history_raw=()) -> ExperimentResult:
    """Відновлення результату збереженого проєкту: ваги/пороги відомі, ГА не запускається."""
    init = initial_model(cfg)
    res = TuningResult(np.asarray(W, float), list(thresholds or []), list(history_mse), list(history_raw))
    return assemble_result(cfg, init, res)
