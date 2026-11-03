"""Прямий та зворотний вивід на НКК у шкалі [-1, 1] (статті 1 і 4, узагальнені динамічні системи).

Повторює legacy/transport_forecast.py та legacy/transport_diagnosis.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fcm
from .ga import GAConfig, run_ga


# ---------- Прямий вивід (прогнозування) ----------

@dataclass
class ForecastResult:
    history: np.ndarray       # (кроки+1, n) — сирі значення концептів, history[0] = початковий вектор
    final: np.ndarray         # сирий стаціонарний вектор
    normalized: np.ndarray    # final / max|final|
    steps: int


def forecast(W, causes, cause_idx, max_iter: int = 15, eps: float | None = 1e-4) -> ForecastResult:
    """Траєкторія НКК від заданих причин (legacy/transport_forecast.py: 15 кроків, ε = 1e-4)."""
    x0 = np.zeros(W.shape[0])
    x0[cause_idx] = causes
    final, hist = fcm.run_until_stable(x0, W, max_iter, eps, record=True)
    return ForecastResult(hist, final, fcm.normalize_maxabs(final), len(hist) - 1)


def predict_effects(W, causes, cause_idx, effect_idx, max_iter: int = 50, eps: float | None = 1e-3) -> np.ndarray:
    """Наслідки для заданих причин (legacy::simulate_fcm_step_1_4): нормування на max|x|, обрізання [-1, 1]."""
    x0 = np.zeros(W.shape[0])
    x0[cause_idx] = causes
    final, _ = fcm.run_until_stable(x0, W, max_iter, eps)
    return np.clip(fcm.normalize_maxabs(final)[effect_idx], -1.0, 1.0)


# ---------- Зворотний вивід (діагностика) ----------

def default_diagnosis_ga() -> GAConfig:
    """Параметри ГА з legacy/transport_diagnosis.py."""
    return GAConfig(num_generations=100, num_parents_mating=4, sol_per_pop=4, parent_selection_type="rws",
                    keep_elitism=2, crossover_type="uniform", mutation_type="random",
                    mutation_probability=0.2, random_seed=None)


@dataclass
class DiagnosisResult:
    causes: np.ndarray                   # відновлені причини x*
    predicted_effects: np.ndarray        # наслідки для x*
    error: float                         # Σ(наслідки(x*) − цільові)^2
    best_run: int
    run_errors: list[float]
    cause_history: np.ndarray | None = None   # (покоління, q) — найкращий розв'язок по поколіннях
    error_history: list[float] = field(default_factory=list)


def backward_inference(
    W, target_effects, cause_idx, effect_idx,
    ga: GAConfig | None = None,
    n_runs: int = 50,
    bounds=(-1.0, 1.0),
    reference=None,
    seed: int | None = None,
    record_history: bool = False,
    max_iter: int = 50,
    eps: float | None = 1e-3,
    fitness_threshold: float = 50.0,
) -> DiagnosisResult:
    """Пошук причин x*, для яких наслідки НКК найближчі до target_effects (min Σ(ŷ − y)²).

    Як legacy::run_diagnosis_ga: n_runs незалежних запусків ГА, fitness = 1/(error + 1e-8).
    Вибір найкращого запуску:
      reference задано (перевірка на еталонних сценаріях, як у legacy): перший запуск, далі —
        запуск із fitness > fitness_threshold і меншою середньою відстанню |x* − reference|;
      reference = None (реальна діагностика): запуск із найменшою помилкою.
    seed — відтворюваність: запуск r отримує random_seed = seed + r (у legacy seed не задавався).
    """
    ga = ga or default_diagnosis_ga()
    target = np.asarray(target_effects, dtype=float)

    def fitness_func(ga_instance, solution, solution_idx):
        pred = predict_effects(W, solution, cause_idx, effect_idx, max_iter, eps)
        return 1.0 / (np.sum((pred - target) ** 2) + 1e-8)

    best = None
    best_dist = float("inf")
    best_fit = -float("inf")
    run_errors = []
    for r in range(n_runs):
        run_cfg = GAConfig(**{**ga.to_dict(), "random_seed": (seed + r) if seed is not None else ga.random_seed})
        hist = []

        def on_generation(gi, hist=hist):
            sol, _, _ = gi.best_solution(pop_fitness=gi.last_generation_fitness)
            hist.append(sol.copy())

        gi = run_ga(fitness_func, {"low": bounds[0], "high": bounds[1]}, None, run_cfg,
                    on_generation if record_history else None, num_genes=len(cause_idx))
        sol, fit, _ = gi.best_solution(pop_fitness=gi.last_generation_fitness)
        run_errors.append(1.0 / fit - 1e-8)

        if reference is not None:
            dist = float(np.mean(np.abs(sol - np.asarray(reference))))
            better = best is None or (fit > fitness_threshold and dist < best_dist)
        else:
            dist = 0.0
            better = best is None or fit > best_fit
        if better:
            best_dist, best_fit = dist, fit
            best = (r, sol.copy(), list(gi.best_solutions_fitness), np.array(hist) if record_history else None)

    r, sol, fit_curve, hist = best
    pred = predict_effects(W, sol, cause_idx, effect_idx, max_iter, eps)
    return DiagnosisResult(sol, pred, float(np.sum((pred - target) ** 2)), r, run_errors,
                           hist, [1.0 / f - 1e-8 for f in fit_curve])
