"""Налаштування НКК генетичним алгоритмом: хромосома [W | T] (ваги ненульових дуг + пороги класів).

Повторює legacy/med_experiment.py, розділи 2.1–4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import fcm, metrics
from .ga import GAConfig, run_ga


@dataclass
class ChromosomeSpec:
    """Опис хромосоми: які дуги налаштовуються, межі генів, початкові значення."""
    gene_indices: list[tuple[int, int]]
    gene_space: list[dict]
    initial_genes: list[float]
    num_weights: int
    num_thresholds: int

    def decode(self, solution, W_base: np.ndarray):
        W = W_base.copy()
        for k, (i, j) in enumerate(self.gene_indices):
            W[i, j] = solution[k]
        thresholds = sorted(solution[self.num_weights:self.num_weights + self.num_thresholds])
        return W, list(thresholds)


def build_chromosome(
    W_initial: np.ndarray,
    initial_thresholds,
    weight_delta: float = 0.3,
    weight_abs_bounds: tuple[float, float] = (0.02, 0.98),
    threshold_bounds: tuple[float, float] = (0.05, 0.95),
    tune_thresholds: bool = True,
) -> ChromosomeSpec:
    """Гени = ненульові ваги (рядки 0..n-2, як у legacy) ± weight_delta в межах ±[0.02, 0.98] + пороги."""
    n = W_initial.shape[0]
    lo_abs, hi_abs = weight_abs_bounds
    gene_indices, gene_space, initial = [], [], []
    for i in range(n - 1):
        for j in range(n):
            w = W_initial[i, j]
            if w != 0:
                gene_indices.append((i, j))
                initial.append(w)
                if w > 0:
                    gene_space.append({"low": max(lo_abs, w - weight_delta), "high": min(hi_abs, w + weight_delta)})
                else:
                    gene_space.append({"low": max(-hi_abs, w - weight_delta), "high": min(-lo_abs, w + weight_delta)})
    num_t = len(initial_thresholds) if tune_thresholds else 0
    if tune_thresholds:
        initial.extend(initial_thresholds)
        gene_space.extend({"low": threshold_bounds[0], "high": threshold_bounds[1]} for _ in range(num_t))
    return ChromosomeSpec(gene_indices, gene_space, initial, len(gene_indices), num_t)


def initial_population(spec: ChromosomeSpec, pop_size: int, threshold_bounds=(0.05, 0.95)) -> list[list[float]]:
    """Перша особина — початкові гени; решта — рівномірно випадкові (глобальний np.random, як у legacy)."""
    pop = [list(spec.initial_genes)]
    for _ in range(pop_size - 1):
        genes = [np.random.uniform(s["low"], s["high"]) for s in spec.gene_space[:spec.num_weights]]
        if spec.num_thresholds:
            genes.extend(sorted(np.random.uniform(threshold_bounds[0], threshold_bounds[1], spec.num_thresholds)))
        pop.append(genes)
    return pop


def hybrid_fitness(matches: int, penalty: float, ssd: float) -> float:
    """Legacy-критерій: 10·збіги − Σ|клас_прогн − клас_реальн| + 1/(SSD + 1e-4)."""
    return matches * 10 - penalty + 1.0 / (ssd + 1e-4)


def ssd_fitness(matches: int, penalty: float, ssd: float) -> float:
    """Чистий МНК (як у тексті статті): максимізуємо −SSD."""
    return -ssd


FITNESS = {"hybrid": hybrid_fitness, "ssd": ssd_fitness}


@dataclass
class TuningResult:
    W: np.ndarray
    thresholds: list[float]
    history_mse: list[float]
    history_mse_raw: list[float] = field(default_factory=list)
    best_fitness: float = float("nan")


def tune(
    W_initial: np.ndarray,
    A_input: np.ndarray,
    real_output: np.ndarray,
    real_class: np.ndarray,
    spec: ChromosomeSpec,
    ga_cfg: GAConfig,
    fixed_thresholds=None,
    fitness: str | Callable = "hybrid",
    iterations: int = 30,
    exact: bool = True,
    monotone_history: bool = True,
    population=None,
    progress: Callable[[int, int], None] | None = None,
) -> TuningResult:
    """Запускає ГА. history_mse — MSE найкращого розв'язку покоління (за замовчуванням — неспадна, як у legacy)."""
    fit = FITNESS[fitness] if isinstance(fitness, str) else fitness
    N = len(real_output)

    def unpack(solution):
        W, t = spec.decode(solution, W_initial)
        if spec.num_thresholds:
            return W, t
        return W, (list(fixed_thresholds) if fixed_thresholds else [])

    def fitness_func(ga_instance, solution, solution_idx):
        W, t = unpack(solution)
        pred = fcm.predict_output(A_input, W, -1, iterations, exact)
        s = metrics.ssd(pred, real_output)
        if not t:                        # числовий вихід: класів немає, лише МНК
            return fit(0, 0.0, s)
        cls = metrics.classify(pred, t)
        matches = int(np.sum(cls == real_class))
        penalty = float(np.sum(np.abs(cls - real_class)))
        return fit(matches, penalty, s)

    hist, hist_raw = [], []

    def on_generation(ga_instance):
        sol, _, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        W, _ = unpack(sol)
        m = metrics.ssd(fcm.predict_output(A_input, W, -1, iterations, exact), real_output) / N
        hist_raw.append(m)
        hist.append(min(m, hist[-1]) if (monotone_history and hist) else m)
        if progress:
            progress(ga_instance.generations_completed, ga_cfg.num_generations)

    if population is None:
        population = initial_population(spec, ga_cfg.sol_per_pop)
    ga = run_ga(fitness_func, spec.gene_space, population, ga_cfg, on_generation)
    best, best_fit, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
    W, t = unpack(best)
    return TuningResult(W, t, hist, hist_raw, float(best_fit))


# ======================================================================
# Налаштування ваг НКК "багато входів – багато виходів" (стаття 2, legacy/saaty_tuning.py)
# ======================================================================

@dataclass
class MimoTuningResult:
    W_before: np.ndarray
    W_after: np.ndarray
    genes_after: list[float]
    history_mse: list[float]                 # MSE найкращого розв'язку по поколіннях
    pred_before: np.ndarray                  # (N, |outputs|)
    pred_after: np.ndarray
    sse_before: np.ndarray                   # (N,) сума квадратів по виходах для кожного сценарію
    sse_after: np.ndarray
    mse_before: float
    mse_after: float
    mad_before: float
    mad_after: float


def structure_to_matrix(structure: list[dict], genes, n: int) -> np.ndarray:
    W = np.zeros((n, n))
    for k, c in enumerate(structure):
        W[c["src"], c["tgt"]] = genes[k]
    return W


def predict_mimo(W, X_in, input_idx, output_idx, steps: int = 50, blowup: float | None = 1e5) -> np.ndarray:
    """Виходи для кожного рядка входів: рівно `steps` кроків (без ε), нормування на max|x| (legacy::predict_outputs)."""
    n = W.shape[0]
    out = []
    for row in np.atleast_2d(X_in):
        x0 = np.zeros(n)
        x0[input_idx] = row
        final, _ = fcm.run_until_stable(x0, W, steps, eps=None, blowup=blowup)
        out.append(fcm.normalize_maxabs(final)[output_idx])
    return np.array(out)


def tune_mimo(
    structure: list[dict],
    X_in: np.ndarray,
    Y_out: np.ndarray,
    input_idx, output_idx, n: int,
    ga_cfg: GAConfig,
    steps: int = 50,
    blowup: float | None = 1e5,
    population_seed: int | None = 42,
    progress: Callable[[int, int], None] | None = None,
) -> MimoTuningResult:
    """ГА над вагами дуг структури (fitness = 1/MSE). Перша особина — експертні ваги, решта — U[min, max].

    population_seed — seed для генерації початкової популяції (у legacy не задавався → результат
    між запусками трохи відрізнявся). None = як legacy (без seed).
    """
    X_in, Y_out = np.asarray(X_in, float), np.asarray(Y_out, float)

    def mse_of(genes):
        P = predict_mimo(structure_to_matrix(structure, genes, n), X_in, input_idx, output_idx, steps, blowup)
        total = 0  # порядок підсумовування як у legacy::calculate_mse (по рядках)
        for p, y in zip(P, Y_out):
            total += np.sum((p - y) ** 2)
        return total / Y_out.size

    def fitness_func(ga_instance, solution, solution_idx):
        m = mse_of(solution)
        return 99999999 if m == 0 else 1.0 / m

    hist = []

    def on_generation(gi):
        _, f, _ = gi.best_solution(pop_fitness=gi.last_generation_fitness)
        hist.append(1.0 / f if f > 0 else 100)
        if progress:
            progress(gi.generations_completed, ga_cfg.num_generations)

    if population_seed is not None:
        np.random.seed(population_seed)
    init = [c["init"] for c in structure]
    pop = [init] + [[np.random.uniform(c["min"], c["max"]) for c in structure]
                    for _ in range(ga_cfg.sol_per_pop - 1)]
    gene_space = [{"low": c["min"], "high": c["max"]} for c in structure]
    gi = run_ga(fitness_func, gene_space, pop, ga_cfg, on_generation)
    best, _, _ = gi.best_solution(pop_fitness=gi.last_generation_fitness)
    return mimo_result(structure, list(best), hist, X_in, Y_out, input_idx, output_idx, n, steps, blowup)


def mimo_result(structure, genes, history, X_in, Y_out, input_idx, output_idx, n: int,
                steps: int = 50, blowup: float | None = 1e5) -> MimoTuningResult:
    """Результат налаштування з відомих генів (без ГА) — для відновлення збереженого проєкту."""
    X_in, Y_out = np.asarray(X_in, float), np.asarray(Y_out, float)
    init = [c["init"] for c in structure]
    best = list(genes)
    hist = list(history)
    Wb, Wa = structure_to_matrix(structure, init, n), structure_to_matrix(structure, best, n)
    Pb = predict_mimo(Wb, X_in, input_idx, output_idx, steps, blowup)
    Pa = predict_mimo(Wa, X_in, input_idx, output_idx, steps, blowup)
    return MimoTuningResult(
        Wb, Wa, list(best), hist, Pb, Pa,
        np.sum((Pb - Y_out) ** 2, axis=1), np.sum((Pa - Y_out) ** 2, axis=1),
        metrics.mse(Pb, Y_out), metrics.mse(Pa, Y_out), metrics.mad(Pb, Y_out), metrics.mad(Pa, Y_out),
    )
