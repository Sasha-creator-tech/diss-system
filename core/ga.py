"""Тонка обгортка над pygad. Параметри зі значенням None не передаються (діють значення pygad за замовчуванням)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields

import pygad


@dataclass
class GAConfig:
    num_generations: int = 400
    num_parents_mating: int = 15
    sol_per_pop: int = 50
    parent_selection_type: str = "sss"
    keep_parents: int | None = None
    keep_elitism: int | None = None
    crossover_type: str = "single_point"
    mutation_type: str = "random"
    mutation_percent_genes: float | None = None
    mutation_probability: float | None = None
    random_seed: int | None = 42

    @classmethod
    def from_dict(cls, d: dict | None) -> "GAConfig":
        d = d or {}
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


def run_ga(fitness_func, gene_space, initial_population, cfg: GAConfig, on_generation=None,
           num_genes: int | None = None) -> pygad.GA:
    """Запускає pygad.GA і повертає екземпляр.

    initial_population=None — pygad генерує популяцію сам (як у legacy/transport_diagnosis.py).
    gene_space може бути списком словників або одним словником {'low', 'high'} для всіх генів.
    """
    optional = {k: v for k, v in {
        "keep_parents": cfg.keep_parents,
        "keep_elitism": cfg.keep_elitism,
        "mutation_percent_genes": cfg.mutation_percent_genes,
        "mutation_probability": cfg.mutation_probability,
        "random_seed": cfg.random_seed,
        "initial_population": initial_population,
    }.items() if v is not None}
    if num_genes is None:
        num_genes = len(gene_space) if isinstance(gene_space, list) else len(initial_population[0])
    ga = pygad.GA(
        num_generations=cfg.num_generations,
        num_parents_mating=cfg.num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=cfg.sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type=cfg.parent_selection_type,
        crossover_type=cfg.crossover_type,
        mutation_type=cfg.mutation_type,
        on_generation=on_generation,
        suppress_warnings=True,
        **optional,
    )
    ga.run()
    return ga
