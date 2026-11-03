"""Конфіг задачі емпіричного моделювання (YAML). Значення за замовчуванням = legacy/med_experiment.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .ga import GAConfig


@dataclass
class Concept:
    id: str
    name: str = ""
    rule: str | float = "a"   # 'a' більше-краще, 'b' менше-краще, 'c' середина, або число b
    unit: str = ""


@dataclass
class ExperimentConfig:
    name: str
    data: pd.DataFrame
    concepts: list[Concept]                # вхідні концепти (у порядку стовпців)
    output_column: str
    class_labels: list[str]                # назви класів d1..dm (порожньо для числового виходу)
    output_type: str = "classes"           # 'classes' — клас 1..m з порогами; 'numeric' — числове значення
    scale: str = "unit"                    # шкала відображення: 'unit' [0, 1] або 'sym' [-1, 1] (core.scale)
    perfection_c: float = 0.389
    significance: float = 0.7
    weight_threshold: float = 0.09
    no_incoming: list[str] = field(default_factory=list)   # на ці концепти ніщо не впливає
    no_outgoing_output: bool = True                        # вихід ні на що не впливає
    iterations: int = 30
    exact: bool = True
    seed: int = 42
    train_fraction: float = 1.0            # 1.0 = legacy: train = test = усі дані
    initial_thresholds: list[float] | None = None
    tune_thresholds: bool = True
    weight_delta: float = 0.3
    weight_abs_bounds: tuple = (0.02, 0.98)
    threshold_bounds: tuple = (0.05, 0.95)
    fitness: str = "hybrid"                # 'hybrid' (legacy) або 'ssd' (чистий МНК)
    monotone_history: bool = True
    ga: GAConfig = field(default_factory=GAConfig)

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    @property
    def is_numeric(self) -> bool:
        return self.output_type == "numeric"

    @property
    def concept_ids(self) -> list[str]:
        return [c.id for c in self.concepts] + [self.output_column]

    def matrix(self) -> np.ndarray:
        """Сира таблиця (N, n): входи + вихід останнім стовпцем."""
        cols = [c.id for c in self.concepts] + [self.output_column]
        return self.data[cols].to_numpy(dtype=float)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_path = (path.parent / raw["data"]).resolve()
    data = pd.read_csv(data_path)
    concepts = [Concept(**c) for c in raw["concepts"]]
    out = raw["output"]
    model = raw.get("model", {})
    tuning = raw.get("tuning", {})
    cfg = ExperimentConfig(
        name=raw.get("name", path.stem),
        data=data,
        concepts=concepts,
        output_column=out["column"],
        class_labels=out.get("classes", []),
        output_type=out.get("type", "classes"),
        ga=GAConfig.from_dict(tuning.get("ga")),
        **{k: v for k, v in {**model, **{k: v for k, v in tuning.items() if k != "ga"}}.items()},
    )
    if cfg.is_numeric:
        cfg.initial_thresholds, cfg.tune_thresholds = None, False
        if "fitness" not in tuning:
            cfg.fitness = "ssd"
    elif cfg.initial_thresholds is None:
        m = cfg.n_classes
        cfg.initial_thresholds = [round(k / m, 4) for k in range(1, m)]
    return cfg
