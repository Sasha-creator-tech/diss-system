"""Готові моделі: збереження у JSON, відкриття та використання без повторного навчання.

EmpiricalModel — модель «за даними» (шкала [0, 1], перфектність), прогноз класу з СИРИХ показників.
GDSModel       — експертна НКК (шкала [-1, 1]): прогноз і зворотний вивід.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import diagnosis, fcm, metrics, normalize

FORMAT_VERSION = 1


@dataclass
class EmpiricalModel:
    name: str
    concept_ids: list[str]           # входи + вихід останнім
    concept_names: list[str]
    units: list[str]
    rules: list                      # правило a/b/c (або число b) для кожного входу
    class_labels: list[str]
    x_min: list[float]               # межі нормалізації (входи + вихід)
    x_max: list[float]
    W: list[list[float]]
    thresholds: list[float]
    output_type: str = "classes"     # 'classes' або 'numeric'
    project_scale: str = "unit"      # шкала відображення (core.scale)
    perfection_c: float = normalize.DEFAULT_C
    iterations: int = 30
    exact: bool = False
    meta: dict = field(default_factory=dict)
    kind: str = "empirical"
    scale: str = "[0, 1]"

    @property
    def n_inputs(self) -> int:
        return len(self.concept_ids) - 1

    def normalize_inputs(self, X_raw) -> np.ndarray:
        """Сирі входи (N, n_inputs) → вхідна матриця НКК (N, n) з виходом = 0 (як experiment.prepare)."""
        X = np.atleast_2d(np.asarray(X_raw, dtype=float))
        lo, hi = np.asarray(self.x_min[:-1]), np.asarray(self.x_max[:-1])
        A = normalize.normalize_table(X, self.rules, self.perfection_c, bounds=(lo, hi))
        return np.hstack([A, np.zeros((len(A), 1))])

    @property
    def is_numeric(self) -> bool:
        return self.output_type == "numeric"

    def predict(self, X_raw):
        """→ (нормований вихід у [0, 1], клас 1..m або None для числового виходу)."""
        W = np.asarray(self.W)
        pred = fcm.predict_output(self.normalize_inputs(X_raw), W, -1, self.iterations, self.exact)
        if self.is_numeric or not self.thresholds:
            return pred, None
        return pred, metrics.classify(pred, self.thresholds)

    def physical(self, v01):
        """Нормований вихід [0, 1] → одиниці вихідної змінної."""
        return self.x_min[-1] + np.asarray(v01, float) * (self.x_max[-1] - self.x_min[-1])

    @classmethod
    def from_result(cls, r, name: str | None = None) -> "EmpiricalModel":
        """З ExperimentResult (core.experiment)."""
        cfg = r.config
        raw = r.raw
        return cls(
            name=name or cfg.name,
            concept_ids=cfg.concept_ids,
            concept_names=[c.name or c.id for c in cfg.concepts] + [cfg.output_column],
            units=[c.unit for c in cfg.concepts] + [""],
            rules=[c.rule for c in cfg.concepts],
            class_labels=list(cfg.class_labels),
            x_min=raw.min(axis=0).tolist(),
            x_max=raw.max(axis=0).tolist(),
            W=np.asarray(r.tuning.W).tolist(),
            thresholds=[float(t) for t in (r.tuning.thresholds or [])],
            output_type=cfg.output_type,
            project_scale=getattr(cfg, "scale", "unit") or "unit",
            perfection_c=cfg.perfection_c,
            iterations=cfg.iterations,
            exact=cfg.exact,
            meta={"N": int(len(raw)), "matches_after": r.after.matches, "matches_before": r.before.matches,
                  "r2_after": r.after.r2, "fitness": cfg.fitness,
                  "generations": cfg.ga.num_generations, "seed": cfg.seed},
        )


@dataclass
class GDSModel:
    name: str
    concept_ids: list[str]
    concept_names: list[str]
    cause_idx: list[int]
    effect_idx: list[int]
    W: list[list[float]]
    forecast: dict = field(default_factory=lambda: {"max_iter": 15, "eps": 1e-4})
    diagnosis: dict = field(default_factory=lambda: {"max_iter": 50, "eps": 1e-3})
    meta: dict = field(default_factory=dict)
    project_scale: str = "sym"
    kind: str = "gds"
    scale: str = "[-1, 1]"

    def forecast_run(self, causes) -> diagnosis.ForecastResult:
        return diagnosis.forecast(np.asarray(self.W), causes, self.cause_idx,
                                  self.forecast.get("max_iter", 15), self.forecast.get("eps", 1e-4))

    def effects(self, causes) -> np.ndarray:
        return diagnosis.predict_effects(np.asarray(self.W), causes, self.cause_idx, self.effect_idx,
                                         self.diagnosis.get("max_iter", 50), self.diagnosis.get("eps", 1e-3))


def to_json(model) -> str:
    d = asdict(model)
    d["format_version"] = FORMAT_VERSION
    return json.dumps(d, ensure_ascii=False, indent=1)


def from_json(text: str):
    d = json.loads(text)
    d.pop("format_version", None)
    kind = d.get("kind")
    if kind == "empirical":
        return EmpiricalModel(**d)
    if kind == "gds":
        return GDSModel(**d)
    raise ValueError(f"Невідомий тип моделі: {kind!r}")


def save(model, path) -> Path:
    path = Path(path)
    path.write_text(to_json(model), encoding="utf-8")
    return path


def load(path):
    return from_json(Path(path).read_text(encoding="utf-8"))
