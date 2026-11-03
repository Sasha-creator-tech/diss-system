"""Узагальнені динамічні системи (УДС) у шкалі [-1, 1]: прогноз, діагностика, проєктування за Сааті + налаштування.

Конфіг: examples/transport/config.yaml (task: gds). Відтворює legacy/transport_forecast.py,
legacy/transport_diagnosis.py, legacy/saaty_main.py, legacy/saaty_tuning.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import diagnosis, weights_expert
from .ga import GAConfig
from .tuning import MimoTuningResult, tune_mimo


@dataclass
class GDSConfig:
    name: str
    concept_ids: list[str]
    concept_names: list[str]
    cause_idx: list[int]
    effect_idx: list[int]
    W: np.ndarray
    forecast: dict = field(default_factory=dict)
    diagnosis: dict = field(default_factory=dict)
    expert: dict = field(default_factory=dict)
    base_dir: Path = Path(".")

    @property
    def n(self) -> int:
        return len(self.concept_ids)

    @property
    def scale(self) -> str:
        return "[-1, 1]"


def load_gds_config(path) -> GDSConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    concepts = raw["concepts"]
    ids = [c["id"] for c in concepts]
    return GDSConfig(
        name=raw.get("name", path.stem),
        concept_ids=ids,
        concept_names=[c.get("name", c["id"]) for c in concepts],
        cause_idx=[i for i, c in enumerate(concepts) if c["role"] == "cause"],
        effect_idx=[i for i, c in enumerate(concepts) if c["role"] == "effect"],
        W=np.array(raw["W"], dtype=float),
        forecast=raw.get("forecast", {}),
        diagnosis=raw.get("diagnosis", {}),
        expert=raw.get("expert", {}),
        base_dir=path.parent,
    )


# ---------------- Прогноз ----------------

def run_forecast(cfg: GDSConfig, causes=None) -> diagnosis.ForecastResult:
    f = cfg.forecast
    return diagnosis.forecast(cfg.W, causes if causes is not None else f["causes"], cfg.cause_idx,
                              f.get("max_iter", 15), f.get("eps", 1e-4))


# ---------------- Діагностика ----------------

@dataclass
class DiagnosisReport:
    table: pd.DataFrame                      # сценарії: x, ŷ, x*, MSE, MAD
    results: dict                            # назва сценарію -> DiagnosisResult


def run_diagnosis(cfg: GDSConfig, scenarios: dict | None = None, progress=None) -> DiagnosisReport:
    d = cfg.diagnosis
    ga = GAConfig.from_dict({**diagnosis.default_diagnosis_ga().to_dict(), **d.get("ga", {})})
    scenarios = scenarios or d["scenarios"]
    record = d.get("record_scenario")
    rows, results = [], {}
    q = len(cfg.cause_idx)
    for k, (name, x) in enumerate(scenarios.items()):
        x = np.asarray(x, float)
        y = diagnosis.predict_effects(cfg.W, x, cfg.cause_idx, cfg.effect_idx, d.get("max_iter", 50), d.get("eps", 1e-3))
        r = diagnosis.backward_inference(
            cfg.W, y, cfg.cause_idx, cfg.effect_idx, ga=ga, n_runs=d.get("n_runs", 50),
            reference=x if d.get("select_by_reference", True) else None, seed=d.get("seed", 42),
            record_history=(name == record), max_iter=d.get("max_iter", 50), eps=d.get("eps", 1e-3))
        results[name] = r
        row = {"Scenario": name}
        row.update({f"x{cfg.cause_idx[i] + 1}": x[i] for i in range(q)})
        row.update({f"hat_x{e + 1}": y[i] for i, e in enumerate(cfg.effect_idx)})
        row.update({f"x{cfg.cause_idx[i] + 1}*": r.causes[i] for i in range(q)})
        row["MSE"] = float(np.mean((x - r.causes) ** 2))
        row["MAD"] = float(np.mean(np.abs(x - r.causes)))
        rows.append(row)
        if progress:
            progress(k + 1, len(scenarios))
    return DiagnosisReport(pd.DataFrame(rows), results)


# ---------------- Метод Сааті + налаштування ----------------

@dataclass
class ExpertReport:
    saaty_table: pd.DataFrame
    W_saaty: np.ndarray
    structure: list[dict]
    tuning: MimoTuningResult | None


def load_training(cfg: GDSConfig) -> pd.DataFrame:
    t = cfg.expert["tuning"]
    return pd.read_csv(cfg.base_dir / t["data"])


def run_expert(cfg: GDSConfig, tune: bool = True, progress=None) -> ExpertReport:
    e = cfg.expert
    if "ranks" in e:   # експерт задає ранги напряму: [{src: C1, tgt: C6, sign: +, rank: 3}, ...]
        ids = cfg.concept_ids
        links = [weights_expert.Link(ids.index(l["src"]), ids.index(l["tgt"]),
                                     1 if str(l["sign"]).strip() in ("+", "1", "+1") else -1, int(l["rank"]))
                 for l in e["ranks"]]
    else:              # ранги з відомої матриці W (як у статті)
        links = weights_expert.ranks_from_matrix(cfg.W)
    W_s, table = weights_expert.weights_from_ranks(links, cfg.n)
    structure = weights_expert.weight_intervals(W_s, e.get("variation", 0.2), tuple(e.get("abs_bounds", (0.02, 0.98))),
                                                e.get("interval_decimals", 2), e.get("init_decimals", 4))
    res = None
    if tune and "tuning" in e:
        t = e["tuning"]
        df = load_training(cfg)
        cols = cfg.concept_ids
        X = df[[cols[i] for i in cfg.cause_idx]].to_numpy(float)
        Y = df[[cols[i] for i in cfg.effect_idx]].to_numpy(float)
        ga = GAConfig.from_dict(t.get("ga"))
        res = tune_mimo(structure, X, Y, cfg.cause_idx, cfg.effect_idx, cfg.n, ga, t.get("steps", 50),
                        t.get("blowup", 1e5), t.get("population_seed", 42), progress)
    return ExpertReport(table, W_s, structure, res)


# ---------------- Звіти ----------------

def _style(ax, width=1.5):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)


def plot_forecast(cfg: GDSConfig, f: diagnosis.ForecastResult, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.grid(True, linestyle="--", alpha=0.5, color="silver", zorder=0)
    for i in range(cfg.n):
        ax.plot(f.history[:, i], color="#1a1a1a", linewidth=2.5, zorder=2)
    final = f.history[-1].copy()
    adj = final.copy()
    order = np.argsort(final)
    for k in range(1, len(order)):           # розсування підписів, як у legacy
        if adj[order[k]] - adj[order[k - 1]] < 0.075:
            adj[order[k]] = adj[order[k - 1]] + 0.075
    x_lab = len(f.history) - 1 + 0.3
    for i in range(cfg.n):
        ax.text(x_lab, adj[i], f"$C_{{{i + 1}}}$", va="center", ha="left", fontsize=20)
    ax.set_xlim(0, len(f.history) + 0.8)
    _style(ax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_error_curve(curve, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(curve, color="black", linewidth=2.5)
    _style(ax, 2.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    return fig


def plot_cause_dynamics(hist: np.ndarray, i: int, path=None):
    """Динаміка відновленої причини i по поколіннях (лог. шкала X, як legacy)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, LinearLocator
    gens = np.arange(1, len(hist) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    y = hist[:, i]
    ax.plot(gens, y, color="black", linewidth=3.5)
    off = 0.1 if y.max() == y.min() else (y.max() - y.min()) * 0.05
    ax.set_ylim(y.min() - off, y.max() + off)
    ticks = [t for t in [1, 2, 5, 10, 20, 50, 100] if t <= gens[-1]]
    if gens[-1] not in ticks:
        ticks.append(gens[-1])
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _style(ax, 3.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    return fig


def save_forecast(cfg, f, out: Path, plots=True):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(f.history, columns=cfg.concept_ids).rename_axis("step").to_csv(out / "forecast_history.csv")
    pd.DataFrame({"concept": cfg.concept_ids, "final": f.final, "normalized": f.normalized}).to_csv(
        out / "forecast_final.csv", index=False)
    if plots:
        plot_forecast(cfg, f, out / "stabilization.png")


def save_diagnosis(cfg, rep: DiagnosisReport, out: Path, plots=True):
    out.mkdir(parents=True, exist_ok=True)
    rep.table.to_csv(out / "diagnosis_table.csv", index=False)
    if plots:
        for name, r in rep.results.items():
            if r.cause_history is not None:
                plot_error_curve(r.error_history, out / f"convergence_{name}.png")
                for i in range(len(cfg.cause_idx)):
                    plot_cause_dynamics(r.cause_history, i, out / f"dynamics_{name}_C{cfg.cause_idx[i] + 1}.png")


def save_expert(cfg, rep: ExpertReport, out: Path, plots=True):
    out.mkdir(parents=True, exist_ok=True)
    rep.saaty_table.to_csv(out / "saaty_table.csv", index=False)
    pd.DataFrame(rep.W_saaty, cfg.concept_ids, cfg.concept_ids).to_csv(out / "W_saaty.csv")
    r = rep.tuning
    if r is None:
        return
    pd.DataFrame([{"Link": f"w{c['src'] + 1},{c['tgt'] + 1}", "Before": c["init"],
                   "Interval": f"[{c['min']}, {c['max']}]", "After": round(g, 4)}
                  for c, g in zip(rep.structure, r.genes_after)]).to_csv(out / "weights_tuning.csv", index=False)
    pd.DataFrame({"No": range(1, len(r.sse_before) + 1), "SSE Before": r.sse_before, "SSE After": r.sse_after,
                  "Improvement": r.sse_before - r.sse_after}).to_csv(out / "sse_by_scenario.csv", index=False)
    pd.DataFrame({"generation": range(len(r.history_mse)), "mse": r.history_mse}).to_csv(out / "convergence.csv", index=False)
    if plots:
        plot_error_curve(r.history_mse, out / "optimization_dynamics.png")
