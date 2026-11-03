"""Таблиці та графіки результатів (формат = legacy/med_experiment.py, розділи 4, 7–9)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import scale as sc
from .experiment import ExperimentResult


def _sc(r: ExperimentResult) -> str:
    return getattr(r.config, "scale", sc.UNIT) or sc.UNIT


def interval_str(cls: int, t, scale: str = sc.UNIT) -> str:
    """Інтервал виходу для класу: [0; t1), [t1; t2), ..., [t_{m-1}; 1] (як legacy::get_interval), у шкалі проєкту."""
    m = len(t) + 1
    t = [float(sc.convert(x, sc.UNIT, scale)) for x in t]
    lo, hi = sc.lo(scale), sc.hi(scale)
    if cls == 1:
        return f"[{lo:.3f}; {t[0]:.3f})"
    if cls == m:
        return f"[{t[-1]:.3f}; {hi:.3f}]"
    return f"[{t[cls - 2]:.3f}; {t[cls - 1]:.3f})"


def output_physical(r: ExperimentResult, v01):
    """Нормований вихід [0, 1] → одиниці вихідної змінної (min–max навчальних даних)."""
    lo, hi = float(r.raw[:, -1].min()), float(r.raw[:, -1].max())
    return lo + np.asarray(v01, float) * (hi - lo)


def weights_df(W: np.ndarray, labels) -> pd.DataFrame:
    return pd.DataFrame(W, index=labels, columns=labels)


def table3_raw(r: ExperimentResult) -> pd.DataFrame:
    n_in = len(r.config.concepts)
    df = pd.DataFrame(r.raw, columns=[f"x{i + 1}" for i in range(n_in)] + ["Diagnosis"])
    df.index += 1
    df.index.name = "Patient"
    return df


def table4_normalized(r: ExperimentResult) -> pd.DataFrame:
    n_in = len(r.config.concepts)
    A = sc.convert(r.A_norm, sc.UNIT, _sc(r))
    df = pd.DataFrame({f"A{i + 1}": [f"{v:.3f}" for v in A[:, i]] for i in range(n_in)})
    if r.config.is_numeric:
        df[f"A{n_in + 1}"] = [f"{v:.3f}" for v in A[:, -1]]
    else:
        df[f"A{n_in + 1}"] = [interval_str(int(d), r.tuning.thresholds, _sc(r)) for d in r.real_class]
    df.index += 1
    df.index.name = "Patient"
    return df


def table5_weights(r: ExperimentResult) -> pd.DataFrame:
    rows = []
    n = r.W_initial.shape[0]
    for i in range(n):
        for j in range(n):
            w0, w1 = r.W_initial[i, j], r.tuning.W[i, j]
            if abs(w0) > 0.001 or abs(w1) > 0.001:
                rows.append({"Зв'язок": f"w_{i + 1},{j + 1}",
                             "До налаштування": round(w0, 3), "Після налаштування": round(w1, 3)})
    return pd.DataFrame(rows)


def table6_comparison(r: ExperimentResult) -> pd.DataFrame:
    n_out = r.W_initial.shape[0]
    S = _sc(r)
    conv = lambda v: float(sc.convert(v, sc.UNIT, S))
    if r.config.is_numeric:
        phys = output_physical(r, r.after.pred)
        return pd.DataFrame({
            "Номер": np.arange(1, len(r.raw) + 1),
            "Реальне значення": r.raw[:, -1],
            f"A_{n_out} реальне": [f"{conv(v):.3f}" for v in r.A_norm[:, -1]],
            f"A_{n_out} До налаштування": [f"{conv(v):.3f}" for v in r.before.pred],
            f"A_{n_out} Після налаштування": [f"{conv(v):.3f}" for v in r.after.pred],
            "Прогноз (одиниці)": np.round(phys, 4),
        })
    rows = []
    for i, d in enumerate(r.real_class):
        rows.append({
            "Номер": i + 1,
            "Реальний діагноз (d_i)": f"d_{int(d)}",
            f"Інтервал (A_{n_out})": interval_str(int(d), r.tuning.thresholds, S),
            f"A_{n_out} До налаштування": f"{conv(r.before.pred[i]):.3f}",
            "d_i До налаштування": f"d_{int(r.before.classes[i])}",
            f"A_{n_out} Після налаштування": f"{conv(r.after.pred[i]):.3f}",
            "d_i Після налаштування": f"d_{int(r.after.classes[i])}",
        })
    return pd.DataFrame(rows)


def table7_confusion(r: ExperimentResult) -> pd.DataFrame:
    """Матриця 'реальні × обчислені' у форматі 'до / після' (табл. 7). None для числового виходу."""
    if r.config.is_numeric:
        return None
    labels = r.config.class_labels
    b, a = r.before.confusion, r.after.confusion
    df = pd.DataFrame({lab: [f"{b[i, j]} / {a[i, j]}" for i in range(len(labels))] for j, lab in enumerate(labels)},
                      index=labels)
    df.insert(0, "К-ть", b.sum(axis=1))
    df.index.name = "Реальні"
    return df


def importance_df(r: ExperimentResult) -> pd.DataFrame:
    imp = r.importance
    ids = r.config.concept_ids
    return pd.DataFrame({"Концепт": [f"C{i + 1}" for i in imp.inputs],
                         "Назва": [ids[i] for i in imp.inputs],
                         "I(C_i)": np.round(sc.convert(imp.values, sc.UNIT, _sc(r)), 3)})


def plot_convergence(r: ExperimentResult, path=None, dpi: int = 500):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(r.tuning.history_mse)), r.tuning.history_mse, color="black", linewidth=2)
    ax.set_title("Динаміка Комплексної Оптимізації (Мінімізація Похибки)")
    ax.set_ylabel("Середньоквадратична похибка (MSE)")
    ax.set_xlabel("Ітерація (Generation)")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi)
    return fig


def plot_importance(r: ExperimentResult, path=None, dpi: int = 500):
    import matplotlib.pyplot as plt
    vals = r.importance.values
    labels = [f"$C_{{{i + 1}}}$" for i in r.importance.inputs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, vals, color="white", edgecolor="black", linewidth=1.5)
    ax.set_title("$I(C_i)$", fontsize=16, loc="left", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.text(len(labels), -max(vals) * 0.05, "$C_i$", fontsize=16, va="top", ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=dpi)
    return fig


def save_all(r: ExperimentResult, out_dir: str | Path, plots: bool = True) -> Path:
    """Зберігає Table_3..6 CSV (той самий формат, що legacy), Table_7, ваги, ранги, графіки."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table3_raw(r).to_csv(out / "Table_3_Clinical_Data.csv", index_label="Patient")
    table4_normalized(r).to_csv(out / "Table_4_Normalized_Data.csv", index_label="Patient")
    table5_weights(r).to_csv(out / "Table_5_Weights.csv", index=False)
    table6_comparison(r).to_csv(out / "Table_6_Comparison.csv", index=False)
    if not r.config.is_numeric:
        table7_confusion(r).to_csv(out / "Table_7_Confusion.csv")
    labels = [f"C{i + 1}" for i in range(r.W_initial.shape[0])]
    weights_df(r.W_initial, labels).to_csv(out / "W_initial.csv")
    weights_df(r.tuning.W, labels).to_csv(out / "W_optimized.csv")
    pd.DataFrame({"generation": range(len(r.tuning.history_mse)), "mse": r.tuning.history_mse,
                  "mse_raw": r.tuning.history_mse_raw}).to_csv(out / "convergence.csv", index=False)
    if r.importance is not None:
        importance_df(r).to_csv(out / "importance.csv", index=False)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        plot_convergence(r, out / "optimization_dynamics.png")
        if r.importance is not None:
            plot_importance(r, out / "concept_ranking.png")
    return out
