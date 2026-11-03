"""CLI: відтворення експериментів без UI.

    python -m core.run examples/ihd/config.yaml                     # емпіричне моделювання (ІХС)
    python -m core.run examples/transport/config.yaml               # УДС: прогноз + діагностика + Сааті
    python -m core.run examples/transport/config.yaml --task diagnosis
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _progress(every):
    def p(g, total):
        if g % every == 0 or g == total:
            print(f"  {g}/{total}", flush=True)
    return p


def run_empirical(a):
    from .config import load_config
    from .experiment import run_experiment
    from .reports import save_all, table7_confusion

    cfg = load_config(a.config)
    if a.generations:
        cfg.ga.num_generations = a.generations
    if a.fast:
        cfg.exact = False
    print(f"Експеримент: {cfg.name}  (N={len(cfg.data)}, концептів={len(cfg.concept_ids)}, шкала [0, 1])")
    t0 = time.time()
    r = run_experiment(cfg, with_importance=not a.no_importance, progress=_progress(50))
    print(f"Готово за {time.time() - t0:.1f} с")

    labels = [f"C{i + 1}" for i in range(r.W_initial.shape[0])]
    fmt = lambda x: f"{x:.3f}"
    print("\nМатриця ваг до налаштування:\n" + pd.DataFrame(r.W_initial, labels, labels).to_string(float_format=fmt))
    print("\nМатриця ваг після налаштування:\n" + pd.DataFrame(r.tuning.W, labels, labels).to_string(float_format=fmt))
    if not cfg.is_numeric:
        print(f"\nПороги: до {cfg.initial_thresholds}  ->  після {np.round(r.tuning.thresholds, 4).tolist()}")
    if cfg.train_fraction >= 1.0:
        print("\n⚠ train_fraction = 1.0: навчання і перевірка на тих самих даних (як у legacy).")
    for name, e in [("ДО налаштування (усі)", r.before), ("ПІСЛЯ налаштування (усі)", r.after),
                    ("ДО налаштування (тест)", r.test_before), ("ПІСЛЯ налаштування (тест)", r.test_after)]:
        if cfg.is_numeric:
            print(f"{name:28s} MAD={e.mad:.4f} MSE={e.mse:.4f} R²={e.r2:.4f}")
        else:
            n = len(e.classes)
            print(f"{name:28s} MAD={e.mad:.4f} MSE={e.mse:.4f} збіги {e.matches}/{n} ({100 * e.matches / n:.1f}%)")
    if not cfg.is_numeric:
        print("\nТаблиця 7 (до / після):\n" + table7_confusion(r).to_string())
    if r.importance is not None:
        imp = r.importance
        print(f"\nA_min={imp.a_min:.3f}, A_max={imp.a_max:.3f}")
        print("I(C_i): " + ", ".join(f"C{i + 1}={v:.3f}" for i, v in zip(imp.inputs, imp.values)))
    out = save_all(r, a.out or f"results/{cfg.name}", plots=not a.no_plots)
    print(f"\nРезультати збережено у {out}/")


def run_gds(a):
    from . import gds

    cfg = gds.load_gds_config(a.config)
    out = Path(a.out or f"results/{cfg.name}")
    tasks = ["forecast", "diagnosis", "expert"] if a.task == "all" else [a.task]
    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.4f}"
    print(f"Система: {cfg.name}  (концептів={cfg.n}, причини={[cfg.concept_ids[i] for i in cfg.cause_idx]}, "
          f"наслідки={[cfg.concept_ids[i] for i in cfg.effect_idx]}, шкала {cfg.scale})")

    if "forecast" in tasks:
        f = gds.run_forecast(cfg)
        print(f"\n=== ПРОГНОЗ (прямий вивід) === кроків: {f.steps}")
        print(pd.DataFrame({"concept": cfg.concept_ids, "name": cfg.concept_names, "final": f.final,
                            "normalized": f.normalized}).to_string(index=False, float_format=fmt))
        gds.save_forecast(cfg, f, out / "forecast", not a.no_plots)

    if "diagnosis" in tasks:
        print("\n=== ДІАГНОСТИКА (зворотний вивід) ===")
        t0 = time.time()
        rep = gds.run_diagnosis(cfg)
        print(rep.table.to_string(index=False, float_format=fmt))
        print(f"Середня MSE: {rep.table['MSE'].mean():.5f}   Середня MAD: {rep.table['MAD'].mean():.5f}"
              f"   ({time.time() - t0:.1f} с)")
        gds.save_diagnosis(cfg, rep, out / "diagnosis", not a.no_plots)

    if "expert" in tasks:
        print("\n=== МЕТОД НАЙМЕНШОГО ВПЛИВУ (Сааті) ===")
        if a.generations:
            cfg.expert.setdefault("tuning", {}).setdefault("ga", {})["num_generations"] = a.generations
        t0 = time.time()
        rep = gds.run_expert(cfg, progress=_progress(100))
        print(rep.saaty_table.to_string(index=False))
        r = rep.tuning
        if r is not None:
            print(f"\nНалаштування ГА ({time.time() - t0:.1f} с):")
            for c, g in zip(rep.structure, r.genes_after):
                print(f"  w{c['src'] + 1},{c['tgt'] + 1}: {c['init']:+.4f}  [{c['min']}, {c['max']}]  ->  {g:+.4f}")
            print(f"MSE: {r.mse_before:.5f} -> {r.mse_after:.5f}   MAD: {r.mad_before:.5f} -> {r.mad_after:.5f}")
        gds.save_expert(cfg, rep, out / "expert", not a.no_plots)

    print(f"\nРезультати збережено у {out}/")


def main(argv=None):
    p = argparse.ArgumentParser(description="DissSystem: моделювання на основі НКК + ГА")
    p.add_argument("config", help="шлях до config.yaml")
    p.add_argument("--out", default=None, help="папка результатів (за замовчуванням results/<name>)")
    p.add_argument("--task", default="all", choices=["all", "forecast", "diagnosis", "expert"],
                   help="для task: gds — яку частину запускати")
    p.add_argument("--generations", type=int, default=None, help="перевизначити кількість поколінь ГА")
    p.add_argument("--fast", action="store_true", help="ІХС: швидке матричне множення")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-importance", action="store_true")
    a = p.parse_args(argv)

    if a.no_plots is False:
        import matplotlib
        matplotlib.use("Agg")
    task = (yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}).get("task", "empirical")
    {"empirical": run_empirical, "gds": run_gds}[task](a)


if __name__ == "__main__":
    main()
