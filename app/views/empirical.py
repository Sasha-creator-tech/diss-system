"""Майстер «Модель за даними» (емпіричне моделювання): Проєкт → Дані → Модель → Налаштування → Використання → Звіт."""
from __future__ import annotations

import io
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app import projects, ui
from app.views import convergence as conv_view
from core import convergence as cv
from core import importance as imp_mod
from core import model_io, reports
from core import scale as sc
from core.config import Concept, ExperimentConfig, load_config
from core.experiment import initial_model, rebuild_result, run_experiment
from core.ga import GAConfig

ROOT = Path(__file__).resolve().parents[2]
IHD_CONFIG = ROOT / "examples" / "ihd" / "config.yaml"

ROLE_IN, ROLE_OUT, ROLE_SKIP = "Вхід", "Вихід", "Не використовувати"
RULE_LABEL = {"a": "більше ↑", "b": "менше ↓", "c": "середина ◆"}
LABEL_RULE = {v: k for k, v in RULE_LABEL.items()}
OUT_CLASSES, OUT_NUMERIC = "classes", "numeric"
STEPS = [("Проєкт", "шкала, тип виходу"), ("Дані", "таблиця"), ("Модель", "карта і ваги"),
         ("Налаштування", "ГА"), ("Використання", "прогноз, що-якщо"), ("Звіт", "експорт")]
S_PROJECT, S_DATA, S_MODEL, S_TUNE, S_USE, S_REPORT = range(6)
PRESETS = {"Швидко · 100 поколінь": 100, "Як у статті · 400": 400, "Ретельно · 1000": 1000}

# Параметри за замовчуванням = legacy/med_experiment.py
DEFAULT_GA = dict(num_generations=400, num_parents_mating=15, sol_per_pop=50, parent_selection_type="sss",
                  keep_parents=5, crossover_type="single_point", mutation_type="random",
                  mutation_percent_genes=15, random_seed=42)


# ---------------- Стан і збереження ----------------

def S() -> dict:
    return st.session_state["emp"]


def _fresh(name: str) -> dict:
    return {"id": projects.new_id(), "name": name, "scale": sc.UNIT, "output_type": OUT_CLASSES,
            "step": 0, "reached": 0, "ver": 0, "df": None, "file_name": None, "settings": None, "classes": None,
            "split": False, "params": dict(significance=0.7, weight_threshold=0.09, no_incoming=[],
                                           no_outgoing_output=True),
            "ga": dict(DEFAULT_GA), "fitness": "hybrid", "exact": False,
            "cfg": None, "init": None, "result": None, "model": None, "importance": None,
            "model_only": False, "saved_at": None, "params_at_tune": None}


def _activate(state: dict):
    st.session_state["emp"] = state
    st.session_state["mode"] = "emp"
    st.rerun()


def new_project():
    _activate(_fresh("Новий проєкт"))


def load_example():
    cfg = load_config(IHD_CONFIG)
    s = _fresh("Диференціальна діагностика ІХС")
    s["df"] = cfg.data.copy()
    s["file_name"] = "ihd_data.csv"
    rows = [{"Стовпець": c.id, "Назва": c.name, "Роль": ROLE_IN, "Що краще": RULE_LABEL[c.rule], "Одиниці": c.unit}
            for c in cfg.concepts]
    rows.append({"Стовпець": cfg.output_column, "Назва": "Діагноз", "Роль": ROLE_OUT, "Що краще": "—", "Одиниці": ""})
    s["settings"] = pd.DataFrame(rows)
    s["classes"] = list(cfg.class_labels)
    s["params"].update(no_incoming=list(cfg.no_incoming))
    _activate(s)


def open_model(m: model_io.EmpiricalModel):
    """Старий файл «лише модель» (без даних): доступні тільки використання та звіт."""
    s = _fresh(m.name)
    s.update(model=m, model_only=True, step=S_USE, reached=S_REPORT, scale=m.project_scale,
             output_type=m.output_type)
    _activate(s)


def to_doc(s) -> dict:
    tuned = None
    if s["result"] is not None:
        t = s["result"].tuning
        tuned = {"W": np.asarray(t.W).tolist(), "thresholds": [float(x) for x in (t.thresholds or [])],
                 "history": [float(x) for x in t.history_mse], "history_raw": [float(x) for x in t.history_mse_raw]}
    state = {k: s[k] for k in ("name", "scale", "output_type", "step", "reached", "file_name", "classes", "split",
                               "params", "ga", "fitness", "exact")}
    state.update(df=None if s["df"] is None else s["df"].to_csv(index=False),
                 settings=None if s["settings"] is None else s["settings"].to_dict("records"), tuned=tuned)
    r = s["result"]
    n_in = 0 if s["settings"] is None else int((s["settings"]["Роль"] == ROLE_IN).sum())
    if r is None:
        status, quality = "Чернетка", "—"
    elif r.config.is_numeric:
        status, quality = "Налаштовано", f"R² {r.after.r2:.3f}"
    else:
        status, quality = "Налаштовано", f"{r.after.matches} / {len(r.real_class)}"
    summary = {"type": "За даними", "concepts": f"{n_in} → 1", "status": status, "quality": quality,
               "rows": None if s["df"] is None else int(len(s["df"]))}
    return projects.make_doc(s["id"], "empirical", s["name"], s["scale"], summary, state)


def from_doc(doc: dict):
    d = doc["state"]
    s = _fresh(d.get("name", doc.get("name", "Проєкт")))
    s["id"] = doc["id"]
    for k in ("scale", "output_type", "step", "reached", "file_name", "classes", "split", "params", "ga",
              "fitness", "exact"):
        if k in d and d[k] is not None:
            s[k] = d[k]
    if d.get("df"):
        s["df"] = pd.read_csv(io.StringIO(d["df"]))
    if d.get("settings"):
        s["settings"] = pd.DataFrame(d["settings"]).fillna("")
    s["saved_at"] = doc.get("saved_at")
    if d.get("tuned") and s["df"] is not None:
        t = d["tuned"]
        cfg = build_cfg(s)
        r = rebuild_result(cfg, t["W"], t["thresholds"], t.get("history", []), t.get("history_raw", []))
        s.update(cfg=cfg, result=r, model=model_io.EmpiricalModel.from_result(r, s["name"]),
                 params_at_tune=dict(s["params"]))
    _activate(s)


def save_project(s, quiet: bool = False) -> bool:
    if s["model_only"] or s["df"] is None:
        if not quiet:
            st.sidebar.warning("Нічого зберігати: спершу завантажте дані.")
        return False
    projects.save(to_doc(s))
    s["saved_at"] = datetime.now().isoformat(timespec="seconds")
    return True


def goto(step: int):
    s = S()
    s["step"] = step
    s["reached"] = max(s["reached"], step)
    s["ver"] += 1           # нова версія віджетів-редакторів → вони стартують із збережених даних
    save_project(s, quiet=True)
    st.rerun()


# ---------------- Допоміжне ----------------

def _short(labels):
    out = []
    for k, lab in enumerate(labels, 1):
        first = str(lab).split()[0] if str(lab).split() else ""
        out.append(first if 0 < len(first) <= 4 else f"{k}")
    return out


def _conv(s, v):
    """Значення у [0, 1] → шкала проєкту."""
    return sc.convert(v, sc.UNIT, s["scale"])


def _default_settings(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out_col = num[-1] if num else df.columns[-1]
    rows = []
    for c in df.columns:
        role = ROLE_OUT if c == out_col else (ROLE_IN if c in num else ROLE_SKIP)
        rows.append({"Стовпець": c, "Назва": c, "Роль": role, "Що краще": "більше ↑" if role == ROLE_IN else "—",
                     "Одиниці": ""})
    return pd.DataFrame(rows)


def _validate(s) -> list[str]:
    errs = []
    st_ = s["settings"]
    if st_ is None:
        return ["Завантажте таблицю."]
    outs = st_[st_["Роль"] == ROLE_OUT]
    ins = st_[st_["Роль"] == ROLE_IN]
    if len(outs) != 1:
        errs.append("Потрібен рівно один стовпець-вихід.")
    if len(ins) < 2:
        errs.append("Потрібно щонайменше 2 вхідні показники.")
    for _, r in ins.iterrows():
        if not pd.api.types.is_numeric_dtype(s["df"][r["Стовпець"]]):
            errs.append(f"Стовпець «{r['Стовпець']}» не числовий — він не може бути входом.")
        if r["Що краще"] not in LABEL_RULE:
            errs.append(f"Для «{r['Стовпець']}» вкажіть, що краще (більше / менше / середина).")
    used = list(ins["Стовпець"]) + list(outs["Стовпець"])
    if len(used) and s["df"][used].isna().any().any():
        errs.append("У вибраних стовпцях є пропуски — заповніть або видаліть ці рядки.")
    if len(outs) == 1:
        out = s["df"][outs.iloc[0]["Стовпець"]]
        if s["output_type"] == OUT_CLASSES and out.nunique() < 2:
            errs.append("У стовпці-виході має бути щонайменше 2 класи.")
        if s["output_type"] == OUT_CLASSES and out.nunique() > 12:
            errs.append(f"У стовпці-виході {out.nunique()} різних значень — схоже на числовий вихід. "
                        "Змініть тип виходу на кроці «Проєкт».")
        if s["output_type"] == OUT_NUMERIC and not pd.api.types.is_numeric_dtype(out):
            errs.append("Для числового виходу стовпець має бути числовим.")
    return errs


def build_cfg(s) -> ExperimentConfig:
    st_ = s["settings"]
    ins = st_[st_["Роль"] == ROLE_IN]
    out = st_[st_["Роль"] == ROLE_OUT].iloc[0]["Стовпець"]
    df = s["df"].copy()
    numeric = s["output_type"] == OUT_NUMERIC
    if numeric:
        labels, thresholds = [], None
    else:
        values = sorted(df[out].unique())
        df[out] = df[out].map({v: k + 1 for k, v in enumerate(values)})    # класи → 1..m (порядок = тяжкість)
        m = len(values)
        labels = s["classes"] if s["classes"] and len(s["classes"]) == m else [str(v) for v in values]
        thresholds = [round(float(t), 4) for t in np.linspace(0.2, 0.8, m - 1)] if m > 2 else [0.5]
    concepts = [Concept(id=r["Стовпець"], name=str(r["Назва"] or r["Стовпець"]), rule=LABEL_RULE[r["Що краще"]],
                        unit="" if pd.isna(r["Одиниці"]) else str(r["Одиниці"]))
                for _, r in ins.iterrows()]
    p = s["params"]
    return ExperimentConfig(
        name=s["name"], data=df, concepts=concepts, output_column=out, class_labels=list(labels),
        output_type=s["output_type"], scale=s["scale"],
        significance=p["significance"], weight_threshold=p["weight_threshold"],
        no_incoming=[c for c in p["no_incoming"] if c in [x.id for x in concepts]],
        no_outgoing_output=p["no_outgoing_output"], exact=s["exact"], seed=s["ga"]["random_seed"],
        train_fraction=0.8 if s["split"] else 1.0,
        initial_thresholds=thresholds, tune_thresholds=not numeric,
        fitness="ssd" if numeric else s["fitness"], ga=GAConfig(**s["ga"]),
    )


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a is None or b is None:
        return a is b
    return a.fillna("").astype(str).reset_index(drop=True).equals(b.fillna("").astype(str).reset_index(drop=True))


def _invalidate(s):
    s["result"] = s["model"] = s["importance"] = None


# ---------------- Кроки ----------------

def step_project(s):
    ui.header("Крок 1. Проєкт", "Назва, шкала значень концептів і тип результату.", s["name"], s["scale"])
    with st.container(border=True):
        s["name"] = st.text_input("Назва проєкту", s["name"], key=f"emp_name_{s['ver']}")
        st.markdown("**Шкала значень концептів**")
        new_scale = st.radio("Шкала", [sc.UNIT, sc.SYM], index=[sc.UNIT, sc.SYM].index(s["scale"]), horizontal=True,
                             key=f"emp_scale_{s['ver']}", label_visibility="collapsed",
                             format_func=lambda x: {sc.UNIT: "[0, 1] — від мінімуму 0 до максимуму 1",
                                                    sc.SYM: "[−1, 1] — від −1 через 0 (середнє) до +1"}[x])
        s["scale"] = new_scale
        st.caption("Шкала визначає, як вводяться і показуються рівні концептів, вихід і межі класів. Розрахунок "
                   "перфектності ведеться у [0, 1] (формула 8), перетворення лінійне — результати не змінюються.")
    with st.container(border=True):
        st.markdown("**Що є результатом моделі**")
        new_out = st.radio("Тип виходу", [OUT_CLASSES, OUT_NUMERIC], horizontal=True, key=f"emp_out_{s['ver']}",
                           index=[OUT_CLASSES, OUT_NUMERIC].index(s["output_type"]), label_visibility="collapsed",
                           format_func=lambda x: {OUT_CLASSES: "Класи (діагнози, рівні тяжкості)",
                                                  OUT_NUMERIC: "Числове значення"}[x])
        if new_out != s["output_type"]:
            s["output_type"] = new_out
            _invalidate(s)
        if s["output_type"] == OUT_CLASSES:
            st.caption("Вихідний стовпець містить номери класів, упорядковані від легшого до тяжчого. ГА налаштовує "
                       "ваги та межі між класами.")
        else:
            st.caption("Вихідний стовпець — число (наприклад, показник ефективності). ГА налаштовує ваги за МНК, "
                       "якість оцінюється MAD, MSE і R².")
    if st.button("Далі: дані →", type="primary", key="emp_next0"):
        goto(S_DATA)


def step_data(s):
    ui.header("Крок 2. Дані", "Перевірте, як система зрозуміла таблицю. Для кожного показника вкажіть роль і «що краще».",
              s["name"], s["scale"])
    up = st.file_uploader("Таблиця спостережень (CSV)", type=["csv", "txt"], key=f"emp_csv_{s['ver']}")
    if up is not None and up.name != s.get("file_name"):
        try:
            df = pd.read_csv(io.BytesIO(up.getvalue()), sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            st.error(f"Не вдалося прочитати CSV: {e}")
            return
        s.update(df=df, settings=_default_settings(df), classes=None, file_name=up.name)
        _invalidate(s)
        s["ver"] += 1
        st.rerun()

    if s["df"] is None:
        ui.note("Формат: один рядок — один випадок (пацієнт, об’єкт); стовпці — показники; один стовпець — результат "
                + ("(клас 1, 2, 3… упорядкований від легшого до тяжчого)." if s["output_type"] == OUT_CLASSES
                   else "(числове значення).") + " Або відкрийте приклад ІХС на стартовому екрані.", "muted")
        return

    df = s["df"]
    miss = int(df.isna().sum().sum())
    ui.note(f"<b>{s.get('file_name') or 'data.csv'}</b> · {len(df)} рядків · {df.shape[1]} стовпців · "
            f"{'пропусків немає' if miss == 0 else f'пропусків: {miss}'}", "info")

    st.markdown("#### Показники (концепти)")
    settings = s["settings"].copy()
    settings["Діапазон"] = [f"{df[c].min():g} – {df[c].max():g}" if pd.api.types.is_numeric_dtype(df[c]) else "текст"
                            for c in settings["Стовпець"]]
    edited = st.data_editor(
        settings, key=f"emp_settings_{s['ver']}", hide_index=True, width="stretch",
        column_config={
            "Стовпець": st.column_config.TextColumn(disabled=True),
            "Назва": st.column_config.TextColumn(help="Як показувати в інтерфейсі"),
            "Роль": st.column_config.SelectboxColumn(options=[ROLE_IN, ROLE_OUT, ROLE_SKIP], required=True),
            "Що краще": st.column_config.SelectboxColumn(options=list(RULE_LABEL.values()) + ["—"],
                                                         help="Нечітка перфектність: більше ↑ (b=1), менше ↓ (b=0), середина (b=0.5)"),
            "Одиниці": st.column_config.TextColumn(),
            "Діапазон": st.column_config.TextColumn(disabled=True),
        })
    new_settings = edited.drop(columns=["Діапазон"]).reset_index(drop=True)
    if not _same(new_settings, s["settings"]):
        _invalidate(s)
    s["settings"] = new_settings

    outs = s["settings"][s["settings"]["Роль"] == ROLE_OUT]
    if len(outs) == 1 and s["output_type"] == OUT_CLASSES:
        out = outs.iloc[0]["Стовпець"]
        values = sorted(df[out].dropna().unique())
        if len(values) <= 12:
            counts = df[out].value_counts()
            st.markdown("#### Класи результату")
            st.caption("Упорядковані від легшого до тяжчого — це важливо для меж між класами.")
            names = s["classes"] if s["classes"] and len(s["classes"]) == len(values) else [f"Клас {v}" for v in values]
            cls_df = pd.DataFrame({"Значення": [str(v) for v in values], "Назва": names,
                                   "Кількість": [int(counts.get(v, 0)) for v in values]})
            ed = st.data_editor(cls_df, key=f"emp_classes_{s['ver']}", hide_index=True, width="stretch",
                                column_config={"Значення": st.column_config.TextColumn(disabled=True),
                                               "Кількість": st.column_config.NumberColumn(disabled=True)})
            s["classes"] = list(ed["Назва"])
    elif len(outs) == 1:
        col = df[outs.iloc[0]["Стовпець"]]
        if pd.api.types.is_numeric_dtype(col):
            st.caption(f"Числовий вихід «{outs.iloc[0]['Стовпець']}»: від {col.min():g} до {col.max():g}, "
                       f"середнє {col.mean():.3g}.")

    st.markdown("#### Перевірка моделі")
    choice = st.radio("Перевірка моделі", ["Використати всі дані (як у статті)", "Відкласти 20% для чесної перевірки"],
                      index=1 if s["split"] else 0, key=f"emp_split_{s['ver']}", label_visibility="collapsed")
    split = choice.startswith("Відкласти")
    if split != s["split"]:
        s["split"] = split
        _invalidate(s)

    errs = _validate(s)
    for e in errs:
        st.error(e)
    c1, c2 = st.columns([1, 5])
    if c1.button("← Назад", key="emp_back1"):
        goto(S_PROJECT)
    if c2.button("Далі: побудувати модель →", type="primary", disabled=bool(errs), key="emp_next1"):
        goto(S_MODEL)


def step_model(s):
    ui.header("Крок 3. Модель", "Ваги зв’язків обчислено з даних за подібністю показників (формули 10–15).",
              s["name"], s["scale"])
    errs = _validate(s)
    if errs:
        st.error("Спершу виправте дані на кроці 2: " + errs[0])
        return
    cfg = build_cfg(s)
    ids = cfg.concept_ids
    names = [c.name or c.id for c in cfg.concepts] + ["Вихід"]

    left, right = st.columns([3, 1.25], gap="large")
    with right, st.container(border=True):
        st.markdown("**Правила побудови**")
        p = s["params"]
        p["significance"] = st.number_input("Поріг значущості |S|", 0.5, 0.95, float(p["significance"]), 0.05,
                                            key=f"emp_sig_{s['ver']}")
        p["weight_threshold"] = st.number_input("Відкинути ваги слабші за", 0.0, 0.5, float(p["weight_threshold"]), 0.01,
                                                key=f"emp_wt_{s['ver']}")
        p["no_incoming"] = st.multiselect("На ці показники ніщо не впливає", [c.id for c in cfg.concepts],
                                          default=[c for c in p["no_incoming"] if c in ids], key=f"emp_noin_{s['ver']}",
                                          format_func=lambda c: names[ids.index(c)])
        p["no_outgoing_output"] = st.checkbox("Вихід ні на що не впливає", p["no_outgoing_output"],
                                              key=f"emp_noout_{s['ver']}")
        if s["result"] is not None and s["params_at_tune"] is not None and p != s["params_at_tune"]:
            _invalidate(s)
            st.caption("Правила змінено — налаштування на кроці 4 треба повторити.")
        cfg = build_cfg(s)

    raw, A_norm, A_input, real_class, tr, te, W0 = initial_model(cfg)
    s["cfg"], s["init"] = cfg, W0
    n = W0.shape[0]
    n_links = int(np.count_nonzero(W0))
    labels = [f"C{i + 1}" for i in range(n)]

    with left:
        st.markdown(f"Знайдено **{n_links}** зв’язків, з них до виходу — **{int(np.count_nonzero(W0[:, -1]))}**.")
        t_graph, t_mat, t_list = st.tabs(["Граф", "Матриця", "Список зв’язків"])
        with t_graph:
            only = st.checkbox("Лише зв’язки з виходом", False, key="emp_only_out")
            st.graphviz_chart(ui.graph_dot(W0, labels, only_to=n - 1 if only else None, center=n - 1),
                              width="stretch")
            st.caption("Синій — посилює (+), помаранчевий пунктир — послаблює (−), товщина — сила впливу. "
                       + " · ".join(f"C{i + 1} — {names[i]}" for i in range(n)))
        with t_mat:
            dfw = pd.DataFrame(W0, index=labels, columns=labels)

            def color(v):
                if v == 0:
                    return "color: #C4CAD3"
                rgb = "31,79,201" if v > 0 else "194,98,12"
                return (f"background-color: rgba({rgb},{min(0.15 + abs(v) * 2.5, 0.9):.2f}); "
                        f"color: {'#FFF' if abs(v) > 0.2 else '#18212C'}")
            st.dataframe(dfw.style.map(color).format("{:.3f}"), width="stretch")
        with t_list:
            rows = [{"Зв’язок": f"C{i + 1} → C{j + 1}", "Від": names[i], "До": names[j], "Вага": round(W0[i, j], 3)}
                    for i in range(n) for j in range(n) if W0[i, j] != 0]
            st.dataframe(pd.DataFrame(rows).sort_values("Вага", key=abs, ascending=False), hide_index=True,
                         width="stretch")

    c1, c2 = st.columns([1, 5])
    if c1.button("← Назад", key="emp_back2"):
        goto(S_DATA)
    if c2.button("Далі: налаштувати точність →", type="primary", key="emp_next2", disabled=n_links == 0):
        goto(S_TUNE)


def step_tune(s):
    numeric = s["output_type"] == OUT_NUMERIC
    ui.header("Крок 4. Налаштування", "Генетичний алгоритм підбирає ваги зв’язків"
              + ("" if numeric else " і межі між класами") + ", щоб модель якнайточніше відтворювала дані.",
              s["name"], s["scale"])
    if _validate(s):
        st.error("Спершу виправте дані на кроці 2.")
        return
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        cur = next((k for k, v in PRESETS.items() if v == s["ga"]["num_generations"]), None)
        mode = c1.segmented_control("Режим", list(PRESETS), default=cur, key=f"emp_preset_{s['ver']}")
        if mode:
            s["ga"]["num_generations"] = PRESETS[mode]
        with st.expander("Розширені параметри"):
            a, b, c, d = st.columns(4)
            if numeric:
                a.markdown("Критерій: **МНК**")
            else:
                s["fitness"] = a.selectbox("Критерій", ["hybrid", "ssd"], index=0 if s["fitness"] == "hybrid" else 1,
                                           format_func=lambda x: {"hybrid": "Змішаний (як у коді)", "ssd": "МНК"}[x],
                                           key=f"emp_fit_{s['ver']}")
            s["ga"]["sol_per_pop"] = int(b.number_input("Розмір популяції", 10, 300, int(s["ga"]["sol_per_pop"]), 10,
                                                        key=f"emp_pop_{s['ver']}"))
            s["ga"]["random_seed"] = int(c.number_input("Seed", 0, 10 ** 6, int(s["ga"]["random_seed"]), 1,
                                                        key=f"emp_seed_{s['ver']}"))
            s["exact"] = d.checkbox("Побітово як legacy (повільніше)", s["exact"], key=f"emp_exact_{s['ver']}")
        run = c2.button("▶ Запустити", type="primary", width="stretch", key="emp_run")

    if run:
        cfg = build_cfg(s)
        bar = st.progress(0.0, text="Генетичний алгоритм…")

        def prog(g, total):
            if g % 5 == 0 or g == total:
                bar.progress(g / total, text=f"Покоління {g} / {total}")
        r = run_experiment(cfg, with_importance=False, progress=prog)
        bar.empty()
        s.update(cfg=cfg, result=r, model=model_io.EmpiricalModel.from_result(r, s["name"]), importance=None,
                 params_at_tune=dict(s["params"]))
        s["reached"] = max(s["reached"], S_REPORT)
        save_project(s, quiet=True)
        st.rerun()                               # щоб збереження дійшло до браузера

    r = s["result"]
    if r is None:
        ui.note("Натисніть «Запустити». Для 65 записів режим «Як у статті» займає приблизно 10–20 секунд.", "muted")
    else:
        _tune_results(s, r)

    c1, c2 = st.columns([1, 5])
    if c1.button("← Назад", key="emp_back3"):
        goto(S_MODEL)
    if c2.button("Далі: використати модель →", type="primary", key="emp_next3", disabled=s["model"] is None):
        goto(S_USE)


def _tune_results(s, r):
    N = len(r.raw)
    numeric = r.config.is_numeric
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if numeric:
            ui.kpi("R² (уся вибірка)", f"{r.after.r2:.3f}", f"було {r.before.r2:.3f}")
        else:
            ui.kpi("Правильні класи", f"{r.after.matches} / {N}",
                   f"було {r.before.matches} / {N} → {100 * r.after.matches / N:.1f}%")
    with k2:
        ui.kpi("Середня похибка MAD", f"{r.after.mad:.4f}", f"було {r.before.mad:.4f} (у [0, 1])")
    with k3:
        if numeric:
            mae = float(np.mean(np.abs(reports.output_physical(r, r.after.pred) - r.raw[:, -1])))
            ui.kpi("Похибка в одиницях", f"{mae:.4g}", "середня абсолютна")
        else:
            ui.kpi("Налаштовано параметрів", f"{int(np.count_nonzero(r.W_initial))} + {len(r.tuning.thresholds)}",
                   "ваги + межі класів")
    with k4:
        if r.config.train_fraction >= 1.0:
            ui.kpi("Перевірка", "на тих самих даних", "відкладіть 20% у кроці 2", text=True)
        else:
            n_te = len(r.test_idx)
            if numeric:
                ui.kpi("Відкладені дані", f"R² {r.test_after.r2:.3f}", f"{n_te} записів")
            else:
                ui.kpi("Перевірка на відкладених", f"{r.test_after.matches} / {n_te}",
                       f"{100 * r.test_after.matches / n_te:.1f}%")

    g1, g2 = st.columns([1.3, 1], gap="large")
    with g1, st.container(border=True):
        st.markdown("**Збіжність (MSE найкращого розв’язку)**")
        if r.tuning.history_mse:
            st.altair_chart(ui.line_chart(r.tuning.history_mse), width="stretch")
        else:
            st.caption("Історію збіжності не збережено.")
        if not numeric:
            t0 = " · ".join(f"{_conv(s, t):.2f}" for t in r.config.initial_thresholds)
            t1 = " · ".join(f"{_conv(s, t):.2f}" for t in r.tuning.thresholds)
            st.markdown(f"Межі класів (шкала {sc.LABELS[s['scale']]}): <span class='mono' style='color:#5A6573'>{t0}"
                        f"</span> → <span class='mono'><b>{t1}</b></span>", unsafe_allow_html=True)
    with g2, st.container(border=True):
        if numeric:
            st.markdown("**Реальне ↔ прогноз (одиниці виходу)**")
            st.altair_chart(ui.scatter_chart(r.raw[:, -1], reports.output_physical(r, r.after.pred)), width="stretch")
        else:
            st.markdown("**Реальний ↓ / обчислений →**")
            which = st.segmented_control("Стан", ["До", "Після"], default="Після", key="emp_cm_which",
                                         label_visibility="collapsed")
            M = r.before.confusion if which == "До" else r.after.confusion
            ui.confusion(M, _short(r.config.class_labels))
    with st.expander("Таблиця прогнозів по кожному запису"):
        st.dataframe(reports.table6_comparison(r), hide_index=True, width="stretch")


def _set_inputs_from_row(s, m):
    idx = st.session_state.get("emp_row")
    if idx is None or s["df"] is None:
        return
    row = s["df"].iloc[int(idx)]
    for i, cid in enumerate(m.concept_ids[:-1]):
        st.session_state[f"emp_in_{i}"] = float(row[cid])


def _result_text(m, v):
    """Головний підпис результату для нормованого виходу v ∈ [0, 1]."""
    if m.is_numeric:
        unit = m.units[-1] if m.units and m.units[-1] else ""
        return f"{float(m.physical(v)):.4g} {unit}".strip()
    from core import metrics
    c = int(metrics.classify([v], m.thresholds)[0])
    return m.class_labels[c - 1]


def _conv_task(s, m) -> conv_view.Task:
    """Дані для вкладки «Збіжність»: усі рядки навчальних даних (або випадкові вектори, якщо даних немає)."""
    n = len(m.concept_ids)
    labels = [f"C{i + 1}" for i in range(n)]
    r = s["result"]
    if r is None:                                   # файл «лише модель»: даних немає
        rng = np.random.default_rng(0)
        A0 = np.hstack([rng.random((100, n - 1)), np.zeros((100, 1))])
        return conv_view.Task("emp_conv", np.asarray(m.W, float), A0, labels, [f"Вектор {i + 1}" for i in range(100)],
                              "empirical", cv.UNIT, [n - 1], list(range(n - 1)), m.iterations,
                              "100 випадкових векторів U(0, 1) (даних у проєкті немає)")
    which = "after"
    if r.W_initial is not None:
        which = st.radio("Ваги для перевірки", ["after", "before"], horizontal=True, key="emp_conv_w",
                         format_func=lambda x: "налаштовані ГА" if x == "after" else "початкові (з даних)")
    W = r.tuning.W if which == "after" else r.W_initial
    thr = None if r.config.is_numeric else (r.tuning.thresholds if which == "after" else r.config.initial_thresholds)
    A0 = r.A_norm.copy()
    A0[:, -1] = 0.0
    short = _short(m.class_labels) if not m.is_numeric else []
    cases = [f"№ {i + 1}" + (f" · {m.class_labels[c - 1]}" if short else "") for i, c in enumerate(r.real_class)]
    return conv_view.Task("emp_conv", np.asarray(W, float), A0, labels, cases, "empirical", cv.UNIT, [n - 1],
                          list(range(n - 1)), r.config.iterations,
                          f"{len(A0)} рядків даних (нормовані показники, вихід = 0)",
                          real_output=r.A_norm[:, -1], real_class=None if r.config.is_numeric else r.real_class,
                          thresholds=thr)


def step_use(s):
    m: model_io.EmpiricalModel = s["model"]
    S_ = s["scale"]
    ui.header("Крок 5. Використання", "Введіть показники — результат оновлюється одразу.", m.name, S_)
    short = _short(m.class_labels) if not m.is_numeric else []
    t1, t2, t3, t4 = st.tabs(["Один випадок", "Пакетно (CSV)", "Вплив показників", "Збіжність"])

    with t1:
        left, right = st.columns([1.1, 1], gap="large")
        with left, st.container(border=True):
            st.markdown("**Показники**")
            if s["df"] is not None:
                st.selectbox("Взяти рядок з даних", list(range(len(s["df"]))), index=None, key="emp_row",
                             placeholder="оберіть номер рядка…", format_func=lambda i: f"№ {i + 1}",
                             on_change=_set_inputs_from_row, args=(s, m))
            cols = st.columns(2)
            x = []
            for i, cid in enumerate(m.concept_ids[:-1]):
                lo, hi = m.x_min[i], m.x_max[i]
                key = f"emp_in_{i}"
                if key not in st.session_state:
                    st.session_state[key] = float(np.round((lo + hi) / 2, 3))
                unit = f", {m.units[i]}" if m.units[i] else ""
                with cols[i % 2]:
                    x.append(st.number_input(f"C{i + 1} · {m.concept_names[i]}{unit}", key=key,
                                             help=f"Діапазон у навчальних даних: {lo:g} – {hi:g}", format="%.3f"))
        val, _ = m.predict([x])
        v = float(val[0])
        with right:
            with st.container(border=True):
                st.markdown("<span style='font-size:13px;color:#5A6573;text-transform:uppercase;letter-spacing:.04em'>"
                            "Результат моделі</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='display:flex;flex-wrap:wrap;align-items:baseline;gap:12px'>"
                            f"<span style='font-size:28px;font-weight:700'>{_result_text(m, v)}</span>"
                            f"<span class='mono' style='color:#4A5563'>вихід = {_conv(s, v):.3f}</span></div>",
                            unsafe_allow_html=True)
                if m.is_numeric:
                    ui.value_bar(v, S_)
                else:
                    ui.class_scale(v, m.thresholds, short, S_)
                    near = [t for t in m.thresholds if abs(v - t) < 0.03]
                    if near:
                        ui.note(f"Значення близько до межі між класами ({_conv(s, near[0]):.3f}). "
                                "Результат варто перевірити.", "warn")
                out_of_range = [m.concept_names[i] for i, xi in enumerate(x) if xi < m.x_min[i] or xi > m.x_max[i]]
                if out_of_range:
                    ui.note("Поза діапазоном навчальних даних: " + ", ".join(out_of_range), "warn")
            with st.container(border=True):
                st.markdown("**Що-якщо**")
                i = st.selectbox("Показник", list(range(m.n_inputs)), key="emp_wi_i",
                                 format_func=lambda k: f"C{k + 1} · {m.concept_names[k]}")
                lo, hi = float(m.x_min[i]), float(m.x_max[i])
                new = st.slider("Нове значення", lo, hi, float(min(max(x[i], lo), hi)), key=f"emp_wi_v_{i}")
                x2 = list(x)
                x2[i] = new
                v2 = float(m.predict([x2])[0][0])
                a, b = _result_text(m, v), _result_text(m, v2)
                change = "результат не змінився" if a == b else f"{a} → <b>{b}</b>"
                st.markdown(f"<span class='mono'>{x[i]:g} → {new:g}</span>: вихід {_conv(s, v):.3f} → "
                            f"<b>{_conv(s, v2):.3f}</b>; {change}", unsafe_allow_html=True)

    with t2:
        st.markdown("Завантажте CSV з тими самими стовпцями показників — модель обчислить результат для кожного рядка.")
        st.caption("Потрібні стовпці: " + ", ".join(m.concept_ids[:-1]))
        f = st.file_uploader("CSV для пакетної обробки", type=["csv"], key="emp_batch")
        if f is not None:
            df = pd.read_csv(io.BytesIO(f.getvalue()), sep=None, engine="python")
            missing = [c for c in m.concept_ids[:-1] if c not in df.columns]
            if missing:
                st.error("Немає стовпців: " + ", ".join(missing))
            else:
                vals, _ = m.predict(df[m.concept_ids[:-1]].to_numpy(float))
                res = df.copy()
                res[f"вихід {sc.LABELS[S_]}"] = np.round(_conv(s, vals), 4)
                res["результат"] = [_result_text(m, float(v)) for v in vals]
                st.dataframe(res, hide_index=True, width="stretch")
                st.download_button("Завантажити результат (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                                   "batch_results.csv", "text/csv")

    with t3:
        st.markdown("Ранг I(Cᵢ): нормований вихід, коли «увімкнено» лише показник Cᵢ (межі — перебір усіх комбінацій).")
        if s["importance"] is None and st.button("Обчислити ранги", key="emp_imp"):
            with st.spinner(f"Перебір 2^{m.n_inputs} комбінацій…"):
                s["importance"] = imp_mod.concept_importance(np.asarray(m.W), -1, m.iterations, exact=False)
        res = s["importance"]
        if res is not None:
            order = np.argsort(-res.values)
            st.altair_chart(ui.hbar_chart([f"C{res.inputs[k] + 1} · {m.concept_names[res.inputs[k]]}" for k in order],
                                          _conv(s, res.values[order]), domain=[sc.lo(S_), sc.hi(S_)]), width="stretch")

    with t4:
        conv_view.render(_conv_task(s, m))

    ui.note("Дослідницька модель. Не є справжнім висновком і не замінює рішення фахівця.", "muted")
    c1, c2 = st.columns([1, 5])
    if not s["model_only"] and c1.button("← Назад", key="emp_back4"):
        goto(S_TUNE)
    if c2.button("Далі: звіт →", type="primary", key="emp_next4"):
        goto(S_REPORT)


def step_report(s):
    m: model_io.EmpiricalModel = s["model"]
    ui.header("Крок 6. Звіт", "Таблиці та графіки для тексту, файл проєкту для колег.", m.name, s["scale"])
    r = s["result"]
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Файл проєкту (.json)**  \nУсі кроки, дані й результати. Відкривається на стартовому екрані.")
        if s["model_only"]:
            st.download_button("Завантажити модель", model_io.to_json(m).encode("utf-8"),
                               f"{_slug(m.name)}.json", "application/json", type="primary")
        else:
            st.download_button("Завантажити проєкт", projects.to_bytes(to_doc(s)), f"{_slug(m.name)}.json",
                               "application/json", type="primary")
    with c2, st.container(border=True):
        st.markdown("**Таблиці та графіки (.zip)**  \nТабл. 3–7, ваги до/після, збіжність, PNG-графіки.")
        if r is None:
            st.caption("Доступно після налаштування на кроці 4.")
        else:
            st.download_button("Завантажити архів", _zip_reports(r), f"{_slug(m.name)}_results.zip", "application/zip")
    if r is not None:
        st.markdown("#### Ваги до і після налаштування")
        st.dataframe(reports.table5_weights(r), hide_index=True, width="stretch", height=320)
        t7 = reports.table7_confusion(r)
        if t7 is not None:
            st.markdown("#### Матриця класів (до / після)")
            st.dataframe(t7, width="stretch")
    if st.button("← Назад", key="emp_back5"):
        goto(S_USE)


def _slug(name: str) -> str:
    keep = "".join(ch if ch.isalnum() else "_" for ch in name.strip())
    return keep.strip("_")[:40] or "model"


@st.cache_data(show_spinner="Готую архів…")
def _zip_bytes(_r, key):
    with tempfile.TemporaryDirectory() as d:
        reports.save_all(_r, d, plots=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(Path(d).iterdir()):
                z.write(p, p.name)
        return buf.getvalue()


def _zip_reports(r):
    import matplotlib
    matplotlib.use("Agg")
    return _zip_bytes(r, (id(r), r.config.scale))


# ---------------- Точка входу ----------------

def render():
    s = S()
    locked = tuple(range(S_USE)) if s["model_only"] else tuple(
        k for k in (S_USE, S_REPORT) if s["model"] is None)
    clicked = ui.stepper(STEPS, s["step"], s["reached"], "emp", locked=locked)
    if s["model_only"]:
        st.sidebar.caption("Відкрито файл лише з моделлю (без даних): кроки 1–4 недоступні.")
    else:
        ui.save_box(lambda: save_project(s), s.get("saved_at"), "emp")
    if clicked is not None and clicked != s["step"]:
        goto(clicked)
    step = s["step"]
    if step in (S_USE, S_REPORT) and s["model"] is None:
        step = S_TUNE
    [step_project, step_data, step_model, step_tune, step_use, step_report][step](s)
    ui.footer(medical=True)
