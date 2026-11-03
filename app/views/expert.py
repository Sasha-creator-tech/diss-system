"""Майстер «Експертна модель» (УДС): Проєкт → Концепти → Зв’язки → Налаштування → Використання → Звіт.

Розрахунок ведеться у шкалі [-1, 1] (статті 1, 2, 4); шкала проєкту визначає лише, як значення
вводяться й показуються (core.scale, лінійне перетворення).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app import projects, ui
from core import diagnosis, gds, model_io, weights_expert
from core import scale as sc
from core.ga import GAConfig
from core.tuning import mimo_result, tune_mimo

ROOT = Path(__file__).resolve().parents[2]
TRANSPORT = ROOT / "examples" / "transport" / "config.yaml"
NATIVE = sc.SYM

CAUSE, EFFECT = "Причина (вхід)", "Наслідок (вихід)"
M_SAATY, M_NUM = "Ранги Сааті 1–9 (метод найменшого впливу)", "Числові ваги від −1 до 1"
STEPS = [("Проєкт", "назва, шкала"), ("Концепти", "фактори"), ("Зв’язки", "оцінки впливів"),
         ("Налаштування", "ГА, необов’язково"), ("Використання", "прогноз, пошук причин"), ("Звіт", "експорт")]
S_PROJECT, S_CONCEPTS, S_LINKS, S_TUNE, S_USE, S_REPORT = range(6)
PRESETS = {"Швидко · 100 поколінь": 100, "Як у статті · 400": 400, "Ретельно · 1000": 1000}
DEFAULT_GA = dict(num_generations=400, num_parents_mating=10, sol_per_pop=50, parent_selection_type="rws",
                  keep_parents=2, crossover_type="single_point", mutation_type="random",
                  mutation_percent_genes=10, random_seed=42)


def S() -> dict:
    return st.session_state["gds"]


def _fresh(name: str) -> dict:
    return {"id": projects.new_id(), "name": name, "scale": sc.SYM, "step": 0, "reached": 0, "ver": 0,
            "concepts": pd.DataFrame({"ID": ["C1", "C2", "C3"], "Назва": ["Фактор 1", "Фактор 2", "Результат"],
                                      "Роль": [CAUSE, CAUSE, EFFECT]}),
            "links": pd.DataFrame({"Від": ["C1", "C2"], "До": ["C3", "C3"], "Знак": ["+", "−"],
                                   "Ранг Сааті": [3, 1], "Вага": [0.5, -0.2]}),
            "method": M_SAATY, "variation": 0.2, "train": None, "ga": dict(DEFAULT_GA), "population_seed": 42,
            "tuning": None, "tuned": None, "use_tuned": True, "W_expert": None, "model": None, "model_only": False,
            "forecast_causes": None, "scenarios": {}, "diag": None, "diag_n_runs": 50, "diag_seed": 42,
            "saved_at": None}


def _activate(state):
    st.session_state["gds"] = state
    st.session_state["mode"] = "gds"
    st.rerun()


def new_project():
    _activate(_fresh("Нова експертна модель"))


def load_example():
    cfg = gds.load_gds_config(TRANSPORT)
    s = _fresh("Надійність автопідприємства")
    s["concepts"] = pd.DataFrame({"ID": cfg.concept_ids, "Назва": cfg.concept_names,
                                  "Роль": [CAUSE if i in cfg.cause_idx else EFFECT for i in range(cfg.n)]})
    ranks = {(l.src, l.tgt): l.rank for l in weights_expert.ranks_from_matrix(cfg.W)}
    rows = [{"Від": cfg.concept_ids[i], "До": cfg.concept_ids[j], "Знак": "+" if cfg.W[i, j] > 0 else "−",
             "Ранг Сааті": ranks[(i, j)], "Вага": float(cfg.W[i, j])}
            for i in range(cfg.n) for j in range(cfg.n) if cfg.W[i, j] != 0]
    s["links"] = pd.DataFrame(rows)
    s["method"] = M_NUM                      # як у статтях 1 і 4; ранги Сааті — стаття 2
    s["train"] = gds.load_training(cfg).drop(columns=["No", "label"], errors="ignore")
    s["forecast_causes"] = list(cfg.forecast["causes"])
    s["scenarios"] = {k: list(v) for k, v in cfg.diagnosis["scenarios"].items()}
    _activate(s)


def open_model(m: model_io.GDSModel):
    """Файл «лише модель»: перетворюємо на проєкт із числовими вагами — доступні всі кроки."""
    s = _fresh(m.name)
    ids = m.concept_ids
    s["concepts"] = pd.DataFrame({"ID": ids, "Назва": m.concept_names,
                                  "Роль": [CAUSE if i in m.cause_idx else EFFECT for i in range(len(ids))]})
    W = np.asarray(m.W)
    s["links"] = pd.DataFrame([{"Від": ids[i], "До": ids[j], "Знак": "+" if W[i, j] > 0 else "−", "Ранг Сааті": 1,
                                "Вага": float(W[i, j])} for i in range(len(ids)) for j in range(len(ids)) if W[i, j]])
    s.update(method=M_NUM, scale=getattr(m, "project_scale", sc.SYM), step=S_USE, reached=S_REPORT,
             scenarios=m.meta.get("scenarios", {}), forecast_causes=m.meta.get("forecast_causes"))
    s["W_expert"] = build_W(s)[0]
    _activate(s)


# ---------------- Збереження ----------------

def to_doc(s) -> dict:
    state = {k: s[k] for k in ("name", "scale", "step", "reached", "method", "variation", "ga", "population_seed",
                               "use_tuned", "forecast_causes", "scenarios", "diag_n_runs", "diag_seed")}
    state.update(concepts=s["concepts"].to_dict("records"), links=s["links"].to_dict("records"),
                 train=None if s["train"] is None else s["train"].to_dict("records"), tuned=s["tuned"])
    ids, _, ci, ei = _concepts(s)
    r = s["tuning"]
    summary = {"type": "Експертна", "concepts": f"{len(ci)} → {len(ei)}",
               "status": "Налаштовано" if r is not None else ("Оцінки задано" if s["W_expert"] is not None and
                                                              np.count_nonzero(s["W_expert"]) else "Чернетка"),
               "quality": f"MSE {r.mse_after:.3f}" if r is not None else "—"}
    return projects.make_doc(s["id"], "gds", s["name"], s["scale"], summary, state)


def from_doc(doc: dict):
    d = doc["state"]
    s = _fresh(d.get("name", doc.get("name", "Проєкт")))
    s["id"] = doc["id"]
    for k in ("scale", "step", "reached", "method", "variation", "ga", "population_seed", "use_tuned",
              "forecast_causes", "scenarios", "diag_n_runs", "diag_seed"):
        if d.get(k) is not None:
            s[k] = d[k]
    s["concepts"] = pd.DataFrame(d["concepts"])
    s["links"] = pd.DataFrame(d["links"])
    if d.get("train"):
        s["train"] = pd.DataFrame(d["train"])
    s["saved_at"] = doc.get("saved_at")
    W, _, errs = build_W(s)
    s["W_expert"] = None if errs else W
    if d.get("tuned") and not errs:
        t = d["tuned"]
        _, _, ci, ei = _concepts(s)
        s["tuned"] = t
        s["structure"] = t["structure"]
        s["tuning"] = mimo_result(t["structure"], t["genes"], t["history"], t["X"], t["Y"], ci, ei, len(W))
    _activate(s)


def save_project(s, quiet: bool = False) -> bool:
    if s["model_only"]:
        return False
    projects.save(to_doc(s))
    s["saved_at"] = datetime.now().isoformat(timespec="seconds")
    return True


def goto(step: int):
    s = S()
    s["step"] = step
    s["reached"] = max(s["reached"], step)
    s["ver"] += 1
    save_project(s, quiet=True)
    st.rerun()


# ---------------- Модель з таблиць ----------------

def _to_view(s, x):
    return sc.convert(x, NATIVE, s["scale"])


def _to_native(s, x):
    return sc.convert(x, s["scale"], NATIVE)


def _concepts(s):
    c = s["concepts"]
    ids = [str(x) for x in c["ID"]]
    names = [str(x) for x in c["Назва"]]
    cause_idx = [i for i, r in enumerate(c["Роль"]) if r == CAUSE]
    effect_idx = [i for i, r in enumerate(c["Роль"]) if r == EFFECT]
    return ids, names, cause_idx, effect_idx


def _validate_concepts(s) -> list[str]:
    c = s["concepts"]
    errs = []
    ids = [str(x).strip() for x in c["ID"]]
    if any(not x or x in ("nan", "None") for x in ids):
        errs.append("У кожного концепту має бути ID.")
    if len(set(ids)) != len(ids):
        errs.append("ID концептів мають бути унікальними.")
    if not (c["Роль"] == CAUSE).any() or not (c["Роль"] == EFFECT).any():
        errs.append("Потрібна хоча б одна причина і хоча б один наслідок.")
    return errs


def build_W(s):
    """→ (W, таблиця розрахунку Сааті або None, помилки)."""
    ids, _, _, _ = _concepts(s)
    n = len(ids)
    L = s["links"].dropna(subset=["Від", "До"])
    errs = []
    W = np.zeros((n, n))
    table = None
    links = []
    for _, r in L.iterrows():
        if r["Від"] not in ids or r["До"] not in ids:
            errs.append(f"Невідомий концепт у зв’язку {r['Від']} → {r['До']}.")
            continue
        i, j = ids.index(r["Від"]), ids.index(r["До"])
        if i == j:
            errs.append(f"Зв’язок {r['Від']} сам на себе не дозволено.")
            continue
        if s["method"] == M_SAATY:
            try:
                rank = int(r["Ранг Сааті"])
            except (TypeError, ValueError):
                errs.append(f"Ранг для {r['Від']} → {r['До']} має бути числом 1–9.")
                continue
            if not 1 <= rank <= 9:
                errs.append(f"Ранг для {r['Від']} → {r['До']} поза межами 1–9.")
                continue
            links.append(weights_expert.Link(i, j, 1 if str(r["Знак"]).strip() in ("+", "1") else -1, rank))
        else:
            try:
                w = float(r["Вага"])
            except (TypeError, ValueError):
                errs.append(f"Вага для {r['Від']} → {r['До']} має бути числом.")
                continue
            if not -1 <= w <= 1:
                errs.append(f"Вага для {r['Від']} → {r['До']} поза межами [−1, 1].")
            W[i, j] = w
    if s["method"] == M_SAATY and links:
        W, table = weights_expert.weights_from_ranks(links, n)
    return W, table, errs


def current_model(s) -> model_io.GDSModel:
    ids, names, cause_idx, effect_idx = _concepts(s)
    W = s["W_expert"]
    tuned = s["tuning"] is not None and s["use_tuned"]
    if tuned:
        W = s["tuning"].W_after
    return model_io.GDSModel(
        name=s["name"], concept_ids=ids, concept_names=names, cause_idx=cause_idx, effect_idx=effect_idx,
        W=np.asarray(W).tolist(), project_scale=s["scale"],
        meta={"weights": "налаштовані ГА" if tuned else ("Сааті" if s["method"] == M_SAATY else "експертні"),
              "scenarios": s["scenarios"], "forecast_causes": s["forecast_causes"]})


# ---------------- Кроки ----------------

def step_project(s):
    ui.header("Крок 1. Проєкт", "Назва та шкала значень концептів.", s["name"], s["scale"])
    with st.container(border=True):
        s["name"] = st.text_input("Назва проєкту", s["name"], key=f"gds_name_{s['ver']}")
        st.markdown("**Шкала значень концептів**")
        s["scale"] = st.radio("Шкала", [sc.SYM, sc.UNIT], index=[sc.SYM, sc.UNIT].index(s["scale"]), horizontal=True,
                              key=f"gds_scale_{s['ver']}", label_visibility="collapsed",
                              format_func=lambda x: {sc.SYM: "[−1, 1] — від −1 через 0 (середнє) до +1",
                                                     sc.UNIT: "[0, 1] — від мінімуму 0 до максимуму 1"}[x])
        st.caption("Шкала визначає, як вводяться і показуються значення причин, наслідків і сценаріїв. Розрахунок НКК "
                   "ведеться у [−1, 1], як у статтях; перетворення лінійне — результати не змінюються. "
                   "Ваги зв’язків завжди від −1 до 1: знак означає напрям впливу.")
    if st.button("Далі: концепти →", type="primary", key="gds_next0"):
        goto(S_CONCEPTS)


def step_concepts(s):
    ui.header("Крок 2. Концепти", "Перелічіть фактори системи і вкажіть, які з них — причини (керовані входи), "
              "а які — наслідки (показники якості).", s["name"], s["scale"])
    ed = st.data_editor(s["concepts"], key=f"gds_concepts_{s['ver']}", num_rows="dynamic", hide_index=True,
                        width="stretch",
                        column_config={"ID": st.column_config.TextColumn(required=True, help="Коротке позначення, напр. C1"),
                                       "Назва": st.column_config.TextColumn(),
                                       "Роль": st.column_config.SelectboxColumn(options=[CAUSE, EFFECT], required=True)})
    s["concepts"] = ed.reset_index(drop=True)
    lo, hi = sc.lo(s["scale"]), sc.hi(s["scale"])
    ui.note(f"Значення концептів задаються на шкалі {sc.LABELS[s['scale']]}: {lo:g} — мінімум, "
            f"{(lo + hi) / 2:g} — середнє, {hi:g} — максимум.", "muted")
    errs = _validate_concepts(s)
    for e in errs:
        st.error(e)
    c1, c2 = st.columns([1, 5])
    if c1.button("← Назад", key="gds_back1"):
        goto(S_PROJECT)
    if c2.button("Далі: зв’язки →", type="primary", disabled=bool(errs), key="gds_next1"):
        goto(S_LINKS)


def step_links(s):
    ui.header("Крок 3. Зв’язки", "Опишіть, хто на кого впливає. Найпростіше — порівняти кожен вплив з найслабшим "
              "у своїй групі за шкалою Сааті.", s["name"], s["scale"])
    ids, names, cause_idx, effect_idx = _concepts(s)
    s["method"] = st.radio("Як задаєте сили впливу", [M_SAATY, M_NUM], index=0 if s["method"] == M_SAATY else 1,
                           key=f"gds_method_{s['ver']}", horizontal=True)
    if s["method"] == M_SAATY:
        ui.note("Для кожної групи (посилюючі «+» і послаблюючі «−») найслабший вплив має ранг 1. Інші — у скільки "
                "разів сильніші: 3 — трохи, 5 — помітно, 7 — значно, 9 — абсолютно.", "info")
    ed = st.data_editor(
        s["links"], key=f"gds_links_{s['ver']}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Від": st.column_config.SelectboxColumn(options=ids, required=True),
            "До": st.column_config.SelectboxColumn(options=ids, required=True),
            "Знак": st.column_config.SelectboxColumn(options=["+", "−"], required=True, help="+ посилює, − послаблює"),
            "Ранг Сааті": st.column_config.NumberColumn(min_value=1, max_value=9, step=1,
                                                       disabled=s["method"] != M_SAATY),
            "Вага": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.01, format="%.2f",
                                                 disabled=s["method"] != M_NUM),
        })
    s["links"] = ed.reset_index(drop=True)
    W, table, errs = build_W(s)
    for e in errs:
        st.error(e)
    if s["W_expert"] is not None and not np.array_equal(s["W_expert"], W):
        s["tuning"] = s["tuned"] = None            # ваги змінились — налаштування застаріло
    s["W_expert"] = W

    kinds = ["input" if i in cause_idx else "effect" for i in range(len(ids))]
    g, t = st.columns([1.4, 1], gap="large")
    with g, st.container(border=True):
        st.markdown("**Карта впливів**")
        if np.count_nonzero(W):
            st.graphviz_chart(ui.graph_dot(W, ids, kinds), width="stretch")
            st.caption(" · ".join(f"{a} — {b}" for a, b in zip(ids, names)))
        else:
            st.caption("Додайте зв’язки в таблицю вище.")
    with t, st.container(border=True):
        st.markdown("**Підсумкові ваги**")
        if table is not None:
            st.dataframe(table[["Дуга", "Знак", "Ранг Сааті", "Оцінка", "w (норм.)"]], hide_index=True, width="stretch")
            pos = table[table["Знак"] == "+"]["Σ рангів групи"]
            st.caption(f"w_min = 1 / Σ рангів{f' = 1/{int(pos.iloc[0])}' if len(pos) else ''}; далі w = w_min·r і ділення "
                       f"на найсильніший у групі.")
        else:
            rows = [{"Дуга": f"{ids[i]} → {ids[j]}", "Вага": round(W[i, j], 3)}
                    for i in range(len(ids)) for j in range(len(ids)) if W[i, j] != 0]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    c1, c2, c3 = st.columns([1, 2.2, 3])
    if c1.button("← Назад", key="gds_back2"):
        goto(S_CONCEPTS)
    ok = not errs and np.count_nonzero(W) > 0
    if c2.button("Далі: налаштування →", type="primary", key="gds_next2", disabled=not ok):
        goto(S_TUNE)
    if c3.button("Пропустити налаштування → використання", key="gds_skip2", disabled=not ok):
        s["reached"] = S_REPORT
        goto(S_USE)


def step_tune(s):
    ui.header("Крок 4. Налаштування (необов’язково)", "Якщо є сценарії «входи → очікувані виходи» (від експертів або "
              "спостережень), ГА уточнить ваги в межах заданої свободи від початкових.", s["name"], s["scale"])
    ids, names, cause_idx, effect_idx = _concepts(s)
    if s["train"] is None or [c for c in ids if c not in s["train"].columns]:
        s["train"] = pd.DataFrame([[0.0] * len(ids)], columns=ids)
    st.markdown(f"#### Навчальні сценарії (шкала {sc.LABELS[s['scale']]})")
    view = s["train"][ids].astype(float).apply(lambda col: _to_view(s, col.to_numpy()))
    lo, hi = sc.lo(s["scale"]), sc.hi(s["scale"])
    ed = st.data_editor(view, key=f"gds_train_{s['ver']}", num_rows="dynamic", hide_index=True, width="stretch",
                        column_config={c: st.column_config.NumberColumn(min_value=lo, max_value=hi, step=0.1,
                                                                        format="%.2f") for c in ids})
    native = ed.reset_index(drop=True).astype(float).apply(lambda col: _to_native(s, col.to_numpy()))
    if not np.allclose(native.to_numpy(), s["train"][ids].astype(float).to_numpy(), equal_nan=True) \
            or len(native) != len(s["train"]):
        s["train"] = native

    with st.container(border=True):
        c1, c2, c3 = st.columns([2.4, 1, 1], vertical_alignment="bottom")
        cur = next((k for k, v in PRESETS.items() if v == s["ga"]["num_generations"]), None)
        mode = c1.segmented_control("Режим", list(PRESETS), default=cur, key=f"gds_preset_{s['ver']}")
        if mode:
            s["ga"]["num_generations"] = PRESETS[mode]
        s["variation"] = c2.number_input("Свобода ваг, ±", 0.05, 0.5, float(s["variation"]), 0.05, key=f"gds_var_{s['ver']}")
        run = c3.button("▶ Запустити", type="primary", width="stretch", key="gds_run")

    if run:
        tr = s["train"][ids].dropna()
        X, Y = tr[[ids[i] for i in cause_idx]].to_numpy(float), tr[[ids[i] for i in effect_idx]].to_numpy(float)
        structure = weights_expert.weight_intervals(s["W_expert"], s["variation"])
        bar = st.progress(0.0, text="Генетичний алгоритм…")

        def prog(g, total):
            if g % 5 == 0 or g == total:
                bar.progress(g / total, text=f"Покоління {g} / {total}")
        res = tune_mimo(structure, X, Y, cause_idx, effect_idx, len(ids), GAConfig(**s["ga"]),
                        population_seed=s["population_seed"], progress=prog)
        bar.empty()
        s["tuning"], s["structure"], s["use_tuned"] = res, structure, True
        s["tuned"] = {"structure": structure, "genes": [float(g) for g in res.genes_after],
                      "history": [float(h) for h in res.history_mse], "X": X.tolist(), "Y": Y.tolist()}
        s["reached"] = max(s["reached"], S_REPORT)
        save_project(s, quiet=True)
        st.rerun()                               # щоб збереження дійшло до браузера

    r = s["tuning"]
    if r is not None:
        k1, k2, k3 = st.columns(3)
        with k1:
            ui.kpi("MSE по виходах", f"{r.mse_after:.4f}", f"було {r.mse_before:.4f} (у [−1, 1])")
        with k2:
            ui.kpi("MAD по виходах", f"{r.mad_after:.4f}", f"було {r.mad_before:.4f}")
        with k3:
            ui.kpi("Налаштовано ваг", f"{len(s['structure'])}", f"сценаріїв: {len(r.sse_before)}")
        g1, g2 = st.columns([1.2, 1], gap="large")
        with g1, st.container(border=True):
            st.markdown("**Збіжність (MSE найкращого розв’язку)**")
            st.altair_chart(ui.line_chart(r.history_mse), width="stretch")
        with g2, st.container(border=True):
            st.markdown("**Ваги до / після**")
            st.dataframe(pd.DataFrame([{"Дуга": f"{ids[c['src']]} → {ids[c['tgt']]}", "До": c["init"],
                                        "Межі": f"[{c['min']}, {c['max']}]", "Після": round(g, 4)}
                                       for c, g in zip(s["structure"], r.genes_after)]),
                         hide_index=True, width="stretch", height=300)
        s["use_tuned"] = st.toggle("Використовувати налаштовані ваги далі", s["use_tuned"], key=f"gds_use_{s['ver']}")
    else:
        ui.note("Для прикладу автопідприємства — 8 сценаріїв від «найгіршого» до «найкращого»; 400 поколінь "
                "займають близько 30–60 секунд.", "muted")

    c1, c2 = st.columns([1, 5])
    if c1.button("← Назад", key="gds_back3"):
        goto(S_LINKS)
    if c2.button("Далі: використати модель →", type="primary", key="gds_next3"):
        s["reached"] = S_REPORT
        goto(S_USE)


def step_use(s):
    m = current_model(s)
    s["model"] = m
    ids, names = m.concept_ids, m.concept_names
    V = lambda x: _to_view(s, x)                 # noqa: E731
    lo, hi = sc.lo(s["scale"]), sc.hi(s["scale"])
    ui.header("Крок 5. Використання", f"Ваги: {m.meta.get('weights', 'експертні')}. Шкала значень "
              f"{sc.LABELS[s['scale']]}.", m.name, s["scale"])
    tab_f, tab_d = st.tabs(["Прогноз (прямий вивід)", "Пошук причин (зворотний вивід)"])
    step = 0.05

    with tab_f:
        left, right = st.columns([1, 1.6], gap="large")
        with left, st.container(border=True):
            st.markdown("**Причини**")
            defaults = s["forecast_causes"] or [0.0] * len(m.cause_idx)
            causes_view = [st.slider(f"{ids[i]} · {names[i]}", lo, hi, float(np.round(V(defaults[k]), 3)), step,
                                     key=f"gds_fc_{s['scale']}_{k}") for k, i in enumerate(m.cause_idx)]
            causes = [float(_to_native(s, c)) for c in causes_view]
            s["forecast_causes"] = causes
        f = m.forecast_run(causes)
        with right:
            with st.container(border=True):
                st.markdown(f"**Наслідки у стаціонарному стані** (крок {f.steps}, нормовано на max|x|)")
                st.altair_chart(ui.hbar_chart([f"{ids[i]} · {names[i]}" for i in m.effect_idx],
                                              V(f.normalized[m.effect_idx]), domain=[lo, hi]), width="stretch")
            with st.container(border=True):
                st.markdown("**Динаміка концептів по кроках**")
                st.altair_chart(ui.trajectories_chart(V(f.history), ids, m.effect_idx), width="stretch")

    with tab_d:
        mode = st.radio("Режим", ["Реальна діагностика: причини невідомі",
                                  "Перевірка: задати причини, відновити їх за наслідками"],
                        key="gds_dmode", horizontal=True)
        check = mode.startswith("Перевірка")
        left, right = st.columns([1, 1.5], gap="large")
        with left, st.container(border=True):
            ref = None
            if check:
                st.markdown("**Еталонні причини**")
                if s["scenarios"]:
                    sc_name = st.selectbox("Сценарій", list(s["scenarios"]), key="gds_sc")
                    base = s["scenarios"][sc_name]
                else:
                    sc_name, base = "custom", [0.0] * len(m.cause_idx)
                ref_view = [st.slider(f"{ids[i]} · {names[i]}", lo, hi, float(np.round(V(base[k]), 3)), step,
                                      key=f"gds_ref_{s['scale']}_{sc_name}_{k}") for k, i in enumerate(m.cause_idx)]
                ref = np.array([float(_to_native(s, v)) for v in ref_view])
                target = m.effects(ref)
                st.markdown("<span style='font-size:13px'>Наслідки за моделлю:</span> <span class='mono' "
                            "style='font-size:13px'>" + "  ".join(f"{ids[i]}={float(V(target[k])):+.3f}"
                                                                  for k, i in enumerate(m.effect_idx)) + "</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("**Спостережувані наслідки**")
                obs_view = [st.slider(f"{ids[i]} · {names[i]}", lo, hi, float((lo + hi) / 2), 0.01,
                                      key=f"gds_obs_{s['scale']}_{k}") for k, i in enumerate(m.effect_idx)]
                target = np.array([float(_to_native(s, v)) for v in obs_view])
            a, b = st.columns(2)
            s["diag_n_runs"] = a.number_input("Запусків ГА", 5, 200, int(s["diag_n_runs"]), 5, key="gds_nruns")
            s["diag_seed"] = b.number_input("Seed", 0, 10 ** 6, int(s["diag_seed"]), 1, key="gds_dseed")
            go = st.button("Знайти причини", type="primary", width="stretch", key="gds_diag_go")
        if go:
            with st.spinner(f"{s['diag_n_runs']} запусків ГА…"):
                r = diagnosis.backward_inference(np.asarray(m.W), target, m.cause_idx, m.effect_idx,
                                                 n_runs=int(s["diag_n_runs"]), reference=ref, seed=int(s["diag_seed"]),
                                                 max_iter=m.diagnosis.get("max_iter", 50),
                                                 eps=m.diagnosis.get("eps", 1e-3))
            s["diag"] = {"r": r, "ref": ref, "target": target}
        with right, st.container(border=True):
            d = s["diag"]
            if d is None:
                st.caption("Задайте наслідки й натисніть «Знайти причини».")
            else:
                r = d["r"]
                st.markdown("**Відновлені причини x\\***" + (" · риска — еталон" if d["ref"] is not None else ""))
                st.altair_chart(ui.hbar_chart([f"{ids[i]} · {names[i]}" for i in m.cause_idx], V(r.causes),
                                              domain=[lo, hi], ref=None if d["ref"] is None else V(d["ref"])),
                                width="stretch")
                k1, k2 = st.columns(2)
                with k1:
                    ui.kpi("Нев’язка наслідків", f"{r.error:.2e}", "Σ(модель − задані)², шкала [−1, 1]")
                with k2:
                    if d["ref"] is not None:
                        ui.kpi("MAD до еталона", f"{np.mean(np.abs(r.causes - d['ref'])):.4f}",
                               f"MSE {np.mean((r.causes - d['ref']) ** 2):.4f} (шкала [−1, 1])")
                    else:
                        ui.kpi("Найкращий запуск", f"№ {r.best_run + 1}", f"з {len(r.run_errors)}")
                order = np.argsort(r.causes)
                worst = [f"{ids[m.cause_idx[k]]} ({names[m.cause_idx[k]]})" for k in order[:2] if r.causes[k] < -0.2]
                if worst:
                    ui.note("Найнижчі причини: " + ", ".join(worst) + ".", "info")

    c1, c2 = st.columns([1, 5])
    if c1.button("← Назад", key="gds_back4"):
        goto(S_TUNE)
    if c2.button("Далі: звіт →", type="primary", key="gds_next4"):
        goto(S_REPORT)


def step_report(s):
    m = current_model(s)
    ui.header("Крок 6. Звіт", "Збережіть проєкт і результати.", m.name, s["scale"])
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Файл проєкту (.json)**  \nУсі кроки та результати. Відкривається на стартовому екрані.")
        st.download_button("Завантажити проєкт", projects.to_bytes(to_doc(s)), "expert_project.json",
                           "application/json", type="primary")
    with c2, st.container(border=True):
        st.markdown("**Матриця ваг (.csv)**")
        W = pd.DataFrame(m.W, index=m.concept_ids, columns=m.concept_ids)
        st.download_button("Завантажити матрицю", W.to_csv().encode("utf-8-sig"), "weights.csv", "text/csv")
    st.markdown("#### Матриця впливів W")
    st.dataframe(W.round(3), width="stretch")
    if s["diag"] is not None:
        r = s["diag"]["r"]
        st.markdown(f"#### Останній зворотний вивід (шкала {sc.LABELS[s['scale']]})")
        st.dataframe(pd.DataFrame({"Причина": [m.concept_ids[i] for i in m.cause_idx],
                                   "x*": np.round(_to_view(s, r.causes), 4)}), hide_index=True)
    if st.button("← Назад", key="gds_back5"):
        goto(S_USE)


def render():
    s = S()
    ready = s["W_expert"] is not None and np.count_nonzero(s["W_expert"]) > 0
    locked = () if ready else (S_TUNE, S_USE, S_REPORT)
    clicked = ui.stepper(STEPS, s["step"], s["reached"], "gds", locked=locked)
    ui.save_box(lambda: save_project(s), s.get("saved_at"), "gds")
    if clicked is not None and clicked != s["step"]:
        goto(clicked)
    step = s["step"]
    if step >= S_TUNE and not ready:
        step = S_LINKS
    [step_project, step_concepts, step_links, step_tune, step_use, step_report][step](s)
    ui.footer()
