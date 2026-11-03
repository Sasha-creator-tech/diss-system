"""Вкладка «Збіжність»: порівняння правил виведення та порогових функцій (core/convergence.py).

Спільна для обох майстрів. Майстер передає ваги W, початкові вектори A0 і дані для оцінки якості.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import ui
from core import convergence as cv

EPS_OPTIONS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
SCALE_TXT = {cv.UNIT: "[0, 1]", cv.SYM: "[−1, 1]", None: "без меж"}
DEFAULT_TRACE = [("native", "none"), ("kosko", "sigmoid"), ("quasi", "sigmoid"), ("kosko", "tanh")]


@dataclass
class Task:
    """Що порівнювати (готує майстер)."""
    key: str                          # префікс ключів віджетів
    W: np.ndarray
    A0: np.ndarray                    # (N, n) початкові вектори — по одному на випадок
    labels: list[str]                 # підписи концептів
    case_labels: list[str]            # підписи випадків (рядків A0)
    kind: str                         # "empirical" або "gds"
    native_scale: str                 # cv.UNIT або cv.SYM — робоча шкала моделі
    out_idx: list[int]                # вихідні концепти (графік траєкторій)
    input_idx: list[int]              # входи / причини (для утримання)
    T: int                            # кроків за замовчуванням (як у моделі)
    source: str                       # пояснення, звідки початкові вектори
    real_output: np.ndarray | None = None
    real_class: np.ndarray | None = None
    thresholds: list[float] | None = None
    Y: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def has_quality(self) -> bool:
        return self.real_output is not None or self.Y is not None

    @property
    def numeric(self) -> bool:
        return not self.thresholds


@st.cache_data(show_spinner=False, max_entries=16)
def _run(kind, W, A0, variants, params, T, eps, clamp, real_output, real_class, thresholds, Y, out_idx):
    if kind == "empirical":
        scorer = None if real_output is None else cv.empirical_scorer(
            W, real_output, real_class, list(thresholds) if thresholds else None, out_idx[0])
    else:
        scorer = cv.gds_scorer(Y, out_idx)
    vs = [cv.Variant(r, a) for r, a in variants]
    return cv.compare(W, A0, vs, dict(params), T, eps, list(clamp) or None, scorer)


def _fmt_cond(c: cv.Condition | None) -> str:
    if c is None:
        return "—"
    mark = {True: " ✓", False: " ✗", None: ""}[c.holds]
    return f"{c.value}{mark}"


def _quality_cols(t: Task):
    """(ключ метрики, заголовок, формат, краще: max/min)."""
    if not t.has_quality:
        return []
    if t.kind == "gds":
        return [("MSE", "MSE", "%.4f", "min"), ("MAD", "MAD", "%.4f", "min")]
    if t.numeric:
        return [("R²", "R²", "%.3f", "max"), ("MAD", "MAD", "%.4f", "min")]
    return [("збіги", "Збіги", "%d", "max"), ("rs", "Спірмен", "%.3f", "max")]


def _table(t: Task, reps: list[cv.VariantReport]) -> pd.DataFrame:
    N = len(t.A0)
    rows = []
    for r in reps:
        d = r.dyn
        a = cv.ACTIVATIONS[r.variant.act]
        img = SCALE_TXT[a.image]
        if a.image is not None and a.image != t.native_scale:
            img += " ⚠"
        fx = d.status == cv.FIXED
        row = {
            "Правило": ("● " if r.variant.is_baseline else "") + r.label.split(" · ")[0],
            "Функція": r.label.split(" · ")[1],
            "Область f": img,
            "Динаміка": _dyn_text(d, N) + (" · одна для всіх ⚠" if d.unique_fixed_point else ""),
            "Кроків": float(np.mean(d.steps[fx])) if fx.any() else np.nan,
            "Умова збіжності": _fmt_cond(r.conds[0] if r.conds else None),
        }
        for k, lab, _, _ in _quality_cols(t):
            row[f"{lab} · з (11)" if t.kind == "empirical" else f"{lab} · з нормуванням"] = r.quality["з"].get(k)
            row[f"{lab} · без"] = r.quality["без"].get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def _dyn_text(d: cv.Dynamics, N: int) -> str:
    """Стисло: «✓ точка 8/8» або «точка 1 · цикл 7»."""
    if d.count(cv.FIXED) == N:
        return f"✓ точка {N}/{N}"
    parts = [(cv.FIXED, "точка"), (cv.CYCLE, "цикл"), (cv.NONCONV, "не збіглась"), (cv.DIVERGED, "розбіжність")]
    return " · ".join(f"{lab} {d.count(st_)}" for st_, lab in parts if d.count(st_))


PALETTE = ["#1F4FC9", "#C2620C", "#0F8B8D", "#7A4CC2", "#3B8B2E", "#B5325A", "#6B7480", "#B08A00"]


def _overlay_chart(df: pd.DataFrame, symlog: bool, height=320):
    y = alt.Y("значення:Q", scale=alt.Scale(type="symlog") if symlog else alt.Scale(zero=False))
    return (alt.Chart(df).mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=18))
            .encode(x=alt.X("крок:Q"), y=y,
                    color=alt.Color("варіант:N", sort=None, scale=alt.Scale(range=PALETTE),
                                    legend=alt.Legend(orient="bottom", columns=2, labelLimit=460, title=None)),
                    tooltip=["варіант", "крок", alt.Tooltip("значення:Q", format=".4f")])
            .properties(height=height))


def render(t: Task):
    k = t.key
    ui.note("Перевірка, чи доходить карта до стаціонарного стану з різними <b>правилами виведення</b> "
            "(огляд Nápoles et al., 2026: формули (1), (3), (5), (6)) і <b>пороговими функціями</b> "
            "(там само, табл. 1). Базовий варіант ● — як у моделі: рекурентна формула статей "
            "<b>без порогової функції</b>, а вихід нормується формулою (11). Це додаткова перевірка: "
            "результати й формули моделі вона не змінює.", "info")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        rules = c1.multiselect("Правила виведення", list(cv.RULES), default=list(cv.RULES), key=f"{k}_rules",
                               format_func=lambda r: cv.RULES[r].label)
        acts = c2.multiselect("Порогові функції", list(cv.ACTIVATIONS), default=list(cv.ACTIVATIONS),
                              key=f"{k}_acts", format_func=lambda a: cv.ACTIVATIONS[a].label)
        c1, c2, c3 = st.columns([1, 1, 2])
        T = int(c1.number_input("Кроків T", 5, 1000, int(t.T), 5, key=f"{k}_T",
                                help=f"Як у моделі — {t.T}. Вихід береться на кроці T."))
        eps = c2.selectbox("Точність ε", EPS_OPTIONS, index=2, key=f"{k}_eps", format_func=lambda e: f"{e:.0e}",
                           help="Стан вважається стаціонарним, коли max|A⁽ᵗ⁾ − A⁽ᵗ⁻¹⁾| < ε.")
        clamp_lbl = "Утримувати входи сталими" if t.kind == "empirical" else "Утримувати причини сталими"
        clamp = c3.checkbox(clamp_lbl, key=f"{k}_clamp",
                            help="На кожному кроці вхідні концепти повертаються до заданих значень. "
                                 "У моделях статей входи не утримуються.")
        with st.expander("Параметри функцій і правил"):
            cols = st.columns(3)
            params = {}
            for i, (pk, pv) in enumerate(cv.DEFAULT_PARAMS.items()):
                params[pk] = float(cols[i % 3].number_input(cv.PARAM_LABELS[pk], value=float(pv), step=0.1,
                                                            format="%.3f", key=f"{k}_p_{pk}"))
            st.caption("Обмеження з огляду: 0 < k₁, k₂ ≤ 1; 0 ≤ φ ≤ 1; λ > 0.")
        with st.expander("Формули правил і функцій"):
            a, b = st.columns(2)
            a.markdown("\n".join(f"- **{r.label}**: <span class='mono'>{r.formula}</span>" for r in cv.RULES.values()),
                       unsafe_allow_html=True)
            b.markdown("\n".join(f"- **{f.label}** ({SCALE_TXT[f.image]}): <span class='mono'>{f.formula}</span>"
                                  for f in cv.ACTIVATIONS.values()), unsafe_allow_html=True)
        st.caption(f"Початкові вектори: {t.source}.")

    variants = [(v.rule, v.act) for v in cv.all_variants(rules, acts)]
    clamp_idx = tuple(t.input_idx) if clamp else ()
    with st.spinner("Моделювання…"):
        reps = _run(t.kind, np.asarray(t.W, float), np.asarray(t.A0, float), tuple(variants),
                    tuple(sorted(params.items())), T, float(eps), clamp_idx, t.real_output, t.real_class,
                    None if t.numeric else tuple(t.thresholds), t.Y, tuple(t.out_idx))
    base = reps[0]
    N = len(t.A0)

    # ---- підсумок
    rho = cv.spectral_radius(t.W)
    q_cols = _quality_cols(t)
    c = st.columns(4)
    with c[0]:
        ui.kpi("Спектральний радіус ρ(W)", f"{rho:.3f}",
               "< 1: базова формула збігається" if rho < 1 else "≥ 1: без нормування значення ростуть")
    with c[1]:
        ok = sum(1 for r in reps if r.dyn.count(cv.FIXED) == N)
        ui.kpi("Збіглися для всіх випадків", f"{ok} / {len(reps)}", "варіантів (нерухома точка)")
    with c[2]:
        d = base.dyn
        st_txt = _dyn_text(d, N)
        ui.kpi("Базовий варіант ●", st_txt, f"за T = {T}, ε = {eps:.0e}", text=True)
    with c[3]:
        if q_cols:
            key, lab, fmt, better = q_cols[0]
            vals = [(r.quality["з"].get(key), r) for r in reps if r.quality["з"].get(key) is not None
                    and np.isfinite(r.quality["з"].get(key))]
            if vals:
                v, best = (max if better == "max" else min)(vals, key=lambda x: x[0])
                tail = f"/{N}" if key == "збіги" else ""
                ui.kpi(f"Найкраще: {lab}", f"{fmt % v}{tail}", best.label)
        else:
            ui.kpi("Якість", "—", "немає даних для оцінки")

    if not (rho < 1) and base.dyn.count(cv.FIXED) < N:
        ui.note(f"У базовому варіанті стаціонарний стан не досягається (ρ(W) = {rho:.3f} ≥ 1): значення концептів "
                "зростають з кожним кроком. Модель працює завдяки нормуванню (11), яке переводить вихід у [0, 1] "
                "відносно меж, отриманих тією самою процедурою. Саме про це речення статті: «нормування (11) "
                "потрібне через відсутність порогової функції».", "warn")
    uniq = [r.label for r in reps if r.dyn.unique_fixed_point]
    if uniq:
        ui.note(f"{len(uniq)} варіант(и) приводять <b>усі</b> випадки до однієї точки — така карта не розрізняє "
                "входи, і прогноз втрачає сенс (огляд, розд. 3). Найчастіше це класичне правило (1) із сигмоїдою "
                "λ = 1; допомагає більше λ, правило (6) з φ < 1 або утримання входів.", "warn")

    # ---- таблиця
    df = _table(t, reps)
    st.markdown("**Порівняння варіантів**")
    small = st.column_config.Column(width="small")
    cfg = {"Область f": small,
           "Кроків": st.column_config.Column(width="small", help="Середній крок, з якого |ΔA| < ε"),
           "Динаміка": st.column_config.Column(help="Скільки випадків дійшли до нерухомої точки, зациклились, не "
                                                    "збіглись за T кроків або розбіглись; «одна для всіх» — усі "
                                                    "випадки в одній точці, карта не розрізняє входи"),
           "Умова збіжності": st.column_config.Column(width="small", help="Головна теоретична умова; деталі — нижче")}
    show = df.copy()
    show["Кроків"] = [f"{v:.1f}" if np.isfinite(v) else "—" for v in df["Кроків"]]
    for key, lab, fmt, _ in q_cols:
        for col in df.columns:
            if col.startswith(lab + " ·"):
                cfg[col] = small
                show[col] = ["—" if v is None or not np.isfinite(v) else
                             (f"{int(v)}/{N}" if key == "збіги" else fmt % v) for v in df[col]]
    styled = show.style.apply(lambda r: ["background-color:#EEF3FE" if str(r["Правило"]).startswith("●") else ""
                                         for _ in r], axis=1)
    st.dataframe(styled, hide_index=True, width="stretch", column_config=cfg,
                 height=min(36 * len(df) + 40, 600))
    leg = ("Область f з ⚠ — область значень функції не збігається з робочою шкалою моделі "
           f"{SCALE_TXT[t.native_scale]}. ")
    if q_cols:
        leg += ("«з (11)» — вихід нормовано формулою (11), межі отримано тим самим варіантом; «без» — сире "
                "значення виходу на кроці T. " if t.kind == "empirical" else
                "«з нормуванням» — ділення на max|x| (як у моделі); «без» — сирі значення наслідків. ")
        if t.kind == "empirical" and not t.numeric:
            leg += (f"Збіги — з {N} за порогами моделі; Спірмен — рангова кореляція виходу з класом "
                    "(не залежить від порогів). ")
        leg += "«—» — якість не оцінюється (є розбіжні випадки або межі (11) збіглися)."
    st.caption(leg)
    st.download_button("Завантажити таблицю (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       "activation_check.csv", "text/csv", key=f"{k}_dl")

    with st.expander("Умови збіжності для варіанта"):
        i = st.selectbox("Варіант", range(len(reps)), key=f"{k}_cond_v", format_func=lambda j: reps[j].label)
        for cnd in reps[i].conds:
            mark = {True: "✓", False: "✗", None: "·"}[cnd.holds]
            st.markdown(f"{mark} **{cnd.name}** — <span class='mono'>{cnd.value}</span>: {cnd.meaning}",
                        unsafe_allow_html=True)
        st.caption("ρ(W) — спектральний радіус; ‖·‖₂ — спектральна норма, ‖·‖_F — норма Фробеніуса; L — стала "
                   "Ліпшиця функції (сигмоїда λ/4, tanh λ). Умова стиску достатня, але не необхідна.")

    # ---- траєкторії
    st.markdown("**Вихід у стаціонарний стан**")
    names = {j: r.label for j, r in enumerate(reps)}
    dflt = [j for j, r in enumerate(reps) if (r.variant.rule, r.variant.act) in DEFAULT_TRACE]
    c1, c2, c3 = st.columns([1.2, 1, 1], vertical_alignment="bottom")
    case = c1.selectbox("Випадок", range(N), key=f"{k}_case", format_func=lambda j: t.case_labels[j])
    if len(t.out_idx) > 1:
        oi = c2.selectbox("Концепт", t.out_idx, key=f"{k}_oi", format_func=lambda j: t.labels[j])
    else:
        oi = t.out_idx[0]
        c2.markdown(f"Концепт: **{t.labels[oi]}** (вихід)")
    symlog = c3.checkbox("Логарифмічна вісь", key=f"{k}_symlog", help="Зручно, коли якийсь варіант розбігається")
    chosen = st.multiselect("Варіанти на графіку", list(names), default=dflt or [0], key=f"{k}_trace",
                            format_func=lambda j: names[j], max_selections=len(PALETTE))
    p = dict(params)
    rows = []
    for j in chosen:
        H = cv.simulate(t.W, t.A0[case:case + 1], reps[j].variant, p, T, list(clamp_idx) or None)
        for step in range(T + 1):
            val = H[step, 0, oi]
            if np.isfinite(val):
                rows.append({"крок": step, "варіант": reps[j].label, "значення": float(val)})
    if rows:
        st.altair_chart(_overlay_chart(pd.DataFrame(rows), symlog), width="stretch")
    else:
        st.caption("Оберіть варіанти для графіка.")
    st.caption("Сирі значення концептів у робочій шкалі розрахунку, без нормування і без перетворення шкали "
               "проєкту: без порогової функції вони можуть виходити за межі шкали. Розбіжна траєкторія "
               "обривається на кроці, де значення перевищили 10⁶.")

    with st.expander("Усі концепти для одного варіанта"):
        j = st.selectbox("Варіант", range(len(reps)), key=f"{k}_all_v", format_func=lambda x: reps[x].label)
        H = cv.simulate(t.W, t.A0[case:case + 1], reps[j].variant, p, T, list(clamp_idx) or None)[:, 0, :]
        H = H[np.all(np.isfinite(H), axis=1)]
        st.altair_chart(ui.trajectories_chart(H, t.labels, t.out_idx), width="stretch")
