"""Спільний вигляд застосунку: стилі, шапка, кроки майстра, картки, графіки."""
from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BLUE = "#1F4FC9"
ORANGE = "#C2620C"
INK = "#18212C"
MUTED = "#5A6573"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea, select {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.block-container { padding-top: 3.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E2E5EA; }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; border-radius: 10px; min-height: 48px; }
[data-testid="stSidebar"] .stButton button > div { justify-content: flex-start; width: 100%; }
[data-testid="stSidebar"] .stButton button p { text-align: left; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
.ds-brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:18px; margin-bottom: 4px; }
.ds-logo { width:28px; height:28px; border-radius:8px; background:#1F4FC9; display:inline-flex; align-items:center; justify-content:center; }
.ds-head { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin: 0 0 6px; }
.ds-title { font-size: 28px; font-weight: 700; margin: 0 0 4px; line-height: 1.25; }
.ds-sub { color:#5A6573; font-size: 15px; margin: 0 0 16px; }
.ds-badge { font-size:12px; padding:4px 8px; border-radius:6px; font-weight:500; }
.ds-badge.unit { background:#EAF0FE; color:#1F4FC9; }
.ds-badge.sym { background:#FDF0E3; color:#8A4A0A; }
.ds-card { background:#FFFFFF; border:1px solid #E2E5EA; border-radius:12px; padding:16px 18px; }
.ds-kpi .l { font-size:13px; color:#5A6573; }
.ds-kpi .v { font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:500; }
.ds-kpi .v.txt { font-family:'IBM Plex Sans',sans-serif; font-size:18px; font-weight:600; padding: 6px 0 4px; }
.ds-kpi .s { font-size:13px; color:#5A6573; }
.ds-note { padding:10px 12px; border-radius:8px; font-size:14px; line-height:1.45; margin: 4px 0 10px; }
.ds-note.warn { background:#FFF6E5; color:#6B4A06; }
.ds-note.info { background:#F5F8FF; color:#2D3A4B; }
.ds-note.muted { background:#EEF1F5; color:#3C4653; }
.ds-tpl { display:flex; flex-direction:column; gap:8px; min-height: 150px; }
.ds-tpl .t { font-size:19px; font-weight:600; }
.ds-tpl .d { font-size:14px; color:#4A5563; line-height:1.5; }
.ds-chip { display:inline-block; font-size:12px; padding:4px 8px; border-radius:6px; background:#F1F3F6; color:#3C4653; margin: 2px 4px 0 0; }
.ds-scale { position:relative; height:44px; margin-top:6px; }
.ds-scale .bar { display:flex; height:16px; margin-top:20px; border-radius:8px; overflow:hidden; }
.ds-scale .mk { position:absolute; top:0; transform:translateX(-50%); display:flex; flex-direction:column; align-items:center; }
.ds-scale .mk .tri { width:0; height:0; border-left:7px solid transparent; border-right:7px solid transparent; border-top:10px solid #18212C; }
.ds-scale .mk .ln { width:2px; height:28px; background:#18212C; }
.ds-scale-l { display:flex; font-size:11px; color:#4A5563; font-family:'IBM Plex Mono',monospace; }
.ds-cm { display:grid; gap:4px; font-family:'IBM Plex Mono',monospace; font-size:13px; text-align:center; }
.ds-cm span { padding:7px 0; border-radius:6px; }
.ds-foot { margin-top: 28px; padding-top: 12px; border-top:1px solid #E2E5EA; font-size:13px; color:#5A6573; }
</style>
"""

LOGO = ('<span class="ds-logo"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" '
        'stroke-width="2" stroke-linecap="round"><circle cx="6" cy="7" r="2.5"/><circle cx="18" cy="7" r="2.5"/>'
        '<circle cx="12" cy="18" r="2.5"/><path d="M8.3 8.2 10.6 15.8M15.7 8.2 13.4 15.8M8.5 7h7"/></svg></span>')


def setup_page():
    st.set_page_config(page_title="DissSystem", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.sidebar.markdown(f'<div class="ds-brand">{LOGO}DissSystem</div>'
                        f'<div style="font-size:13px;color:{MUTED};margin-bottom:14px">Моделювання на нечітких когнітивних картах</div>',
                        unsafe_allow_html=True)


def header(title: str, sub: str = "", project: str | None = None, scale: str | None = None):
    """Заголовок кроку. `scale` не показується (вибирається на кроці «Проєкт»), параметр лишено для сумісності."""
    proj = f'<span style="color:{MUTED};font-size:14px">{html.escape(project)}</span>' if project else ""
    st.markdown(f'<div class="ds-head">{proj}</div><div class="ds-title">{html.escape(title)}</div>'
                + (f'<div class="ds-sub">{html.escape(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def note(text: str, kind: str = "info"):
    st.markdown(f'<div class="ds-note {kind}">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", text: bool = False):
    st.markdown(f'<div class="ds-card ds-kpi"><div class="l">{label}</div><div class="v{" txt" if text else ""}">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def footer(medical: bool = False):
    st.markdown(f'<div class="ds-foot">Дослідницький інструмент.  Дані обробляються локально на вашому комп’ютері.</div>',
                unsafe_allow_html=True)


def stepper(steps: list[tuple[str, str]], current: int, reached: int, key: str, locked=()) -> int | None:
    """Кроки майстра в боковій панелі. Повертає номер натиснутого кроку або None."""
    st.sidebar.markdown(f'<div style="font-size:12px;color:{MUTED};text-transform:uppercase;letter-spacing:.05em;'
                        f'margin:6px 0 6px">Кроки</div>', unsafe_allow_html=True)
    clicked = None
    for i, (name, hint) in enumerate(steps):
        done = i < reached and i != current
        mark = "✓" if done else str(i + 1)
        label = f"{mark}   {name}  ·  {hint}"
        if st.sidebar.button(label, key=f"{key}_step{i}", width="stretch",
                             type="primary" if i == current else "secondary", disabled=i > reached or i in locked):
            clicked = i
    return clicked


def save_box(save_fn, saved_at: str | None, key: str):
    """Кнопка збереження проєкту в боковій панелі (сховище — localStorage браузера)."""
    st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.sidebar.button("Зберегти проєкт", key=f"{key}_save", width="stretch", icon=":material/save:"):
        if save_fn():
            st.session_state["_saved_msg"] = True
            st.rerun()
    if st.session_state.pop("_saved_msg", False):
        st.sidebar.success("Збережено у вашому браузері")
    when = saved_at.replace("T", " ")[:16] if saved_at else "ще не збережено"
    st.sidebar.caption(f"Останнє збереження: {when}. Проєкт також зберігається автоматично при переході між кроками.")


# ---------------- Графіки ----------------

def graph_dot(W: np.ndarray, labels: list[str], node_kind: list[str] | None = None,
              min_abs: float = 0.0, only_to: int | None = None, max_edges: int | None = None,
              center: int | None = None) -> str:
    """DOT-граф НКК з компактним круговим розміщенням (neato, фіксовані координати).

    Вузли — по колу, `center` (напр. вихід моделі за даними) — у центрі. Синій — посилює (+),
    помаранчевий пунктир — послаблює (−), товщина = |w|. Розміщення не залежить від кількості дуг.
    """
    import math
    n = W.shape[0]
    kinds = node_kind or [""] * n
    edges = [(i, j, W[i, j]) for i in range(n) for j in range(n)
             if W[i, j] != 0 and abs(W[i, j]) >= min_abs and (only_to is None or j == only_to)]
    edges.sort(key=lambda e: -abs(e[2]))
    if max_edges:
        edges = edges[:max_edges]
    ring = [i for i in range(n) if i != center]
    R = max(1.6, 0.36 * len(ring))                     # радіус кола (дюйми): ~0.36" на вузол
    pos = {}
    for k, i in enumerate(ring):
        a = math.pi / 2 - 2 * math.pi * k / max(1, len(ring))
        pos[i] = (R * math.cos(a), R * math.sin(a))
    if center is not None:
        pos[center] = (0.0, 0.0)
    lines = ['digraph G {', 'graph [layout=neato, bgcolor="transparent", pad=0.3, splines=curved, outputorder=edgesfirst];',
             'node [shape=circle, style=filled, fontname="IBM Plex Sans", fontsize=11, width=0.5, fixedsize=true, '
             'color="#9DB4EE", fillcolor="#FFFFFF", penwidth=1.6];',
             'edge [arrowsize=0.55];']
    for i in range(n):
        extra = ""
        if kinds[i] == "output" or i == center:
            extra = ', fillcolor="#FDE7D0", color="#C2620C", width=0.7'
        elif kinds[i] == "effect":
            extra = ', fillcolor="#FDF0E3", color="#E0A56B"'
        x, y = pos[i]
        lines.append(f'n{i} [label="{labels[i]}", pos="{x:.3f},{y:.3f}!"{extra}];')
    for i, j, w in edges:
        col = BLUE if w > 0 else ORANGE
        style = "solid" if w > 0 else "dashed"
        lines.append(f'n{i} -> n{j} [color="{col}", style={style}, penwidth={0.8 + abs(w) * 4:.2f}, '
                     f'tooltip="{labels[i]} → {labels[j]}: {w:+.3f}"];')
    lines.append("}")
    return "\n".join(lines)


def line_chart(y, x_title="покоління", y_title="MSE", height=240):
    df = pd.DataFrame({"x": np.arange(len(y)), "y": np.asarray(y, float)})
    return (alt.Chart(df).mark_line(color=BLUE, strokeWidth=2.5)
            .encode(x=alt.X("x:Q", title=x_title), y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=False)))
            .properties(height=height))


def trajectories_chart(history: np.ndarray, labels: list[str], effect_idx: list[int], height=300):
    rows = []
    for k in range(history.shape[0]):
        for i, lab in enumerate(labels):
            rows.append({"крок": k, "концепт": lab, "значення": history[k, i],
                         "група": "наслідок" if i in effect_idx else "причина"})
    df = pd.DataFrame(rows)
    return (alt.Chart(df).mark_line(strokeWidth=2)
            .encode(x="крок:Q", y=alt.Y("значення:Q"), color=alt.Color("концепт:N", sort=list(labels), legend=alt.Legend(orient="right")),
                    strokeDash=alt.StrokeDash("група:N", legend=None), tooltip=["концепт", "крок", alt.Tooltip("значення:Q", format=".3f")])
            .properties(height=height))


def hbar_chart(names, values, domain=None, height=None, color=BLUE, ref=None):
    df = pd.DataFrame({"name": names, "value": np.asarray(values, float)})
    if ref is not None:
        df["ref"] = np.asarray(ref, float)
    height = height or 34 * len(names) + 30
    x = alt.X("value:Q", title=None, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    base = alt.Chart(df).encode(y=alt.Y("name:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12)))
    chart = base.mark_bar(color=color, cornerRadius=3, height=16).encode(
        x=x, tooltip=["name", alt.Tooltip("value:Q", format=".3f")])
    pos = base.transform_filter("datum.value >= 0").mark_text(align="left", dx=4, fontSize=12, font="IBM Plex Mono").encode(
        x="value:Q", text=alt.Text("value:Q", format="+.3f"))
    neg = base.transform_filter("datum.value < 0").mark_text(align="right", dx=-4, fontSize=12, font="IBM Plex Mono").encode(
        x="value:Q", text=alt.Text("value:Q", format="+.3f"))
    layers = [chart, pos, neg]
    if ref is not None:
        layers.append(base.mark_tick(color=INK, thickness=3, size=26).encode(x="ref:Q"))
    return alt.layer(*layers).properties(height=height)


def class_scale(value01: float, thresholds, labels_short: list[str], scale: str = "unit"):
    """Шкала виходу з межами класів і маркером значення. Значення та межі — у [0, 1], підписи — у шкалі проєкту."""
    from core import scale as sc
    t = [0.0] + list(thresholds) + [1.0]
    widths = [max(0.0, (t[k + 1] - t[k]) * 100) for k in range(len(t) - 1)]
    pal = ["#CFDAF7", "#B3C5F2", "#97B0EC", "#F6D2AE", "#EDB57F", "#E39851", "#D97B2C", "#C8661A"]
    segs = "".join(f'<div style="width:{w:.2f}%;background:{pal[k % len(pal)]};'
                   f'{"border-left:2px solid #FFF" if k else ""}"></div>' for k, w in enumerate(widths))
    labs = "".join(f'<span style="width:{w:.2f}%;overflow:hidden">{html.escape(labels_short[k])}</span>'
                   for k, w in enumerate(widths))
    pos = min(max(value01, 0.0), 1.0) * 100
    ends = f'<div class="ds-scale-l" style="justify-content:space-between;color:#8A94A3"><span>{sc.lo(scale):g}</span><span>{sc.hi(scale):g}</span></div>'
    st.markdown(f'<div class="ds-scale"><div class="bar">{segs}</div>'
                f'<div class="mk" style="left:{pos:.2f}%"><span class="tri"></span><span class="ln"></span></div></div>'
                f'<div class="ds-scale-l">{labs}</div>{ends}', unsafe_allow_html=True)


def value_bar(value01: float, scale: str = "unit"):
    """Шкала без класів (числовий вихід): маркер на відрізку від мінімуму до максимуму."""
    from core import scale as sc
    pos = min(max(value01, 0.0), 1.0) * 100
    st.markdown(f'<div class="ds-scale"><div class="bar" style="background:linear-gradient(90deg,#CFDAF7,#97B0EC 50%,#E39851)"></div>'
                f'<div class="mk" style="left:{pos:.2f}%"><span class="tri"></span><span class="ln"></span></div></div>'
                f'<div class="ds-scale-l" style="justify-content:space-between"><span>{sc.lo(scale):g} (мін.)</span>'
                f'<span>{sc.hi(scale):g} (макс.)</span></div>', unsafe_allow_html=True)


def scatter_chart(real, pred, x_title="реальне", y_title="прогноз", height=280):
    df = pd.DataFrame({"real": np.asarray(real, float), "pred": np.asarray(pred, float)})
    lo, hi = float(min(df.min())), float(max(df.max()))
    line = pd.DataFrame({"real": [lo, hi], "pred": [lo, hi]})
    pts = alt.Chart(df).mark_circle(size=55, color=BLUE, opacity=0.75).encode(
        x=alt.X("real:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("pred:Q", title=y_title, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("real:Q", format=".3f"), alt.Tooltip("pred:Q", format=".3f")])
    diag = alt.Chart(line).mark_line(color="#A3ABB6", strokeDash=[4, 4]).encode(x="real:Q", y="pred:Q")
    return (diag + pts).properties(height=height)


def confusion(M: np.ndarray, labels_short: list[str]):
    m = len(labels_short)
    vmax = max(1, M.max())
    cells = ['<span></span>'] + [f'<span style="color:{MUTED}">{html.escape(l)}</span>' for l in labels_short]
    for i in range(m):
        cells.append(f'<span style="color:{MUTED};text-align:left">{html.escape(labels_short[i])}</span>')
        for j in range(m):
            v = int(M[i, j])
            if v == 0:
                cells.append('<span style="background:#F6F7F9;color:#A3ABB6">0</span>')
            elif i == j:
                a = 0.45 + 0.55 * v / vmax
                cells.append(f'<span style="background:rgba(31,79,201,{a:.2f});color:#FFF">{v}</span>')
            else:
                cells.append(f'<span style="background:#FDE7D0;color:#8A4A0A">{v}</span>')
    st.markdown(f'<div class="ds-cm" style="grid-template-columns:repeat({m + 1},minmax(0,1fr))">{"".join(cells)}</div>',
                unsafe_allow_html=True)
