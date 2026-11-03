"""Стартовий екран: шаблони, приклади, «Мої проєкти» (локальна папка projects/), імпорт файлу."""
from __future__ import annotations

import json

import streamlit as st

from app import projects, ui
from app.views import empirical, expert
from core import model_io
from core import scale as sc

KIND_LABEL = {"empirical": "За даними", "gds": "Експертна"}


def open_project(doc: dict):
    (empirical if doc["kind"] == "empirical" else expert).from_doc(doc)


def _import(raw: bytes):
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        st.error(f"Не вдалося прочитати файл: {e}")
        return
    if projects.is_project(doc):
        if doc.get("id") in {p["id"] for p in projects.list_projects()}:
            doc["id"] = projects.new_id()          # не перезаписуємо наявний проєкт
        projects.save(doc)
        open_project(doc)
    else:
        try:
            m = model_io.from_json(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            st.error(f"Це не файл проєкту DissSystem: {e}")
            return
        (empirical if m.kind == "empirical" else expert).open_model(m)


@st.dialog("Видалити проєкт?")
def _confirm_delete(pid: str, name: str):
    st.markdown(f"Проєкт **{name}** буде видалено з цього браузера. Якщо потрібна копія — спершу завантажте файл.")
    a, b = st.columns(2)
    if a.button("Видалити", type="primary", width="stretch", key="dlg_del"):
        projects.delete(pid)
        st.rerun()
    if b.button("Скасувати", width="stretch", key="dlg_cancel"):
        st.rerun()


COLS = [3.0, 1.3, 1.3, 1.0, 1.3, 1.2, 1.5, 0.45, 0.45]


def _projects_table():
    if not projects.ready():
        st.caption("Завантаження проєктів із браузера…")
        return
    items = projects.list_projects()
    if not items:
        st.markdown("<div style='color:#5A6573;padding:6px 2px'>Тут з’являться ваші проєкти. Створіть новий або "
                    "відкрийте приклад — він збережеться автоматично.</div>", unsafe_allow_html=True)
        return
    head = st.columns(COLS, vertical_alignment="center")
    for col, t in zip(head, ["Назва", "Тип", "Шкала", "Концепти", "Стан", "Якість", "Змінено", "", ""]):
        col.markdown(f"<span style='font-size:12px;color:#5A6573;text-transform:uppercase;letter-spacing:.04em'>{t}</span>",
                     unsafe_allow_html=True)
    for p in items:
        sm = p.get("summary") or {}
        c = st.columns(COLS, vertical_alignment="center")
        if c[0].button(p["name"] or "Без назви", key=f"open_{p['id']}", type="tertiary", help="Відкрити проєкт"):
            open_project(projects.load(p["id"]))
        c[1].markdown(KIND_LABEL.get(p["kind"], p["kind"]))
        c[2].markdown(f"<span class='mono'>{sc.LABELS.get(p.get('scale'), '—')}</span>", unsafe_allow_html=True)
        c[3].markdown(f"<span class='mono'>{sm.get('concepts', '—')}</span>", unsafe_allow_html=True)
        status = sm.get("status", "—")
        color = ("#E6F2EC", "#1C6B45") if status == "Налаштовано" else ("#F1F3F6", "#3C4653")
        c[4].markdown(f"<span style='font-size:12px;padding:4px 8px;border-radius:999px;background:{color[0]};"
                      f"color:{color[1]};font-weight:500'>{status}</span>", unsafe_allow_html=True)
        c[5].markdown(f"<span class='mono'>{sm.get('quality', '—')}</span>", unsafe_allow_html=True)
        c[6].markdown(f"<span style='font-size:13px;color:#5A6573'>{(p.get('saved_at') or '').replace('T', ' ')[:16]}"
                      f"</span>", unsafe_allow_html=True)
        c[7].download_button("", projects.to_bytes(projects.load(p["id"])), f"{p['name'] or 'project'}.json",
                             "application/json", key=f"dl_{p['id']}", icon=":material/download:", type="tertiary",
                             help="Завантажити файл проєкту")
        if c[8].button("", key=f"del_{p['id']}", icon=":material/delete:", type="tertiary", help="Видалити"):
            _confirm_delete(p["id"], p["name"] or "Без назви")


def render():
    ui.header("З чого почнемо?", "Оберіть потрібний інструмент відповідно до вихідної інформації.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown('<div class="ds-tpl"><span class="t">Модель за даними</span><span class="d">Є таблиця '
                    'спостережень (CSV): «показники → результат». Результат — класи або число. Система знаходить '
                    'зв’язки, побудує НКК і налаштує її генетичним алгоритмом.</span>'
                    '<div><span class="ds-chip">Приклад: діагностика ІХС</span></div></div>', unsafe_allow_html=True)
        if st.button("Почати з таблиці", key="new_emp", type="primary", width="stretch"):
            empirical.new_project()
        if st.button("Відкрити приклад ІХС", key="ex_emp", width="stretch"):
            empirical.load_example()
    with c2, st.container(border=True):
        st.markdown('<div class="ds-tpl"><span class="t">Експертна модель</span><span class="d">Даних немає, є '
                    'знання фахівця. Ви описуєте фактори і порівнюєте сили впливу за шкалою Сааті 1–9.</span>'
                    '<div><span class="ds-chip">Приклад: автопідприємство</span></div></div>', unsafe_allow_html=True)
        if st.button("Почати з експертних оцінок", key="new_gds", type="primary", width="stretch"):
            expert.new_project()
        if st.button("Відкрити приклад автопідприємства", key="ex_gds", width="stretch"):
            expert.load_example()
    with c3, st.container(border=True):
        st.markdown('<div class="ds-tpl"><span class="t">Відкрити файл проєкту</span><span class="d">Файл .json, '
                    'збережений на кроці «Звіт». Проєкт додасться до списку нижче.</span>'
                    '</div>', unsafe_allow_html=True)
        f = st.file_uploader("Файл проєкту", type=["json"], key="open_file", label_visibility="collapsed")
        if f is not None and st.session_state.get("_imported") != (f.name, f.size):
            st.session_state["_imported"] = (f.name, f.size)
            _import(f.getvalue())

    st.markdown("### Мої проєкти")
    st.caption("Зберігаються у вашому браузері на цьому комп’ютері (localStorage). На іншому комп’ютері чи в "
               "іншому браузері їх не буде — для перенесення завантажте файл проєкту.")
    if projects.last_error():
        st.error(f"Браузер не зміг зберегти проєкт: {projects.last_error()}. Імовірно, переповнене сховище — "
                 "завантажте великі проєкти файлом і видаліть їх зі списку.")
    with st.container(border=True):
        _projects_table()
    ui.footer(medical=True)
