"""DissSystem — локальний інтерфейс (Streamlit).

Запуск із кореня проєкту:
    streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import projects, ui  # noqa: E402
from app.views import empirical, expert, start  # noqa: E402


def main():
    ui.setup_page()
    ui.brand()
    projects.sync()
    mode = st.session_state.get("mode", "start")
    if mode != "start":
        if st.sidebar.button("← Усі проєкти", key="nav_home", width="stretch"):
            st.session_state["mode"] = "start"
            st.rerun()
    {"start": start.render, "emp": empirical.render, "gds": expert.render}[mode]()


main()
