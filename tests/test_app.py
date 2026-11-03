"""Наскрізні тести інтерфейсу (Streamlit AppTest, без браузера)."""
from pathlib import Path

import numpy as np
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


@pytest.fixture(autouse=True)
def _store(monkeypatch):
    """Сховище в пам’яті замість localStorage (AppTest не виконує JavaScript)."""
    import sys
    sys.path.insert(0, str(Path(APP).parents[1]))
    from app import projects
    monkeypatch.setenv("DISSSYSTEM_STORAGE", "memory")
    projects._MEMORY.clear()
    return projects._MEMORY


def _app():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception
    return at


def _ihd_to_tune(at):
    at.button(key="ex_emp").click().run()
    at.session_state["emp"]["exact"] = False
    at.button(key="emp_next0").click().run()          # Проєкт → Дані
    at.button(key="emp_next1").click().run()          # Дані → Модель
    assert not at.exception and np.count_nonzero(at.session_state["emp"]["init"]) == 67
    at.button(key="emp_next2").click().run()          # Модель → Налаштування


@pytest.mark.slow
def test_empirical_wizard_ihd_and_reopen(_store):
    at = _app()
    _ihd_to_tune(at)
    at.button(key="emp_run").click().run()            # 400 поколінь, як у статті
    r = at.session_state["emp"]["result"]
    assert (r.before.matches, r.after.matches) == (39, 62)
    at.button(key="emp_next3").click().run()
    at.button(key="emp_next4").click().run()
    assert not at.exception and len(at.get("download_button")) == 2
    # новий сеанс: проєкт у списку, відкривається з результатами, доступні всі кроки
    assert len(_store) == 1
    at2 = _app()
    at2.button(key=f"open_{next(iter(_store))}").click().run()
    s = at2.session_state["emp"]
    assert s["result"].after.matches == 62 and s["step"] == 5
    for k in range(6):
        at2.button(key=f"emp_step{k}").click().run()
        assert not at2.exception


def test_empirical_numeric_output_short():
    at = _app()
    at.button(key="ex_emp").click().run()
    at.radio[1].set_value("numeric").run()
    at.session_state["emp"]["ga"]["num_generations"] = 15
    at.button(key="emp_next0").click().run()
    at.button(key="emp_next1").click().run()
    at.button(key="emp_next2").click().run()
    at.button(key="emp_run").click().run()
    r = at.session_state["emp"]["result"]
    assert r.config.is_numeric and r.after.classes is None and r.after.r2 > 0
    at.button(key="emp_next3").click().run()
    assert not at.exception


def test_expert_wizard_transport_diagnosis_unit_scale():
    at = _app()
    at.button(key="ex_gds").click().run()
    at.radio[0].set_value("unit").run()               # шкала [0, 1]
    at.button(key="gds_next0").click().run()
    at.button(key="gds_next1").click().run()
    assert np.count_nonzero(at.session_state["gds"]["W_expert"]) == 18
    at.button(key="gds_skip2").click().run()          # без налаштування: ваги як у статтях 1 і 4
    assert not at.exception and at.session_state["gds"]["step"] == 4
    assert at.slider[0].value == pytest.approx(0.8)   # причина 0.6 у [-1, 1] → 0.8 у [0, 1]
    at.radio(key="gds_dmode").set_value("Перевірка: задати причини, відновити їх за наслідками").run()
    at.button(key="gds_diag_go").click().run()
    r = at.session_state["gds"]["diag"]["r"]
    ref = at.session_state["gds"]["diag"]["ref"]
    assert round(float(np.mean(np.abs(r.causes - ref))), 4) == 0.1047   # = legacy S1 з seed 42


@pytest.mark.slow
def test_expert_tuning_short_and_reopen(_store):
    at = _app()
    at.button(key="ex_gds").click().run()
    at.button(key="gds_next0").click().run()
    at.button(key="gds_next1").click().run()
    at.button(key="gds_next2").click().run()
    at.session_state["gds"]["ga"]["num_generations"] = 20
    at.button(key="gds_run").click().run()
    r = at.session_state["gds"]["tuning"]
    assert not at.exception and r.mse_after <= r.mse_before
    at2 = _app()
    at2.button(key=f"open_{next(iter(_store))}").click().run()
    assert at2.session_state["gds"]["tuning"].mse_after == pytest.approx(r.mse_after)
