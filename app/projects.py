"""Сховище проєктів у браузері користувача (localStorage).

Кожен проєкт — JSON-рядок у localStorage під ключем «dissystem:project:<id>». Тож проєкти лежать на
комп’ютері того, хто відкрив застосунок, — і локально, і на хостингу; сервер нічого не зберігає.

Обмін із браузером — через невидимий компонент (st.components.v2): Python ставить у чергу операції
запису/видалення, JavaScript виконує їх і повертає актуальний вміст сховища. Після кожного збереження
сторінка перезапускається (st.rerun), щоб черга дійшла до браузера.

Для тестів (AppTest не виконує JavaScript): DISSSYSTEM_STORAGE=memory — сховище в пам’яті процесу.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import streamlit as st

FORMAT = "dissystem-project"
VERSION = 1
PREFIX = "dissystem:project:"

_MEMORY: dict[str, str] = {}          # бекенд для тестів

_JS = """
export default function(component) {
  const { data, setStateValue } = component;
  const PREFIX = "%s";
  const w = window;
  w.__dsApplied = w.__dsApplied || 0;
  let error = null;
  for (const op of ((data && data.ops) || [])) {
    if (op.id <= w.__dsApplied) continue;
    try {
      if (op.value === null) localStorage.removeItem(PREFIX + op.key);
      else localStorage.setItem(PREFIX + op.key, op.value);
    } catch (e) { error = String(e); }
    w.__dsApplied = Math.max(w.__dsApplied, op.id);
  }
  const items = {};
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (k && k.startsWith(PREFIX)) items[k.slice(PREFIX.length)] = localStorage.getItem(k);
  }
  const payload = JSON.stringify({ items, applied: w.__dsApplied, error });
  if (payload !== w.__dsLast) {
    w.__dsLast = payload;
    setStateValue("store", { items, applied: w.__dsApplied, error, t: Date.now() });
  }
}
""" % PREFIX


def _backend() -> str:
    return os.environ.get("DISSSYSTEM_STORAGE", "browser")


@st.cache_resource
def _component():
    return st.components.v2.component("dissystem_storage", js=_JS)


def _ops() -> list[dict]:
    return st.session_state.setdefault("_ls_ops", [])


def sync():
    """Викликати один раз на початку кожного запуску сторінки (до читання/запису проєктів)."""
    if _backend() == "memory":
        st.session_state["_ls_items"] = dict(_MEMORY)
        st.session_state["_ls_ready"] = True
        return
    ops = _ops()
    with st.sidebar:
        res = _component()(data={"ops": ops}, key="_ls_bridge", default={"store": None},
                           on_store_change=lambda: None)
    store = getattr(res, "store", None) if res is not None else None
    if not store:
        return
    acked = int(store.get("applied") or 0)
    pending = [o for o in ops if o["id"] > acked]
    st.session_state["_ls_ops"] = pending
    items = dict(store.get("items") or {})
    for o in pending:                                   # ще не застосовані операції — поверх
        if o["value"] is None:
            items.pop(o["key"], None)
        else:
            items[o["key"]] = o["value"]
    st.session_state["_ls_items"] = items
    st.session_state["_ls_ready"] = True
    st.session_state["_ls_error"] = store.get("error")


def ready() -> bool:
    return bool(st.session_state.get("_ls_ready"))


def last_error() -> str | None:
    return st.session_state.get("_ls_error")


def _queue(key: str, value: str | None):
    items = st.session_state.setdefault("_ls_items", {})
    if value is None:
        items.pop(key, None)
    else:
        items[key] = value
    if _backend() == "memory":
        if value is None:
            _MEMORY.pop(key, None)
        else:
            _MEMORY[key] = value
        return
    ops = _ops()
    next_id = max([o["id"] for o in ops] + [st.session_state.get("_ls_op_seq", 0)]) + 1
    st.session_state["_ls_op_seq"] = next_id
    ops.append({"id": next_id, "key": key, "value": value})


# ---------------- Документ проєкту ----------------

def new_id() -> str:
    return uuid.uuid4().hex[:10]


def make_doc(pid: str, kind: str, name: str, scale: str, summary: dict, state: dict) -> dict:
    return {"format": FORMAT, "version": VERSION, "id": pid, "kind": kind, "name": name, "scale": scale,
            "saved_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "state": state}


def _default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Не вдається зберегти {type(o).__name__}")


def dumps(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"), default=_default)


def to_bytes(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False, indent=1, default=_default).encode("utf-8")


def is_project(doc) -> bool:
    return isinstance(doc, dict) and doc.get("format") == FORMAT


# ---------------- Операції ----------------

def save(doc: dict) -> None:
    _queue(doc["id"], dumps(doc))


def load(pid: str) -> dict:
    return json.loads(st.session_state.get("_ls_items", {})[pid])


def delete(pid: str) -> None:
    _queue(pid, None)


def list_projects() -> list[dict]:
    """Короткі описи всіх проєктів, найсвіжіші першими."""
    out = []
    for pid, raw in (st.session_state.get("_ls_items") or {}).items():
        try:
            d = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if is_project(d):
            out.append({k: d.get(k) for k in ("id", "kind", "name", "scale", "saved_at", "summary")}
                       | {"bytes": len(raw)})
    return sorted(out, key=lambda d: d.get("saved_at") or "", reverse=True)
