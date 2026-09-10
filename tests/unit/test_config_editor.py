"""
Unit tests for agent/graph/config_editor.py::edit_cube_config.

The tool does one network fetch (`_lib_get`) and one `interrupt()` that pauses for
the UI form answer. Both are monkeypatched so we can drive the parsing logic in
isolation, without a running library API or a LangGraph runtime.
"""
import json

import pytest

from graph import config_editor


@pytest.fixture
def patch_lib(monkeypatch):
    """Make `_lib_get` return a canned single-cube config listing."""
    config = {
        "id": "cfg-123",
        "name": "orders",
        "data": {
            "measures": {"count": {"sql": "id", "type": "count", "title": "Count"}},
            "dimensions": {"status": {"sql": "status", "type": "string", "title": "Status"}},
        },
    }

    async def fake_lib_get(path):
        return {"data": [config]}

    monkeypatch.setattr(config_editor, "_lib_get", fake_lib_get)
    return config


def _patch_form_answer(monkeypatch, answer):
    """Make the interrupt() return `answer` as if the user submitted the form."""
    monkeypatch.setattr(config_editor, "interrupt", lambda payload: answer)


async def _run(**kwargs):
    return await config_editor.edit_cube_config.ainvoke(kwargs)


async def test_unknown_cube_returns_error_with_available_names(monkeypatch, patch_lib):
    _patch_form_answer(monkeypatch, {})  # never reached
    out = json.loads(await _run(cube_name="widgets", intent="add something"))
    assert "No cube named 'widgets'" in out["error"]
    assert "orders" in out["error"]


async def test_add_measure_returns_updated_pool(monkeypatch, patch_lib):
    _patch_form_answer(monkeypatch, json.dumps({
        "field_type": "measure", "action": "add", "key": "aov",
        "sql": "SUM(amount)/COUNT(id)", "type": "number", "title": "Avg Order Value",
    }))
    out = json.loads(await _run(cube_name="orders", intent="add average order value"))

    assert out["config_id"] == "cfg-123"
    assert out["field_type"] == "measure"
    assert out["field_key"] == "aov"
    assert out["action"] == "add"
    # new measure added alongside the existing one
    assert set(out["updated_measures"]) == {"count", "aov"}
    assert out["updated_measures"]["aov"] == {
        "sql": "SUM(amount)/COUNT(id)", "type": "number", "title": "Avg Order Value"
    }
    # dimensions untouched
    assert set(out["updated_dimensions"]) == {"status"}


async def test_add_dimension_updates_dimension_pool(monkeypatch, patch_lib):
    _patch_form_answer(monkeypatch, json.dumps({
        "field_type": "dimension", "action": "add", "key": "bi_month",
        "sql": "CASE WHEN ... END", "type": "string", "title": "Bi-Month",
    }))
    out = json.loads(await _run(cube_name="orders", intent="add a bi-month bucket"))
    assert set(out["updated_dimensions"]) == {"status", "bi_month"}
    assert set(out["updated_measures"]) == {"count"}  # unchanged


async def test_accepts_dict_answer_not_just_json_string(monkeypatch, patch_lib):
    # The UI may hand back a parsed dict rather than a JSON string.
    _patch_form_answer(monkeypatch, {
        "field_type": "measure", "action": "replace", "key": "count",
        "sql": "COUNT(DISTINCT id)", "type": "count_distinct", "title": "Unique",
    })
    out = json.loads(await _run(cube_name="orders", intent="make count distinct"))
    assert out["action"] == "replace"
    assert out["updated_measures"]["count"]["type"] == "count_distinct"


async def test_missing_key_is_rejected(monkeypatch, patch_lib):
    _patch_form_answer(monkeypatch, json.dumps({"field_type": "measure", "key": ""}))
    out = json.loads(await _run(cube_name="orders", intent="add nameless field"))
    assert out["error"] == "No field key provided."


async def test_unparseable_answer_is_reported(monkeypatch, patch_lib):
    _patch_form_answer(monkeypatch, "this is not json")
    out = json.loads(await _run(cube_name="orders", intent="add field"))
    assert "Could not parse form answer" in out["error"]
