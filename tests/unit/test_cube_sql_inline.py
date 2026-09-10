"""Unit tests for mcp/cube_server.py::_inline_sql_params."""
import pytest


def test_inlines_string_and_number(cube_server_mod):
    sql = cube_server_mod._inline_sql_params("WHERE a = $1 AND b = $2", ["x", 5])
    assert sql == "WHERE a = 'x' AND b = 5"


def test_escapes_single_quotes(cube_server_mod):
    sql = cube_server_mod._inline_sql_params("name = $1", ["O'Brien"])
    assert sql == "name = 'O''Brien'"


@pytest.mark.parametrize("value,expected", [
    (None, "NULL"),
    (True, "TRUE"),
    (False, "FALSE"),
    (3.5, "3.5"),
])
def test_scalar_rendering(cube_server_mod, value, expected):
    assert cube_server_mod._inline_sql_params("x = $1", [value]) == f"x = {expected}"


def test_bool_not_treated_as_number(cube_server_mod):
    # bool is a subclass of int — the bool branch must win.
    assert cube_server_mod._inline_sql_params("$1", [True]) == "TRUE"


def test_higher_indices_replaced_first(cube_server_mod):
    # $10 must not be clobbered by the $1 replacement.
    params = list(range(1, 11))  # 10 params → values 1..10
    sql = cube_server_mod._inline_sql_params("$1 and $10", params)
    assert sql == "1 and 10"
