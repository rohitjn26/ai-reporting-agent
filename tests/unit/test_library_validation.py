"""Unit tests for mcp/library_server.py::_validate_fields."""


def test_valid_measure_and_dimension_pass(library_server_mod):
    errors = library_server_mod._validate_fields(
        {"revenue": {"sql": "amount", "type": "sum", "title": "Revenue"}},
        {"status": {"sql": "status", "type": "string", "title": "Status"}},
    )
    assert errors == []


def test_measure_missing_sql(library_server_mod):
    errors = library_server_mod._validate_fields({"x": {"type": "sum"}}, {})
    assert any("missing 'sql'" in e and "x" in e for e in errors)


def test_measure_invalid_type(library_server_mod):
    errors = library_server_mod._validate_fields(
        {"x": {"sql": "id", "type": "bogus"}}, {}
    )
    assert any("invalid type 'bogus'" in e for e in errors)


def test_measure_must_be_object(library_server_mod):
    errors = library_server_mod._validate_fields({"x": "not-a-dict"}, {})
    assert errors == ["measure 'x' must be an object"]


def test_dimension_missing_sql(library_server_mod):
    errors = library_server_mod._validate_fields({}, {"d": {"type": "string"}})
    assert any("missing 'sql'" in e and "d" in e for e in errors)


def test_dimension_with_case_needs_no_sql(library_server_mod):
    errors = library_server_mod._validate_fields(
        {}, {"tier": {"case": {"when": []}, "type": "string"}}
    )
    assert errors == []


def test_dimension_invalid_type(library_server_mod):
    errors = library_server_mod._validate_fields(
        {}, {"d": {"sql": "x", "type": "sum"}}  # sum is a measure type, not a dim type
    )
    assert any("invalid type 'sum'" in e for e in errors)


def test_type_is_optional(library_server_mod):
    # A field with sql but no type is allowed (type only validated when present).
    errors = library_server_mod._validate_fields({"x": {"sql": "id"}}, {})
    assert errors == []


def test_none_inputs_are_safe(library_server_mod):
    assert library_server_mod._validate_fields(None, None) == []


def test_collects_multiple_errors(library_server_mod):
    errors = library_server_mod._validate_fields(
        {"a": {"type": "sum"}},          # missing sql
        {"b": {"sql": "x", "type": "z"}}  # invalid dim type
    )
    assert len(errors) == 2
