"""Unit tests for the schema-discovery module (discovery/).

The repo root is intentionally kept off pytest's pythonpath (see pytest.ini) to
avoid shadowing the installed `mcp` SDK. These tests only touch `discovery`, so
adding the root here is safe.
"""

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

duckdb = pytest.importorskip("duckdb")
from discovery import run_discovery  # noqa: E402


def _csv(path: Path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def ecommerce(tmp_path):
    """orders->customers, order_items->orders, order_items->products; all id PKs."""
    import random
    random.seed(7)
    _csv(tmp_path / "customers.csv", ["id", "name", "country"],
         [[i, f"C{i}", "US"] for i in range(1, 41)])
    _csv(tmp_path / "products.csv", ["id", "name", "price"],
         [[i, f"P{i}", round(random.uniform(5, 200), 2)] for i in range(1, 26)])
    _csv(tmp_path / "orders.csv", ["id", "customer_id", "status", "total_amount"],
         [[i, random.randint(1, 40), "completed", round(random.uniform(10, 500), 2)]
          for i in range(1, 121)])
    items, iid = [], 1
    for oid in range(1, 121):
        for _ in range(random.randint(1, 4)):
            items.append([iid, oid, random.randint(1, 25), random.randint(1, 5)])
            iid += 1
    _csv(tmp_path / "order_items.csv", ["id", "order_id", "product_id", "quantity"], items)
    return sorted(tmp_path.glob("*.csv"))


def test_recovers_exactly_the_true_fks(ecommerce):
    r = run_discovery(ecommerce)
    accepted = {(j.fk_table, j.fk_column, j.pk_table, j.pk_column) for j in r.accepted}
    assert accepted == {
        ("orders", "customer_id", "customers", "id"),
        ("order_items", "order_id", "orders", "id"),
        ("order_items", "product_id", "products", "id"),
    }
    assert all(j.relationship == "many_to_one" for j in r.accepted)


def test_single_column_grain_and_no_uncertain_noise(ecommerce):
    r = run_discovery(ecommerce)
    assert all(g.kind == "single" and g.key == ["id"] for g in r.grains.values())
    # coincidental id/quantity matches are pruned or explained away
    assert r.uncertain == []


def test_quantity_never_accepted_as_fk(ecommerce):
    r = run_discovery(ecommerce)
    assert not any(j.fk_column == "quantity" for j in r.accepted)


def test_cube_join_shape(ecommerce):
    r = run_discovery(ecommerce)
    j = next(j for j in r.accepted if j.fk_column == "customer_id")
    assert j.cube_join == {
        "customers": {
            "sql": "${CUBE}.customer_id = ${customers.id}",
            "relationship": "many_to_one",
        }
    }


@pytest.fixture
def junction(tmp_path):
    """enrollments is a junction: no surrogate id, grain = (student_id, course_id)."""
    import random
    random.seed(1)
    _csv(tmp_path / "students.csv", ["id", "name"], [[i, f"S{i}"] for i in range(1, 21)])
    _csv(tmp_path / "courses.csv", ["id", "title"], [[i, f"C{i}"] for i in range(1, 9)])
    seen, enr = set(), []
    while len(enr) < 40:
        s, c = random.randint(1, 20), random.randint(1, 8)
        if (s, c) in seen:
            continue
        seen.add((s, c))
        enr.append([s, c, round(random.uniform(50, 100), 1)])  # grade: unique float, NOT the key
    _csv(tmp_path / "enrollments.csv", ["student_id", "course_id", "grade"], enr)
    return sorted(tmp_path.glob("*.csv"))


def test_composite_grain_ignores_coincidental_unique_float(junction):
    r = run_discovery(junction)
    g = r.grains["enrollments"]
    assert g.kind == "composite"
    assert set(g.key) == {"student_id", "course_id"}


def test_junction_detected_as_bridge(junction):
    r = run_discovery(junction)
    assert r.bridges() == ["enrollments"]


def test_graph_has_nodes_and_directed_edges(ecommerce):
    g = run_discovery(ecommerce).graph()
    assert {n["id"] for n in g["nodes"]} == {"customers", "products", "orders", "order_items"}
    assert all(e["status"] == "accepted" for e in g["edges"])
    assert ("order_items", "orders") in {(e["source"], e["target"]) for e in g["edges"]}
