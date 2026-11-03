"""Plain-English join/key reasons (discovery/explain.py) and publishing a
dataset, then adding tables to it (discovery/publish.py). No Postgres or
library needed: the library is a fake and only empty loads are run.
"""

import csv
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

duckdb = pytest.importorskip("duckdb")
from discovery import draft_semantic_layer, publish, run_discovery  # noqa: E402


def _csv(path: Path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def shop(tmp_path):
    random.seed(5)
    _csv(tmp_path / "customers.csv", ["id", "name"], [[i, f"C{i}"] for i in range(1, 41)])
    _csv(tmp_path / "orders.csv", ["id", "customer_id", "amount"],
         [[i, random.randint(1, 40), random.randint(5, 90)] for i in range(1, 151)])
    return tmp_path


def _join(r, col):
    return next(j for j in r.to_dict()["joins"]["accepted"] + r.to_dict()["joins"]["uncertain"]
                if j["fk"]["column"] == col)


# ── why ──────────────────────────────────────────────────────────────────────

def test_named_join_explains_each_check_and_verdict(shop):
    j = _join(run_discovery(sorted(shop.glob("*.csv"))), "customer_id")
    assert j["decision"] == "best_name"
    assert [w["kind"] for w in j["why"]] == ["verdict", "values", "name", "luck", "direction"]
    by = {w["kind"]: w for w in j["why"]}
    assert by["verdict"]["ok"] and by["verdict"]["text"] == "Accepted — values and name match"
    assert by["values"]["ok"] and by["values"]["text"] == "100% of values found in customers"
    assert by["name"]["text"] == "Name matches"
    assert by["luck"]["ok"] is None and "no gaps" in by["luck"]["detail"]   # ids 1..40: not proof
    assert by["direction"]["text"] == "Many orders → one customers"
    assert all(len(w["text"]) <= 50 for w in j["why"])     # short; numbers live in detail


def test_proven_join_says_it_cant_be_coincidence(tmp_path):
    random.seed(13)
    ids = [i for i in range(1, 301) if i % 3]
    _csv(tmp_path / "customers.csv", ["id", "name"], [[i, f"C{i}"] for i in ids])
    _csv(tmp_path / "sale.csv", ["id", "buyer"], [[i, random.choice(ids)] for i in range(1, 801)])
    j = _join(run_discovery(sorted(tmp_path.glob("*.csv"))), "buyer")
    by = {w["kind"]: w for w in j["why"]}
    assert j["decision"] == "proven_only"
    assert by["verdict"]["text"] == "Accepted — proven by the data"
    assert by["luck"]["ok"] and by["luck"]["text"] == "Can't be a coincidence"
    assert by["name"]["ok"] is False and by["name"]["text"] == "Name doesn't match"


def test_orphans_hold_a_join_for_review_and_say_so(shop):
    # 5% of orders point at customers that don't exist
    random.seed(3)
    _csv(shop / "orders.csv", ["id", "customer_id", "amount"],
         [[i, random.randint(1, 40) if i % 20 else 900 + i, 10] for i in range(1, 201)])
    r = run_discovery(sorted(shop.glob("*.csv")))
    j = _join(r, "customer_id")
    by = {w["kind"]: w for w in j["why"]}
    assert j["decision"] == "low_containment" and j in r.to_dict()["joins"]["uncertain"]
    assert by["verdict"]["text"] == "Review — too many values missing"
    assert by["values"]["ok"] is False and "isn't in customers" in by["values"]["detail"]


def test_grain_reasons(shop):
    d = run_discovery(sorted(shop.glob("*.csv"))).to_dict()
    why = d["grains"]["customers"]["why"]
    assert why[0]["text"] == "id identifies each row"
    assert "Also unique: name" in why[0]["detail"]


# ── publish + add data ───────────────────────────────────────────────────────

@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    d = tmp_path / "datasets"
    monkeypatch.setattr(publish, "DATASETS_DIR", d)
    return d


def _draft(folder, ds="shop"):
    r = run_discovery(sorted(folder.glob("*.csv")))
    return draft_semantic_layer(r.to_dict(), None, ds)


def test_first_publish_loads_everything_and_flags_clashes(shop, datasets_dir):
    d = _draft(shop)
    plan = publish.plan_publish(d, {"CUBE_CONFIG": {}, "VIEW": {}})
    assert plan["first_publish"] and plan["load_tables"] == ["customers", "orders"]
    assert not plan["clashes"] and publish.check_publishable(plan) is None
    lib = {"CUBE_CONFIG": {"shop_orders": {"id": "x", "data": {}}}, "VIEW": {}}
    plan = publish.plan_publish(d, lib)
    assert plan["clashes"] == ["CUBE_CONFIG shop_orders"]
    assert "already in the library" in publish.check_publishable(plan)
    assert publish.check_publishable(plan, force=True) is None


def test_adding_a_csv_loads_only_the_new_table(shop, datasets_dir):
    first = _draft(shop)
    publish.record_publish(first, {"folder": str(shop)})
    library = {rtype: {c["name"]: {"id": c["name"], "data": c["data"]} for c in first[key]}
               for rtype, key in (("CUBE_CONFIG", "cubes"), ("VIEW", "views"))}

    # a new CSV = a new table, referenced by a column the published orders already has
    _csv(shop / "regions.csv", ["region_code", "region_name"],
         [[f"R{i:02d}", f"Region {i}"] for i in range(1, 6)])
    _csv(shop / "customers.csv", ["id", "name", "region_code"],
         [[i, f"C{i}", f"R{1 + i % 5:02d}"] for i in range(1, 41)])
    plan = publish.plan_publish(_draft(shop), library)
    assert not plan["first_publish"] and not plan["clashes"]
    assert plan["load_tables"] == ["regions"]
    assert plan["kept_tables"] == ["customers", "orders"]
    assert "shop_regions" in plan["cubes"]["new"]
    assert "shop_customers" in plan["cubes"]["changed"]     # gained the join to regions
    assert publish.load_manifest("shop")["state"] == {"folder": str(shop)}
    assert [m["dataset"] for m in publish.list_manifests()] == ["shop"]


def test_adding_never_drops_a_published_table(shop, datasets_dir):
    publish.record_publish(_draft(shop))
    (shop / "customers.csv").unlink()
    _csv(shop / "stores.csv", ["id", "city"], [[i, "X"] for i in range(1, 5)])
    plan = publish.plan_publish(_draft(shop), {"CUBE_CONFIG": {}, "VIEW": {}})
    assert plan["dropped_tables"] == ["customers"]
    assert "never removes tables" in publish.check_publishable(plan)


def test_upsert_skips_unchanged_and_undo_reverses(monkeypatch):
    calls = []

    def fake(url, payload=None, method="GET"):
        calls.append((method, url))
        return {"id": "new1"} if method == "POST" else {}
    monkeypatch.setattr(publish, "_request", fake)
    draft = {"cubes": [{"name": "a", "data": {"x": 1}}, {"name": "b", "data": {"x": 2}},
                       {"name": "c", "data": {"x": 3}}], "views": []}
    library = {"CUBE_CONFIG": {"a": {"id": "ida", "data": {"x": 1}},     # unchanged
                               "b": {"id": "idb", "data": {"x": 0}}},    # changed
               "VIEW": {}}
    done = publish.upsert_library(draft, library, base="L")
    assert [(d["name"], d["created"]) for d in done] == [("b", False), ("c", True)]
    assert calls == [("PUT", "L/v1/CUBE_CONFIG/idb"), ("POST", "L/v1/CUBE_CONFIG")]
    calls.clear()
    assert publish.undo_library(done, base="L") == []
    assert calls == [("DELETE", "L/v1/CUBE_CONFIG/new1"), ("POST", "L/v1/CUBE_CONFIG/idb/rollback")]


def test_upsert_failure_undoes_earlier_writes(monkeypatch):
    calls = []

    def fake(url, payload=None, method="GET"):
        calls.append((method, url))
        if method == "POST" and url == "L/v1/VIEW":
            raise RuntimeError("library POST: HTTP 500")
        return {"id": "new1"} if method == "POST" else {}
    monkeypatch.setattr(publish, "_request", fake)
    draft = {"cubes": [{"name": "a", "data": {"x": 1}}], "views": [{"name": "v", "data": {}}]}
    with pytest.raises(RuntimeError, match="earlier writes undone"):
        publish.upsert_library(draft, {"CUBE_CONFIG": {}, "VIEW": {}}, base="L")
    assert calls[-1] == ("DELETE", "L/v1/CUBE_CONFIG/new1")


def test_load_data_with_no_tables_touches_nothing(shop):
    d = _draft(shop)
    assert publish.load_data(d, "postgresql://nowhere:1/x", tables=[]) == {}
