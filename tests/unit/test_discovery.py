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


@pytest.fixture
def objectid(tmp_path):
    """Mongo-style export: ObjectId string keys, FK columns named freely
    (c_task, author, reviewer) — no `_id` suffix anywhere."""
    import random
    random.seed(3)
    oid = lambda p, i: f"{p}{i:022x}"  # 24-char hex, like an ObjectId
    users = [oid("a1", i) for i in range(30)]
    tasks = [oid("b2", i) for i in range(6)]
    _csv(tmp_path / "account.csv", ["id", "name"], [[u, f"U{i}"] for i, u in enumerate(users)])
    _csv(tmp_path / "c_task.csv", ["id", "title", "author"],
         [[t, f"T{i}", random.choice(users)] for i, t in enumerate(tasks)])
    _csv(tmp_path / "response.csv",
         ["id", "c_task", "author", "reviewer", "score", "c_account"],
         # distinct values into account: reviewer 30 > c_account 10 > author 3
         [[oid("c3", i), random.choice(tasks), users[i % 3], users[i % 30],
           random.randint(1, 30), users[i % 10]] for i in range(200)])
    return sorted(tmp_path.glob("*.csv"))


def _pairs(joins):
    return {(j.fk_table, j.fk_column, j.pk_table) for j in joins}


def test_high_entropy_keys_accepted_on_containment_alone(objectid):
    r = run_discovery(objectid)
    assert _pairs(r.accepted) == {
        ("response", "c_task", "c_task"),       # named after the table
        ("response", "author", "account"),      # no name signal: containment decides
        ("response", "reviewer", "account"),
        ("response", "c_account", "account"),
        ("c_task", "author", "account"),
    }
    assert r.uncertain == []


def test_low_entropy_key_needs_a_name(tmp_path):
    # shelf (1..15) is fully contained in store.id (1..20) with no name signal:
    # with small ints that's plausibly chance, so it goes to review, not accepted
    import random
    random.seed(5)
    _csv(tmp_path / "store.csv", ["id", "city"], [[i, f"C{i}"] for i in range(1, 21)])
    _csv(tmp_path / "sale.csv", ["id", "shelf"],
         [[i, random.randint(1, 15)] for i in range(1, 201)])
    r = run_discovery(sorted(tmp_path.glob("*.csv")))
    assert r.accepted == []
    assert _pairs(r.uncertain) == {("sale", "shelf", "store")}


def test_table_named_column_is_a_name_signal():
    from discovery.joins import _name_signal
    assert _name_signal("x", "c_task", "c_task", "id") == "table"
    assert _name_signal("x", "c_account", "account", "id") == "table"
    assert _name_signal("x", "customer_id", "customers", "id") == "suffix_id"
    assert _name_signal("x", "reviewer", "account", "id") == "none"


# ── semantic-layer draft (discovery/semantic.py) ─────────────────────────────

from discovery import draft_semantic_layer  # noqa: E402


def _cube(draft, name):
    return next(c["data"] for c in draft["cubes"] if c["name"] == name)


def _view(draft, root):
    return next(v["data"] for v in draft["views"] if v["name"] == f"{root}_view")


def _paths(view):
    return {e["join_path"]: e for e in view["cubes"]}


def test_semantic_one_view_per_connected_group(ecommerce):
    d = draft_semantic_layer(run_discovery(ecommerce).to_dict())
    assert d["roles"] == {"orders": "fact", "order_items": "fact",
                          "customers": "dimension", "products": "dimension"}
    # all four tables are joined, so one view; customers reaches the most tables going down
    assert [v["name"] for v in d["views"]] == ["customers_view"]
    assert d["root"] == "customers"
    assert d["notes"] == []


def test_semantic_cubes_are_private_with_pk_and_joins(ecommerce):
    r = run_discovery(ecommerce)
    d = draft_semantic_layer(r.to_dict())
    orders = _cube(d, "orders")
    assert orders["public"] is False
    assert orders["dimensions"]["id"]["primary_key"] is True
    fk_join = next(j for j in r.accepted if j.fk_table == "orders").cube_join
    assert fk_join.items() <= orders["joins"].items()
    assert "customer_id" not in orders["dimensions"]  # FK is join plumbing
    assert orders["measures"]["total_amount"]["type"] == "sum"
    # dimension tables aggregate nothing but count
    assert set(_cube(d, "products")["measures"]) == {"count"}


def test_semantic_view_tree_spine_down_lookups_up(ecommerce):
    d = draft_semantic_layer(run_discovery(ecommerce).to_dict())
    paths = _paths(_view(d, "customers"))
    assert set(paths) == {"customers", "customers.orders", "customers.orders.order_items",
                          "customers.orders.order_items.products"}
    # spine tables are facts at their grain: measures included (Cube dedups by PK)
    assert "total_quantity" in paths["customers.orders.order_items"]["includes"]
    assert "total_amount" in paths["customers.orders"]["includes"]
    assert "id" not in paths["customers.orders"]["includes"]
    # the lookup contributes attributes only
    products = paths["customers.orders.order_items.products"]
    assert not set(products["includes"]) & set(_cube(d, "products")["measures"])
    assert all(e["prefix"] for e in paths.values())
    # the view walks down, so the parent declares the one_to_many join
    assert _cube(d, "customers")["joins"]["orders"] == {
        "sql": "${CUBE}.id = ${orders}.customer_id", "relationship": "one_to_many"}


def test_semantic_root_can_be_chosen(ecommerce):
    d = draft_semantic_layer(run_discovery(ecommerce).to_dict(), root="order_items")
    assert d["root"] == "order_items"
    assert set(_paths(_view(d, "order_items"))) == {
        "order_items", "order_items.orders", "order_items.products", "order_items.orders.customers"}


def test_semantic_branches_never_meet_only_at_the_root(tmp_path):
    """org -> site -> participant -> response -> answer; task is referenced by
    response AND event. Hanging task straight off org would cross-multiply it
    with every response, so each referencing branch gets its own copy."""
    oid = lambda p, i: f"{p}{i:022x}"
    org = oid("0f", 0)
    sites = [oid("5c", i) for i in range(6)]
    users = [oid("a1", i) for i in range(24)]
    tasks = [oid("7a", i) for i in range(4)]
    _csv(tmp_path / "org.csv", ["id", "name"], [[org, "acme"]])
    _csv(tmp_path / "site.csv", ["id", "org", "region"],
         [[s_, org, "EU" if i % 2 else "US"] for i, s_ in enumerate(sites)])
    _csv(tmp_path / "task.csv", ["id", "org", "title"], [[t, org, f"T{i}"] for i, t in enumerate(tasks)])
    _csv(tmp_path / "participant.csv", ["id", "org", "site", "age"],
         [[u, org, sites[i % 6], 20 + i] for i, u in enumerate(users)])
    _csv(tmp_path / "response.csv", ["id", "org", "participant", "task"],
         [[oid("e5", i), org, users[i % 24], tasks[i % 4]] for i in range(60)])
    _csv(tmp_path / "event.csv", ["id", "org", "participant", "task"],
         [[oid("e7", i), org, users[i % 24], tasks[i % 4]] for i in range(90)])
    d = draft_semantic_layer(run_discovery(sorted(tmp_path.glob("*.csv"))).to_dict())
    paths = _paths(_view(d, "org"))
    # the bigger branch gets task itself, the other its own copy cube — a Cube
    # view may reach each cube by one path only
    assert {p for p in paths if p.endswith("task")} == {
        "org.site.participant.event.task", "org.site.participant.response.response_task"}
    assert "org.task" not in paths
    copy = _cube(d, "response_task")
    assert copy["sql"] == _cube(d, "task")["sql"]
    assert _cube(d, "response")["joins"]["response_task"]["sql"] == \
        "${CUBE}.task = ${response_task.id}"
    assert "task" not in _cube(d, "response")["joins"]
    assert "org.site.participant.response" in paths and "org.site.participant.event" in paths


def test_semantic_respects_user_review(ecommerce):
    disc = run_discovery(ecommerce).to_dict()
    kept = [j for j in disc["joins"]["accepted"] if j["fk"]["column"] != "product_id"]
    kept = [dict(j, relationship="one_to_many") if j["fk"]["column"] == "customer_id" else j
            for j in kept]
    d = draft_semantic_layer(disc, kept)
    assert "products" not in _cube(d, "order_items")["joins"]
    assert _cube(d, "orders")["joins"]["customers"]["relationship"] == "one_to_many"
    # a fan-out join builds no view path: orders and customers split into two views
    assert all("customers" not in e["join_path"]
               for v in d["views"] if v["name"] != "customers_view" for e in v["data"]["cubes"])
    assert any("fans out" in n for n in d["notes"])


def test_semantic_bridge_gets_composite_pk_and_view(junction):
    d = draft_semantic_layer(run_discovery(junction).to_dict())
    assert d["roles"]["enrollments"] == "bridge"
    enr = _cube(d, "enrollments")
    assert enr["dimensions"]["pk"]["primary_key"] is True
    assert enr["dimensions"]["pk"]["sql"].startswith("CONCAT(")
    # grade is non-additive: averaged, never summed
    assert "avg_grade" in enr["measures"] and "total_grade" not in enr["measures"]
    # courses and students each reach one table; courses is smaller, so it's the root
    assert set(_paths(_view(d, "courses"))) == {
        "courses", "courses.enrollments", "courses.enrollments.students"}


def test_semantic_dataset_namespaces_everything(ecommerce):
    d = draft_semantic_layer(run_discovery(ecommerce).to_dict(), dataset="Retail Q3")
    assert d["dataset"] == "retail_q3"
    assert {c["name"] for c in d["cubes"]} == {
        "retail_q3_orders", "retail_q3_order_items", "retail_q3_customers", "retail_q3_products"}
    orders = next(c["data"] for c in d["cubes"] if c["name"] == "retail_q3_orders")
    assert orders["sql"] == "SELECT * FROM retail_q3.orders"
    assert orders["joins"]["retail_q3_customers"] == {
        "sql": "${CUBE}.customer_id = ${retail_q3_customers.id}", "relationship": "many_to_one"}
    v = next(v["data"] for v in d["views"] if v["name"] == "retail_q3_customers_view")
    assert [e["join_path"] for e in v["cubes"]][:2] == [
        "retail_q3_customers", "retail_q3_customers.retail_q3_orders"]
    assert d["roles"]["retail_q3_orders"] == "fact"
    assert set(d["sources"]) == {"orders", "order_items", "customers", "products"}


def test_dataset_name_is_schema_safe():
    from discovery.semantic import dataset_name
    assert dataset_name("Retail Q3") == "retail_q3"
    assert dataset_name("2024-sales") == "ds_2024_sales"


def test_load_data_refuses_without_dataset(ecommerce):
    from discovery.publish import load_data
    with pytest.raises(ValueError, match="dataset"):
        load_data(draft_semantic_layer(run_discovery(ecommerce).to_dict()))


def test_seed_from_draft_detects_name_clashes():
    sys.path.insert(0, str(REPO_ROOT / "library"))
    from seed import _draft_collisions
    draft = {"cubes": [{"name": "shop_orders"}, {"name": "orders"}],
             "views": [{"name": "shop_orders_view"}]}
    existing = {"CUBE_CONFIG": {"orders": 1}, "VIEW": {"sales": 2}}
    assert _draft_collisions(draft, existing) == ["CUBE_CONFIG orders"]


def test_highest_cardinality_join_is_primary(objectid):
    r = run_discovery(objectid)
    into_account = {j.fk_column: j.primary for j in r.accepted
                    if (j.fk_table, j.pk_table) == ("response", "account")}
    # reviewer has the most distinct values, so it wins over the better-named c_account
    assert into_account == {"author": False, "reviewer": True, "c_account": False}
    # a lone join into a table is trivially primary
    assert next(j for j in r.accepted if j.fk_column == "c_task").primary


def test_semantic_uses_primary_join_and_notes_the_rest(objectid):
    d = draft_semantic_layer(run_discovery(objectid).to_dict())
    resp = _cube(d, "response")
    # three FK columns reach account; Cube allows one join per target
    assert resp["joins"]["account"]["sql"] == "${CUBE}.reviewer = ${account.id}"
    assert "response -> account: joined on reviewer; not used: author, c_account — " \
           "Relationships → search 'response account' to switch or add one." in d["notes"]


def test_semantic_honours_user_primary_pick(objectid):
    disc = run_discovery(objectid).to_dict()
    joins = disc["joins"]["accepted"]
    for j in joins:  # the user switches response -> account to author in the UI
        if (j["fk"]["table"], j["pk"]["table"]) == ("response", "account"):
            j["primary"] = j["fk"]["column"] == "author"
    d = draft_semantic_layer(disc, joins=joins)
    assert _cube(d, "response")["joins"]["account"]["sql"] == "${CUBE}.author = ${account.id}"


# ── AI descriptions (discovery/describe.py) — fake LLM, no network ───────────

class _FakeDescriber:
    """Stands in for the structured LLM: returns canned text per table, and
    records the prompts so tests can check what would be sent."""
    def __init__(self, fail=()):
        self.prompts, self.fail = [], set(fail)

    def invoke(self, msgs):
        from discovery.describe import ColumnText, TableText
        prompt = msgs[-1][1]
        self.prompts.append(prompt)
        table = prompt.split("Table: ")[1].split(" ")[0]
        if table in self.fail:
            raise RuntimeError("rate limited")
        cols = [line.split(" | ")[0].strip() for line in prompt.splitlines() if " | " in line][1:]
        return TableText(title=table.title() + "s", description=f"All the {table}s.",
                         synonyms=[f"{table} rows"],
                         columns=[ColumnText(name=c, title=c.upper(), description=f"The {c}.")
                                  for c in cols] + [ColumnText(name="made_up", title="X",
                                                               description="invented")])


def test_describe_tables_prompts_with_profile_and_drops_invented_columns(ecommerce):
    from discovery.describe import describe_tables
    disc = run_discovery(ecommerce).to_dict()
    fake = _FakeDescriber(fail={"products"})
    out = describe_tables(disc, disc["joins"]["accepted"], dataset="shop", llm=fake)
    assert set(out["tables"]) == {"customers", "orders", "order_items"}
    assert out["errors"] == {"products": "rate limited"}          # one failure doesn't sink the rest
    orders = out["tables"]["orders"]
    assert orders["title"] == "Orderss" and orders["synonyms"] == ["orders rows"]
    assert "made_up" not in orders["columns"] and orders["columns"]["status"]["title"] == "STATUS"
    p = next(p for p in fake.prompts if "Table: orders " in p)
    assert "references customers via customer_id" in p and "referenced by order_items.order_id" in p
    assert "Dataset: shop" in p and "status | VARCHAR" in p


def test_descriptions_reach_cubes_measures_and_view(ecommerce):
    disc = run_discovery(ecommerce).to_dict()
    desc = {"tables": {"customers": {"title": "Shoppers", "description": "People who buy.",
                                     "synonyms": ["buyers", "clients"],
                                     "columns": {"country": {"title": "Home Country",
                                                             "description": "Where they live."}}},
                       "orders": {"title": "Orders", "description": "One purchase.",
                                  "columns": {"total_amount": {"title": "Order Value",
                                                               "description": "Paid amount."}}}},
            "views": {"customers": {"title": "Shop", "description": "Everything about the shop."}}}
    d = draft_semantic_layer(disc, descriptions=desc)
    cust = _cube(d, "customers")
    assert cust["title"] == "Shoppers"
    assert cust["description"] == "People who buy. Also called: buyers, clients."
    assert cust["measures"]["count"]["description"] == "Number of shoppers (buyers, clients)."
    assert cust["dimensions"]["country"] == {**cust["dimensions"]["country"],
                                             "title": "Home Country", "description": "Where they live."}
    total = _cube(d, "orders")["measures"]["total_amount"]
    assert (total["title"], total["description"]) == ("Total Order Value", "Sum of Order Value: Paid amount.")
    v = _view(d, "customers")
    assert (v["title"], v["description"]) == ("Shop", "Everything about the shop.")
    # the view carries its cubes' descriptions for the router
    assert {c["name"]: c["title"] for c in v["meta"]["cubes"]}["customers"] == "Shoppers"


def test_view_meta_present_without_descriptions(ecommerce):
    v = _view(draft_semantic_layer(run_discovery(ecommerce).to_dict()), "customers")
    assert [c["name"] for c in v["meta"]["cubes"]] == ["customers", "orders", "order_items", "products"]


def test_synonyms_as_one_string_are_split():
    from discovery.describe import TableText
    t = TableText.model_validate({"title": "Users", "description": "d", "columns": [],
                                  "synonyms": "Users, User Accounts, Login Accounts"})
    assert t.synonyms == ["Users", "User Accounts", "Login Accounts"]



def test_second_join_into_same_table_becomes_its_own_copy(objectid):
    """reviewer is the join into account; author is wanted too (a role) — Cube
    allows one join per target cube, so author gets its own copy of account."""
    disc = run_discovery(objectid).to_dict()
    joins = disc["joins"]["accepted"]
    for j in joins:
        if (j["fk"]["table"], j["fk"]["column"]) == ("response", "author"):
            j["role"] = True
    d = draft_semantic_layer(disc, joins=joins)
    resp = _cube(d, "response")
    assert resp["joins"]["account"]["sql"] == "${CUBE}.reviewer = ${account.id}"      # primary stays
    assert resp["joins"]["response_author"] == {
        "sql": "${CUBE}.author = ${response_author.id}", "relationship": "many_to_one"}
    copy = _cube(d, "response_author")
    assert copy["sql"] == _cube(d, "account")["sql"] and copy["title"] == "Account (Author)"
    paths = {e["join_path"] for v in d["views"] for e in v["data"]["cubes"]}
    assert any(p.endswith("response.response_author") for p in paths)
    assert any(n.endswith("'response account' to switch or add one.") and "also author" in n
               and "not used: c_account" in n for n in d["notes"])


def test_spine_never_hangs_a_table_under_an_incomplete_fk(tmp_path):
    """participant.account is filled on 90% of rows. Walking org -> account ->
    participant would drop the other 10% (and everything under them), so the
    participant hangs under site (100% linked) even though account is bigger."""
    oid = lambda p, i: f"{p}{i:022x}"
    org = oid("0f", 0)
    sites = [oid("5c", i) for i in range(6)]
    accounts = [oid("ac", i) for i in range(60)]
    users = [oid("a1", i) for i in range(40)]
    _csv(tmp_path / "org.csv", ["id", "name"], [[org, "acme"]])
    _csv(tmp_path / "site.csv", ["id", "org", "region"],
         [[s_, org, "EU" if i % 2 else "US"] for i, s_ in enumerate(sites)])
    _csv(tmp_path / "account.csv", ["id", "org", "email"],
         [[a, org, f"u{i}@x.io"] for i, a in enumerate(accounts)])
    _csv(tmp_path / "participant.csv", ["id", "org", "site", "account", "age"],
         [[u, org, sites[i % 6], accounts[i] if i % 10 else "", 20 + i] for i, u in enumerate(users)])
    _csv(tmp_path / "response.csv", ["id", "org", "participant"],
         [[oid("e5", i), org, users[i % 40]] for i in range(120)])
    d = draft_semantic_layer(run_discovery(sorted(tmp_path.glob("*.csv"))).to_dict())
    paths = _paths(_view(d, "org"))
    assert "org.site.participant" in paths and "org.site.participant.response" in paths
    assert not any(p.startswith("org.account.") for p in paths)
    assert "org.site.participant.account" in paths          # still reachable, as a lookup
