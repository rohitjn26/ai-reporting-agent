"""Column stats for drill-down: computed by DuckDB (discovery's profiler), applied by
the drill engine (fill rate, distinct values, JSON/ids), and the agent-side hooks."""
import json

import duckdb
import pytest

from discovery import stats
from graph import agent, drill


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE SCHEMA ds")
    c.execute("""CREATE TABLE ds.people AS SELECT * FROM (VALUES
        ('a@x', 'Enrolled', NULL, '{"k": 1}', 1),
        ('b@x', 'Enrolled', NULL, '{"k": 2}', 2),
        ('c@x', 'New',      'x',  '{"k": 3}', 3),
        ('d@x', 'Deactivated', NULL, '[1]',   4)) t(email, status, sparse, blob, n)""")
    return c


def test_compute_profiles_every_column_with_discoverys_profiler(con):
    rows, failed = stats.compute(con, "memory", "ds")
    assert failed == {}
    by = {r["column_name"]: r for r in rows}
    assert set(by) == {"email", "status", "sparse", "blob", "n"}
    assert (by["status"]["row_count"], by["status"]["filled"], by["status"]["distinct_count"]) == (4, 4, 3)
    assert (by["sparse"]["filled"], by["sparse"]["distinct_count"]) == (1, 1)
    assert by["email"]["distinct_count"] == 4
    assert stats.structured(json.loads(by["blob"]["samples"]))
    assert not stats.structured(json.loads(by["status"]["samples"]))
    # the temp copy is cleaned up
    assert not con.execute("SELECT * FROM duckdb_tables() WHERE table_name LIKE '_stats_%'").fetchall()


def test_a_table_that_fails_is_reported_not_fatal(con):
    rows, failed = stats.compute(con, "memory", "ds", tables=["people", "missing"])
    assert "missing" in failed and {r["table_name"] for r in rows} == {"people"}


@pytest.mark.parametrize("samples,expected", [
    (['{"a":1}', '[2]', 'x'], True),
    (["69e6a5eda8281e62845ad69d", "69e6a5eea8281e62845ad6d1"], True),
    (["US", "IN", None], False),
    ([], False),
])
def test_structured_spots_json_and_object_ids(samples, expected):
    assert stats.structured(samples) is expected


# ── drill: map dimensions to tables, then filter and order by stats ───────────

SQL = '''SELECT 1 FROM app.org AS "org"
LEFT JOIN app.site AS "site" ON "org".id = "site".org
LEFT JOIN "app"."part" AS "part" ON "part".site = "site".id
LEFT JOIN app.response AS "resp" ON "part".id = "resp".participant'''


def test_alias_tables_reads_schema_and_table_for_every_alias():
    assert drill.alias_tables(SQL) == {"org": ("app", "org"), "site": ("app", "site"),
                                       "part": ("app", "part"), "resp": ("app", "response")}


VIEW = [{"name": "v", "measures": [{"name": "v.count", "aggType": "count", "aliasMember": "resp.count"}],
         "dimensions": [{"name": f"v.{n}", "type": "string", "aliasMember": a} for n, a in [
             ("site_name", "site.name"), ("site_country", "site.country"), ("site_code", "site.code"),
             ("part_status", "part.status"), ("part_email", "part.email"), ("part_notes", "part.notes"),
             ("part_flag", "part.flag"), ("part_blob", "part.blob"), ("part_new", "part.new_col")]]}]


def S(fill, distinct, structured=False):
    return stats.ColumnStat(100, int(fill * 100), distinct, structured)


STATS = {"v.site_name": S(1, 8), "v.site_country": S(1, 5), "v.site_code": S(1, 1),
         "v.part_status": S(1, 4), "v.part_email": S(.93, 499), "v.part_notes": S(.4, 30),
         "v.part_flag": S(1, 2), "v.part_blob": S(1, 12, structured=True)}   # part_new: no stats


def test_stats_drop_empty_single_value_identifier_and_json_columns():
    kept = {c.member for c in drill.candidates({"measures": ["v.count"]}, VIEW, None, STATS)}
    assert kept == {"v.site_name", "v.site_country", "v.part_status", "v.part_flag", "v.part_new"}


def test_label_columns_first_then_fewest_values_and_unknowns_after():
    dist = drill.cube_distances(drill.join_edges(SQL), "resp")
    order = [c.member for c in drill.candidates({"measures": ["v.count"]}, VIEW, dist, STATS)]
    assert order.index("v.site_name") < order.index("v.site_country")      # label beats category
    assert order.index("v.part_status") < order.index("v.part_flag")       # 3-20 values beat yes/no
    assert order.index("v.part_flag") < order.index("v.part_new")          # known-good beats unknown


# ── drill: blank for most of the rows being explained ─────────────────────────

async def test_a_split_blank_for_most_rows_is_skipped():
    from test_drill import FACTS, META, SCOPE, FEB, MAR, fake_cube
    facts = [{**f, "channel": None if i % 4 else f["channel"]} for i, f in enumerate(FACTS)]
    report = await drill.rank_change(SCOPE, META, fake_cube(facts), time_dimension="sales.created_at",
                                     period=MAR, compare_period=FEB)
    assert report.skipped["sales.channel"].startswith("blank for more than")


# ── agent hooks ───────────────────────────────────────────────────────────────

class FakeStats:
    def __init__(self, saved):
        self.saved, self.refreshed = saved, []

    def load_stats(self, schema, tables):
        return {k: v for k, v in self.saved.get(schema, {}).items() if k[0] in tables}

    def refresh_in_background(self, schema):
        self.refreshed.append(schema)
        return True


def test_member_stats_maps_view_dimensions_to_their_table_columns(monkeypatch):
    fake = FakeStats({"app": {("site", "name"): S(1, 8), ("part", "status"): S(1, 4)}})
    monkeypatch.setattr(agent, "_stats_module", lambda: fake)
    out = agent._member_stats(VIEW, drill.alias_tables(SQL))
    assert out == {"v.site_name": S(1, 8), "v.part_status": S(1, 4)}
    assert fake.refreshed == ["app"]     # org and response have no stats yet → background refresh


def test_member_stats_without_any_saved_stats_returns_none_and_refreshes(monkeypatch):
    fake = FakeStats({})
    monkeypatch.setattr(agent, "_stats_module", lambda: fake)
    assert agent._member_stats(VIEW, drill.alias_tables(SQL)) is None
    assert fake.refreshed == ["app"]


def test_a_commit_refreshes_the_schema_its_sql_reads(monkeypatch):
    fake = FakeStats({})
    monkeypatch.setattr(agent, "_stats_module", lambda: fake)
    agent._refresh_stats_for(json.dumps({"name": "mdbl_c_site", "data": {"sql": "SELECT * FROM mdbl.c_site"}}))
    agent._refresh_stats_for('{"name": "x", "data": {}}')
    assert fake.refreshed == ["mdbl"]
