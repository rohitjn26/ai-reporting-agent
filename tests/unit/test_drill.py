"""The drill engine: candidates from /meta, the drill transform, and ranking —
run against a tiny in-memory fake of Cube so the scores come from real grouping."""
from datetime import date

import pytest

from graph.drill import (OTHER, Candidate, NotAdditive, bucket_range, candidates,
                         drill, rank_breakdown, rank_change)

META = [{
    "name": "sales",
    "measures": [
        {"name": "sales.revenue", "aggType": "sum"},
        {"name": "sales.count", "aggType": "count"},
        {"name": "sales.avg_price", "aggType": "avg"},
    ],
    "dimensions": [
        {"name": "sales.order_id", "type": "number", "primaryKey": True},
        {"name": "sales.customer_id", "type": "string"},
        {"name": "sales.region", "type": "string"},
        {"name": "sales.channel", "type": "string"},
        {"name": "sales.is_gift", "type": "boolean"},
        {"name": "sales.amount", "type": "number"},
        {"name": "sales.created_at", "type": "time"},
    ],
}]

SCOPE = {"measures": ["sales.revenue"], "time_dimensions": [{"dimension": "sales.created_at"}]}
FEB, MAR = ["2026-02-01", "2026-02-28"], ["2026-03-01", "2026-03-31"]


def _row(day, region, channel, revenue, gift=False, customer="c1"):
    return {"created_at": date.fromisoformat(day), "region": region, "channel": channel,
            "is_gift": gift, "customer_id": customer, "revenue": revenue}


# Feb → Mar: total 1000 → 900. Region: West carries the drop (300 → 210 is
# -90 of the -100 although West is 30% of the base). Channel: both halves dip ~evenly.
FACTS = [
    _row("2026-02-10", "West", "Web", 150), _row("2026-02-11", "West", "Store", 150),
    _row("2026-02-12", "East", "Web", 350), _row("2026-02-13", "East", "Store", 350),
    _row("2026-03-10", "West", "Web", 105), _row("2026-03-11", "West", "Store", 105),
    _row("2026-03-12", "East", "Web", 343), _row("2026-03-13", "East", "Store", 347),
]


def fake_cube(facts):
    """Group `facts` the way Cube would for the query shapes the engine sends."""
    queries = []

    async def run(q):
        queries.append(q)
        rows = list(facts)
        for f in q.get("filters", []):
            col = f["member"].split(".")[1]
            rows = [r for r in rows if str(r[col]) in f["values"]]
        keys = [d.split(".")[1] for d in q.get("dimensions", [])]
        for td in q.get("time_dimensions", []):
            if td.get("dateRange"):
                lo, hi = (date.fromisoformat(x) for x in td["dateRange"])
                rows = [r for r in rows if lo <= r["created_at"] <= hi]
            if td.get("granularity"):
                keys.append(("created_at", td["granularity"]))
        groups = {}
        for r in rows:
            k = tuple(r[c] if isinstance(c, str) else _trunc(r["created_at"], c[1]) for c in keys)
            groups[k] = groups.get(k, 0) + (1 if q["measures"][0] == "sales.count" else r["revenue"])
        out = []
        for k, v in groups.items():
            row = {q["measures"][0]: str(v)}
            for c, val in zip(keys, k):
                row["sales." + c if isinstance(c, str) else f"sales.created_at.{c[1]}"] = val
            out.append(row)
        return out[: q.get("limit", 10_000)]

    run.queries = queries
    return run


def _trunc(d, gran):
    if gran == "month":
        return f"{d:%Y-%m}-01T00:00:00.000"
    return f"{d.isoformat()}T00:00:00.000"


# ── candidates ────────────────────────────────────────────────────────────────

def test_candidates_come_from_meta_and_skip_ids_numbers_and_used_members():
    q = {"measures": ["sales.revenue"], "dimensions": ["sales.region"],
         "filters": [{"member": "sales.channel", "operator": "equals", "values": ["Web"]}],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "month"}]}
    assert candidates(q, META) == [Candidate("sales.created_at", "week"), Candidate("sales.is_gift")]


def test_no_time_candidate_below_day():
    q = {"measures": ["sales.revenue"],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "day"}]}
    assert all(not c.is_time for c in candidates(q, META))


# ── drill ─────────────────────────────────────────────────────────────────────

def test_drill_filters_the_clicked_member_and_groups_by_the_next():
    q = {"measures": ["sales.revenue"], "dimensions": ["sales.region"],
         "order": {"sales.region": "asc", "sales.revenue": "desc"}, "limit": 5}
    out = drill(q, {"sales.region": "West"}, Candidate("sales.channel"))
    assert out == {"measures": ["sales.revenue"], "dimensions": ["sales.channel"],
                   "filters": [{"member": "sales.region", "operator": "equals", "values": ["West"]}],
                   "order": {"sales.revenue": "desc"}}


def test_drill_into_a_null_member_uses_notset():
    out = drill({"measures": ["sales.revenue"], "dimensions": ["sales.region"]},
                {"sales.region": None}, Candidate("sales.channel"))
    assert out["filters"] == [{"member": "sales.region", "operator": "notSet"}]


def test_drill_into_a_time_bucket_narrows_the_range_and_steps_finer():
    q = {"measures": ["sales.revenue"],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "month",
                              "dateRange": "this year"}]}
    out = drill(q, {"sales.created_at.month": "2026-02-01T00:00:00.000"},
                Candidate("sales.created_at", "week"))
    assert out["time_dimensions"] == [{"dimension": "sales.created_at", "granularity": "week",
                                       "dateRange": ["2026-02-01", "2026-02-28"]}]


def test_drill_from_a_time_bucket_into_a_dimension_drops_the_time_axis():
    q = {"measures": ["sales.revenue"],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "month"}]}
    out = drill(q, {"sales.created_at.month": "2026-03-01T00:00:00.000"}, Candidate("sales.region"))
    assert out["dimensions"] == ["sales.region"]
    assert out["time_dimensions"] == [{"dimension": "sales.created_at",
                                       "dateRange": ["2026-03-01", "2026-03-31"]}]


@pytest.mark.parametrize("value,gran,expected", [
    ("2024-02-01T00:00:00.000", "month", ["2024-02-01", "2024-02-29"]),
    ("2026-10-01T00:00:00.000", "quarter", ["2026-10-01", "2026-12-31"]),
    ("2026-01-01T00:00:00.000", "year", ["2026-01-01", "2026-12-31"]),
    ("2026-03-02T00:00:00.000", "week", ["2026-03-02", "2026-03-08"]),
    ("2026-03-02", "day", ["2026-03-02", "2026-03-02"]),
])
def test_bucket_range(value, gran, expected):
    assert bucket_range(value, gran) == expected


# ── rank_change ───────────────────────────────────────────────────────────────

async def test_region_outranks_channel_and_west_leads():
    report = await rank_change(SCOPE, META, fake_cube(FACTS),
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB)
    assert (report.base, report.current, report.delta) == (1000, 900, -100)
    top, second = report.rankings[:2]
    assert top.candidate.member == "sales.region"
    assert top.score == pytest.approx(1.2)                     # |−90 − (−30)| + |−10 − (−70)|
    assert second.candidate.member == "sales.channel"
    assert second.score == pytest.approx(0.04)
    west = top.members[0]
    assert (west.value, west.delta, west.contribution) == ("West", -90, pytest.approx(0.9))
    assert west.surprise == pytest.approx(-60)


async def test_skips_members_with_too_many_values_and_single_member_dims():
    facts = [_row("2026-02-01", "West", "Web", 10, customer=f"x{i}") for i in range(5)] + \
            [_row("2026-03-01", "West", "Web", 5, customer=f"y{i}") for i in range(5)]
    meta = [{**META[0], "dimensions": META[0]["dimensions"] + [
        {"name": "sales.customer", "type": "string"}]}]
    report = await rank_change(SCOPE, meta, fake_cube([{**f, "customer": f["customer_id"]} for f in facts]),
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB,
                               max_members=3)
    assert report.skipped["sales.customer"] == "more than 3 members"
    assert report.skipped["sales.region"] == "only one member"


async def test_new_member_carries_its_whole_delta_as_surprise():
    facts = FACTS + [_row("2026-03-20", "North", "Web", 200)]
    report = await rank_change(SCOPE, META, fake_cube(facts),
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB)
    region = next(r for r in report.rankings if r.candidate.member == "sales.region")
    north = next(m for m in region.members if m.value == "North")
    assert (north.base, north.delta, north.surprise) == (0, 200, 200)


async def test_tiny_members_fold_into_other():
    facts = FACTS + [_row("2026-02-14", f"Tiny{i}", "Web", 1) for i in range(3)]
    report = await rank_change(SCOPE, META, fake_cube(facts),
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB)
    region = next(r for r in report.rankings if r.candidate.member == "sales.region")
    assert {m.value for m in region.members} == {"West", "East", OTHER}


async def test_a_nested_finer_split_does_not_beat_its_parent():
    # customers sit inside regions; West's drop is spread over many West customers
    facts = [_row("2026-02-01", "West", "Web", 100, customer=f"w{i}") for i in range(6)] + \
            [_row("2026-02-01", "East", "Web", 100, customer=f"e{i}") for i in range(6)] + \
            [_row("2026-03-01", "West", "Web", 50, customer=f"w{i}") for i in range(6)] + \
            [_row("2026-03-01", "East", "Web", 100, customer=f"e{i}") for i in range(6)]
    meta = [{**META[0], "dimensions": META[0]["dimensions"] + [
        {"name": "sales.customer", "type": "string"}]}]
    report = await rank_change(SCOPE, meta, fake_cube([{**f, "customer": f["customer_id"]} for f in facts]),
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB)
    assert report.rankings[0].candidate.member == "sales.region"


async def test_a_failing_probe_is_reported_not_mistaken_for_empty():
    cube = fake_cube(FACTS)

    async def run(q):
        if "sales.channel" in q.get("dimensions", []):
            raise RuntimeError('column "day_count" does not exist')
        return await cube(q)

    report = await rank_change(SCOPE, META, run,
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB)
    assert report.skipped["sales.channel"].startswith("query failed: column")
    breakdown = await rank_breakdown({"measures": ["sales.revenue"]}, META, run)
    assert breakdown.skipped["sales.channel"].startswith("query failed")


async def test_ratio_measures_are_refused():
    with pytest.raises(NotAdditive):
        await rank_change({**SCOPE, "measures": ["sales.avg_price"]}, META, fake_cube(FACTS),
                          time_dimension="sales.created_at", period=MAR, compare_period=FEB)


async def test_flat_total_with_offsetting_moves_still_scores():
    facts = [_row("2026-02-01", "West", "Web", 100), _row("2026-02-01", "East", "Web", 100),
             _row("2026-03-01", "West", "Web", 50), _row("2026-03-01", "East", "Web", 150)]
    report = await rank_change(SCOPE, META, fake_cube(facts),
                               time_dimension="sales.created_at", period=MAR, compare_period=FEB)
    assert report.delta == 0
    region = report.rankings[0]
    assert region.candidate.member == "sales.region" and region.score == pytest.approx(0.5)
    assert all(m.contribution is None for m in region.members)


# ── rank_breakdown ────────────────────────────────────────────────────────────

async def test_breakdown_prefers_the_concentrated_dimension():
    # region: West 900 / East 100 (concentrated); channel: Web 500 / Store 500 (even)
    facts = [_row("2026-03-01", "West", "Web", 450), _row("2026-03-01", "West", "Store", 450),
             _row("2026-03-02", "East", "Web", 50), _row("2026-03-02", "East", "Store", 50)]
    report = await rank_breakdown({"measures": ["sales.revenue"]}, META, fake_cube(facts))
    assert [r.candidate.member for r in report.rankings][:2] == ["sales.region", "sales.channel"]
    assert report.rankings[1].score == pytest.approx(0)       # 50/50 split says nothing
    assert report.rankings[0].members[0].value == "West"


async def test_breakdown_of_a_ratio_lists_candidates_unscored():
    report = await rank_breakdown({"measures": ["sales.avg_price"]}, META, fake_cube(FACTS))
    assert report.rankings and all(r.score is None for r in report.rankings)
    assert [r.candidate.member for r in report.rankings] == ["sales.region", "sales.channel"]


async def test_breakdown_time_candidate_keeps_buckets_in_order():
    q = {"measures": ["sales.revenue"],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "month"}]}
    report = await rank_breakdown(q, META, fake_cube(FACTS))
    week = next(r for r in report.rankings if r.candidate.is_time)
    assert week.candidate == Candidate("sales.created_at", "week")
    values = [m.value for m in week.members]
    assert values == sorted(values)
