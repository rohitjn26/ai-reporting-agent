"""Averages in the drill engine: mix vs rate between two groups, and spread within one.
Cube is faked by grouping a list of review rows."""
import asyncio

import pytest

from graph import drill

META = [{"name": "v", "measures": [
    {"name": "v.avg_score", "aggType": "avg", "aliasMember": "reviews.avg_score"},
    {"name": "v.count", "aggType": "count", "aliasMember": "reviews.count"},
    {"name": "v.total", "aggType": "sum", "aliasMember": "reviews.total"}],
    "dimensions": [{"name": f"v.{d}", "type": "string", "aliasMember": f"reviews.{d}"}
                   for d in ("category", "seller", "price_band")]}]


def fake_cube(rows):
    def keep(r, f):
        v = r[f["member"].split(".")[1]]
        return v in f["values"] if f["operator"] == "equals" else v not in f["values"]

    async def run(q):
        picked = [r for r in rows if all(keep(r, f) for f in q.get("filters", []))]
        groups = {}
        for r in picked:
            key = tuple(r[d.split(".")[1]] for d in q.get("dimensions", []))
            groups.setdefault(key, []).append(r["score"])
        return [{**dict(zip(q.get("dimensions", []), k)),
                 "v.avg_score": sum(s) / len(s), "v.count": len(s)} for k, s in groups.items()]
    return run


def rows(category, seller, band, score, n):
    return [{"category": category, "seller": seller, "price_band": band, "score": score}] * n


def scopes(segment="electronics"):
    base = {"measures": ["v.avg_score"], "filters": []}
    f = lambda op: {**base, "filters": [{"member": "v.category", "operator": op, "values": [segment]}]}
    return base, f("notEquals"), f("equals")


def test_a_gap_from_a_different_mix_is_put_on_that_dimension():
    # pricey items score 3, cheap ones 5 — in every category and with every seller;
    # electronics is lower only because it sells mostly pricey items
    data = (rows("electronics", "S1", "high", 3, 40) + rows("electronics", "S2", "high", 3, 40)
            + rows("electronics", "S1", "low", 5, 10) + rows("electronics", "S2", "low", 5, 10)
            + rows("home", "S1", "high", 3, 10) + rows("home", "S2", "high", 3, 10)
            + rows("home", "S1", "low", 5, 40) + rows("home", "S2", "low", 5, 40))
    scope, a, b = scopes()
    rep = asyncio.run(drill.rank_average_gap(scope, META, fake_cube(data), a_scope=a, b_scope=b,
                                             exclude=("v.category",)))
    assert (rep.avg_a, rep.avg_b) == (4.6, 3.4) and rep.gap == pytest.approx(-1.2)
    top = rep.rankings[0]
    assert top.candidate.member == "v.price_band" and top.score == pytest.approx(1.0)
    assert top.mix == pytest.approx(-1.2) and top.rate == pytest.approx(0.0)
    assert rep.note is None
    assert "v.seller" in rep.skipped or rep.rankings[-1].score < 0.01    # sellers explain nothing


def test_a_gap_with_the_same_mix_is_called_across_the_board():
    data = (rows("electronics", "S1", "high", 3, 50) + rows("electronics", "S2", "low", 4, 50)
            + rows("home", "S1", "high", 4, 50) + rows("home", "S2", "low", 5, 50))
    scope, a, b = scopes()
    rep = asyncio.run(drill.rank_average_gap(scope, META, fake_cube(data), a_scope=a, b_scope=b,
                                             exclude=("v.category",)))
    assert rep.gap == pytest.approx(-1.0)
    assert all(r.score < 0.01 for r in rep.rankings)
    assert all(r.rate == pytest.approx(-1.0) for r in rep.rankings)
    assert "across the board" in rep.note


def test_totals_and_ratios_are_refused():
    scope, a, b = scopes()
    for m in ("v.total", "v.count"):
        with pytest.raises(drill.NotAdditive):
            asyncio.run(drill.rank_average_gap({**scope, "measures": [m]}, META, fake_cube([]),
                                               a_scope=a, b_scope=b))
    assert drill.weight_measure("v.avg_score", META) == "v.count"
