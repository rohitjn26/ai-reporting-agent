"""Driver tree on cells (graph/tree.py): join safety, CHAID splits, mix vs rate."""
import random

import pytest

from graph import tree
from graph.tree import Cells, Stat

DIMS = ["delay", "state", "pay"]


def ratings(pairs):
    """[(key, {score: count})] -> cells with the score distribution, like a rating query."""
    rows = []
    for key, dist in pairs:
        n = sum(dist.values())
        s = sum(v * c for v, c in dist.items())
        ss = sum(v * v * c for v, c in dist.items())
        rows.append((key, Stat(n, s, ss)))
    return Cells(DIMS, rows, rating=True, per="review", total=sum(st.n for _, st in rows))


def spread(mean, n):
    """A 1-5 distribution of n ratings with roughly this mean."""
    lo, hi = math_floor(mean), math_floor(mean) + 1
    w = mean - lo
    return {lo: round(n * (1 - w)), hi: round(n * w)} if hi <= 5 else {5: n}


def math_floor(x):
    return int(x // 1)


# ── join safety ───────────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "reviews", "data": {"name": "reviews", "joins": {"orders": {"relationship": "many_to_one"}}}},
    {"name": "orders", "data": {"name": "orders", "joins": {
        "customers": {"relationship": "many_to_one"}, "items": {"relationship": "one_to_many"},
        "reviews": {"relationship": "one_to_many"}}}},
    {"name": "items", "data": {"name": "items", "joins": {
        "orders": {"relationship": "many_to_one"}, "products": {"relationship": "many_to_one"}}}},
    {"name": "customers", "data": {"name": "customers", "joins": {}}},
    {"name": "products", "data": {"name": "products", "joins": {}}},
]


def test_only_many_to_one_paths_are_safe():
    g = tree.join_graph(CONFIGS)
    safe, fans = tree.safe_cubes("reviews", g)
    assert safe == {"reviews", "orders", "customers"}
    assert fans == {"items": "orders", "products": "orders"}


def test_per_item_grain_is_the_cube_across_the_first_one_to_many_step():
    g = tree.join_graph(CONFIGS)
    assert tree.grain_for("reviews", "products", g) == "items"
    assert tree.grain_for("reviews", "customers", g) is None        # no fan-out on the way
    safe, _ = tree.safe_cubes("items", g)
    assert {"items", "orders", "products", "customers"} <= safe


# ── statistics ────────────────────────────────────────────────────────────────

def test_chi2_tail_matches_tables():
    for x, k in [(3.841, 1), (5.991, 2), (7.815, 3), (9.488, 4), (11.07, 5)]:
        assert tree.chi2_sf(x, k) == pytest.approx(0.05, abs=0.001)


def test_bonferroni_counts_the_ways_to_merge():
    assert tree.bonferroni(4, 2, ordered=True) == 3       # 3 places to cut an ordered list
    assert tree.bonferroni(4, 2, ordered=False) == 7      # Stirling S(4,2)


# ── the tree ──────────────────────────────────────────────────────────────────

def test_finds_the_planted_split_and_ignores_noise():
    random.seed(1)
    pairs = []
    for delay, m in (("late", 2.6), ("on_time", 4.3)):
        for state in ("SP", "RJ", "MG"):
            for pay in ("card", "boleto"):
                pairs.append(((delay, state, pay), spread(m + random.uniform(-0.01, 0.01), 2000)))
    root = tree.grow(ratings(pairs), DIMS, ordered=set())
    assert root.split == "delay" and root.p < 0.001
    assert [b.values for b in root.branches] == [["late"], ["on_time"]]
    assert all(b.node.split is None for b in root.branches)          # state, pay: noise only


def test_categories_that_do_not_differ_are_merged():
    pairs = [((d, s, "card"), spread(m, 3000)) for d, s, m in
             [("x", "SP", 3.0), ("x", "RJ", 3.0), ("x", "MG", 4.5), ("x", "BA", 4.5)]]
    root = tree.grow(ratings(pairs), DIMS, ordered=set())
    assert root.split == "state"
    assert [b.values for b in root.branches] == [["RJ", "SP"], ["BA", "MG"]]


def test_a_real_but_tiny_difference_is_not_a_split():
    # 0.02★ apart on 200k rows: significant, but under the 0.05★ minimum effect
    pairs = [(("x", "SP", "card"), spread(4.00, 100_000)), (("x", "RJ", "card"), spread(4.02, 100_000))]
    root = tree.grow(ratings(pairs), DIMS, ordered=set())
    assert root.split is None


def test_small_branches_are_absorbed():
    pairs = [(("x", "SP", "card"), spread(4.0, 50_000)), (("x", "RJ", "card"), spread(3.0, 50_000)),
             (("x", "AC", "card"), spread(1.0, 100))]          # 0.1% of rows, far off
    root = tree.grow(ratings(pairs), DIMS, ordered=set())
    assert root.split == "state" and len(root.branches) == 2
    assert all(b.node.rows >= 500 for b in root.branches)


def test_ordered_dimensions_only_merge_neighbours():
    # time: 2019 and 2021 alike, 2020 different — they must not merge across 2020
    pairs = [(("x", y, "card"), spread(m, 5000)) for y, m in [("2019", 4.2), ("2020", 3.0), ("2021", 4.2)]]
    root = tree.grow(ratings(pairs), DIMS, ordered={"state"})
    assert sorted(len(b.values) for b in root.branches) == [1, 1, 1]


def test_depth_is_capped():
    pairs = [((d, s, p), spread(1 + (d == "a") * 2 + (s == "a") * 1 + (p == "a") * 0.8, 4000))
             for d in "ab" for s in "ab" for p in "ab"]
    root = tree.grow(ratings(pairs), DIMS, ordered=set(), depth=2)
    leaves = [b2.node for b in root.branches for b2 in b.node.branches]
    assert root.split and all(n.split is None for n in leaves)


def test_without_a_distribution_it_splits_on_effect_alone():
    rows = [(("late", "SP", "card"), Stat(5000, 5000 * 3.0, None)),
            (("on_time", "SP", "card"), Stat(5000, 5000 * 4.0, None))]
    root = tree.grow(Cells(DIMS, rows, rating=False, per="review", total=10000), DIMS, ordered=set())
    assert root.split == "delay" and root.p is None


# ── mix vs rate on cells ──────────────────────────────────────────────────────

def test_make_up_puts_the_gap_on_the_mix_dimension():
    rows = [(("elec", "high", "x"), Stat(80, 240, None)), (("elec", "low", "x"), Stat(20, 100, None)),
            (("home", "high", "x"), Stat(20, 60, None)), (("home", "low", "x"), Stat(80, 400, None))]
    c = Cells(["category", "band", "pay"], rows, rating=False, per="item", total=200)
    out = tree.make_up(c, "category", ["elec"], None, ["band", "pay"])
    assert (out["avg_segment"], out["avg_others"]) == (3.4, 4.6)
    top = out["by_dimension"][0]
    assert top["dimension"] == "band" and top["explained_by_make_up"] == pytest.approx(1.0)
    assert "note" not in out


def test_cells_from_rating_rows_keep_the_distribution():
    rows = [{"d": "a", "w": 3, "score": 5}, {"d": "a", "w": 1, "score": 1}, {"d": None, "w": 2, "score": 4}]
    cells = dict(tree.cells_from_rows(rows, ["d"], "w", "score", "m"))
    assert cells[("a",)].n == 4 and cells[("a",)].mean == 4.0 and cells[("a",)].ss == 76
    assert cells[(tree.BLANK,)].n == 2


def test_a_dimension_that_restates_the_value_is_left_out():
    # the title is written with the score ("Awful" = 1★): it explains everything
    pairs = [((t, "SP", p), {score: 1000}) for t, score in (("Awful", 1), ("Great", 5)) for p in ("card", "pix")]
    pairs += [(("Okay", "RJ", "card"), {3: 1000})]
    c = ratings(pairs)
    echoes = tree.restating(c, ["delay", "state", "pay"])     # "delay" holds the title here
    assert list(echoes) == ["delay"] and "restates" in echoes["delay"]


def test_significant_but_tiny_differences_are_merged():
    # 3.76 / 3.77 / 3.78 on 200k rows each: "significant", but 0.01★ apart — one branch
    pairs = [(("x", s, "card"), spread(m, 200_000)) for s, m in [("a", 3.76), ("b", 3.77), ("c", 3.78), ("d", 4.35)]]
    root = tree.grow(ratings(pairs), DIMS, ordered=set())
    assert [b.values for b in root.branches] == [["a", "b", "c"], ["d"]]
