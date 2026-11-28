"""
Driver tree for an average: what it depends on, found automatically (no LLM).

One Cube query fetches *cells*: the weight (row count) and the sum of the averaged
value for every combination of a few candidate dimensions. An average is sum ÷ count
and both add up, so any group's average is exact from its cells, and a CHAID-style
tree grown on cells weighted by their counts splits exactly as it would on raw rows.
Plan and reasoning: docs/plans/driver_tree.md.

    safe_cubes(start, joins)              # cubes reachable without double counting
    cells = await fetch_cells(...)        # one query, cached in memory
    grow(cells, dims, kinds)              # the tree
    make_up(cells, ...)                   # why one group's average differs (mix vs rate)

Candidates come from the data model: a dimension is safe when every join step from
the measure's cube is many_to_one / one_to_one (each row has exactly one of it).
Past a one_to_many step it would count rows several times, so it is left out —
unless the question is per item: then the grain moves to the cube across that step.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from graph import drill

Run = Callable[[dict], Awaitable[list[dict]]]

# ── settings (env-overridable) ────────────────────────────────────────────────
ALPHA = float(os.environ.get("TREE_ALPHA", "0.10"))            # split / merge significance
# smallest change in the average worth a split: this share of the value's standard
# deviation (≈0.06★ on a 1-5 rating, ≈2 points on a 0/1 rate) — or, when the spread
# is unknown, this share of the mean
MIN_EFFECT_SD = float(os.environ.get("TREE_MIN_EFFECT_SD", "0.05"))
MIN_EFFECT_SHARE = float(os.environ.get("TREE_MIN_EFFECT_SHARE", "0.02"))
MIN_CHILD_SHARE = float(os.environ.get("TREE_MIN_CHILD_SHARE", "0.01"))
MIN_CHILD_ROWS = int(os.environ.get("TREE_MIN_CHILD_ROWS", "500"))
MAX_DEPTH = int(os.environ.get("TREE_MAX_DEPTH", "3"))
MIN_BRANCH = int(os.environ.get("TREE_MIN_BRANCH", "30"))     # rows, on any data size
MAX_CANDIDATES = 8
CELL_LIMIT = 50_000            # Cube's default row cap; a full result means "too many cells"
CELL_BUDGET = 5_000_000        # estimated combinations allowed: real ones are far fewer,
                               # and a full result just drops one dimension and retries
SAMPLE_ABOVE = int(os.environ.get("TREE_SAMPLE_ABOVE", "2000000"))    # rows; sample past this
SAMPLE_TARGET = int(os.environ.get("TREE_SAMPLE_TARGET", "1000000"))  # about this many rows
RATING_MAX_VALUES = 11         # a value with ≤ this many distinct values is a rating scale
BLANK = "(blank)"

_MANY_TO_ONE = {"many_to_one", "belongsto", "belongs_to", "one_to_one", "hasone", "has_one"}
_ONE_TO_MANY = {"one_to_many", "hasmany", "has_many"}


# ── join safety ───────────────────────────────────────────────────────────────

def join_graph(configs: list[dict]) -> dict[str, dict[str, str]]:
    """{cube: {joined cube: 'one' | 'many' | '?'}} from library cube configs, both
    directions: a declared many_to_one A→B means B→A is one_to_many. 'one' = each
    row has at most one; '?' = relationship not declared."""
    g: dict[str, dict[str, str]] = {}
    for cfg in configs:
        a = (cfg.get("data") or {}).get("name") or cfg.get("name")
        for b, j in ((cfg.get("data") or {}).get("joins") or {}).items():
            rel = str((j or {}).get("relationship", "")).lower()
            kind = "one" if rel in _MANY_TO_ONE else "many" if rel in _ONE_TO_MANY else "?"
            g.setdefault(a, {})[b] = kind
            back = {"one": "many", "many": "one"}.get(kind, "?")
            if rel in ("one_to_one", "hasone", "has_one"):
                back = "one"
            g.setdefault(b, {}).setdefault(a, back)
    return g


def safe_cubes(start: str, graph: dict[str, dict[str, str]]) -> tuple[set[str], dict[str, str]]:
    """-> (cubes whose rows each `start` row has at most one of, {other cube: the cube
    past which it fans out}). Unknown relationships count as safe; the cells' own
    add-up check catches them if they aren't."""
    safe, todo = {start}, deque([start])
    while todo:
        c = todo.popleft()
        for n, kind in graph.get(c, {}).items():
            if kind in ("one", "?") and n not in safe:
                safe.add(n)
                todo.append(n)
    fans = {}
    for c in list(safe):
        for n, kind in graph.get(c, {}).items():
            if kind == "many" and n not in safe:
                fans.setdefault(n, c)
    # everything reachable beyond a fan-out step is fan-out too
    todo = deque(fans)
    while todo:
        c = todo.popleft()
        for n in graph.get(c, {}):
            if n not in safe and n not in fans:
                fans[n] = fans[c]
                todo.append(n)
    return safe, fans


def hops(start: str, graph: dict[str, dict[str, str]]) -> dict[str, int]:
    """Join steps from `start` to every reachable cube."""
    dist, todo = {start: 0}, deque([start])
    while todo:
        c = todo.popleft()
        for n in graph.get(c, {}):
            if n not in dist:
                dist[n] = dist[c] + 1
                todo.append(n)
    return dist


def grain_for(start: str, target: str, graph: dict[str, dict[str, str]]) -> Optional[str]:
    """The cube across the first one_to_many step on the way from `start` to `target`
    (order_items, for reviews → products). Per item, that cube's rows are the grain."""
    prev = {start: None}
    todo = deque([start])
    while todo:
        c = todo.popleft()
        if c == target:
            break
        for n in graph.get(c, {}):
            if n not in prev:
                prev[n] = c
                todo.append(n)
    if target not in prev:
        return None
    path, c = [], target
    while c is not None:
        path.append(c)
        c = prev[c]
    path.reverse()
    for a, b in zip(path, path[1:]):
        if graph.get(a, {}).get(b) == "many":
            return b
    return None


# ── cells ─────────────────────────────────────────────────────────────────────

@dataclass
class Stat:
    n: float = 0.0
    s: float = 0.0
    ss: Optional[float] = 0.0       # None when the value's spread within a cell is unknown

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    @classmethod
    def of(cls, *parts: "Stat") -> "Stat":
        out = cls(ss=0.0 if all(p.ss is not None for p in parts) else None)
        for p in parts:
            out.add(p)
        return out

    def add(self, other: "Stat") -> None:
        self.n += other.n
        self.s += other.s
        self.ss = None if self.ss is None or other.ss is None else self.ss + other.ss


@dataclass
class Cells:
    dims: list[str]                 # the cell key's dimensions, in order
    rows: list[tuple[tuple, Stat]]  # (key values, stat)
    rating: bool                    # the value's spread inside cells is known (a rating's
                                    # distribution, or the avg's sum-of-squares companion)
    per: str                        # what one counted row is: the grain cube's title ("Order Items")
    total: float                    # the weight's total, to check the cells add up
    grain: str = ""                 # the cube whose rows are counted
    fetched_at: float = field(default_factory=time.time)

    def stat(self) -> Stat:
        out = Stat(ss=0.0 if self.rating else None)
        for _, st in self.rows:
            out.add(st)
        return out


def cells_from_rows(rows: list[dict], dims: list[str], weight: str, value: Optional[str],
                    measure: str, spread: Optional[tuple[str, str]] = None) -> list[tuple[tuple, Stat]]:
    """Fold Cube rows into cells. With `value` (a per-row dimension, e.g. the score)
    each row is one value of a cell's distribution. Otherwise rows carry the cell's
    average, and with `spread` = (rows-with-a-value measure, sum-of-squares measure)
    the spread inside the cell is known too; without it, it is unknown."""
    out: dict[tuple, Stat] = {}
    for r in rows:
        key = tuple(BLANK if r.get(d) is None else r.get(d) for d in dims)
        n = drill._num(r.get(weight))
        if value:
            if r.get(value) is None:
                continue
            v = drill._num(r.get(value))
            st = Stat(n, v * n, v * v * n)
        else:
            if r.get(measure) is None:
                continue
            if spread:
                n = drill._num(r.get(spread[0]))
                st = Stat(n, drill._num(r.get(measure)) * n, drill._num(r.get(spread[1])))
            else:
                st = Stat(n, drill._num(r.get(measure)) * n, None)
        out.setdefault(key, Stat(ss=0.0 if value or spread else None)).add(st)
    return list(out.items())


# shortcut: per-process memory, last 20 cell sets for 30 min; fine for one user, not
# shared across server processes — move to Redis or the DB if the UI ever scales out.
_CACHE: dict[tuple, Cells] = {}
CACHE_KEEP, CACHE_TTL = 20, 30 * 60


def cache_get(key: tuple) -> Optional[Cells]:
    hit = _CACHE.get(key)
    if hit and time.time() - hit.fetched_at < CACHE_TTL:
        return hit
    _CACHE.pop(key, None)
    return None


def cache_put(key: tuple, cells: Cells) -> None:
    _CACHE[key] = cells
    while len(_CACHE) > CACHE_KEEP:
        _CACHE.pop(next(iter(_CACHE)))


def clear_cache() -> None:
    """Called when a cube config changes: cells may describe the old model."""
    _CACHE.clear()


# ── statistics ────────────────────────────────────────────────────────────────

def chi2_sf(x: float, k: int) -> float:
    """P(χ²_k ≥ x) for integer k (exact series)."""
    if x <= 0:
        return 1.0
    h = x / 2
    if k % 2 == 0:
        term, total = math.exp(-h), 0.0
        for i in range(k // 2):
            total += term
            term *= h / (i + 1)
        return min(1.0, total)
    total = math.erfc(math.sqrt(h))
    term = math.exp(-h) * math.sqrt(h) / math.gamma(1.5)
    for i in range(1, (k - 1) // 2 + 1):
        total += term
        term *= h / (i + 0.5)
    return min(1.0, total)


def _within_var(groups: list[Stat]) -> Optional[float]:
    if any(g.ss is None for g in groups):
        return None
    n = sum(g.n for g in groups)
    ssw = sum(g.ss - g.n * g.mean ** 2 for g in groups)
    return ssw / (n - len(groups)) if n > len(groups) and ssw > 0 else None


def anova_p(groups: list[Stat]) -> Optional[float]:
    """p-value that the groups' means differ (one-way ANOVA; with groups of hundreds of
    rows, F·(k−1) is χ²(k−1)). None when the spread inside groups is unknown."""
    var = _within_var(groups)
    if var is None or len(groups) < 2:
        return None
    n = sum(g.n for g in groups)
    mean = sum(g.s for g in groups) / n
    ssb = sum(g.n * (g.mean - mean) ** 2 for g in groups)
    return chi2_sf(ssb / var, len(groups) - 1)


def _stirling2(n: int, k: int) -> int:
    row = [1] + [0] * k
    for i in range(1, n + 1):
        new = [0] * (k + 1)
        for j in range(1, min(i, k) + 1):
            new[j] = j * row[j] + row[j - 1]
        row = new
    return row[k]


def bonferroni(categories: int, groups: int, ordered: bool) -> int:
    """How many ways `categories` could have been merged into `groups` (CHAID)."""
    if ordered:
        return math.comb(categories - 1, groups - 1)
    return _stirling2(categories, groups)


# ── the tree ──────────────────────────────────────────────────────────────────

@dataclass
class Branch:
    values: list
    node: "Node"


@dataclass
class Node:
    rows: float
    avg: float
    split: Optional[str] = None
    p: Optional[float] = None
    branches: list[Branch] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {"rows": round(self.rows), "avg": round(self.avg, 3)}
        if self.split:
            out["split_by"] = self.split
            if self.p is not None:
                out["p"] = float(f"{self.p:.2g}")
            out["branches"] = [{"values": b.values, **b.node.to_dict()} for b in self.branches]
        return out


def _group(cells: list[tuple[tuple, Stat]], i: int, rating: bool) -> dict:
    out: dict = {}
    for key, st in cells:
        out.setdefault(key[i], Stat(ss=0.0 if rating else None)).add(st)
    return out


def _sortable(v):
    return (v == BLANK, str(v))


def merge_categories(groups: dict, ordered: bool, min_effect: float, alpha: float = ALPHA) -> list[tuple[list, Stat]]:
    """CHAID merging: repeatedly join the two most alike categories (adjacent ones for an
    ordered dimension such as time) while they don't differ significantly, or their
    averages are closer than half the minimum effect (with ~1M rows, 0.01★ is
    "significant" but too small to keep apart)."""
    items = sorted(([[k], g] for k, g in groups.items()), key=lambda kv: _sortable(kv[0][0]))
    while len(items) > 1:
        pairs = [(i, i + 1) for i in range(len(items) - 1)] if ordered else \
            [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))]

        def alike(ij):
            a, b = items[ij[0]][1], items[ij[1]][1]
            if abs(a.mean - b.mean) < min_effect / 2:   # too close to matter, however "significant"
                return 1.0
            p = anova_p([a, b])
            return p if p is not None else 0.0

        i, j = max(pairs, key=alike)
        if alike((i, j)) <= alpha:
            break
        items[i] = [items[i][0] + items[j][0], Stat.of(items[i][1], items[j][1])]
        del items[j]
    return [(k, g) for k, g in items]


def _absorb_small(groups: list[tuple[list, Stat]], min_rows: float, ordered: bool) -> list:
    """Merge any group under min_rows into its closest neighbour (by average; for an
    ordered dimension, the adjacent one), smallest first."""
    groups = [(list(k), g) for k, g in groups]
    while len(groups) > 1:
        i = min(range(len(groups)), key=lambda x: groups[x][1].n)
        if groups[i][1].n >= min_rows:
            break
        near = [x for x in ((i - 1, i + 1) if ordered else range(len(groups))) if 0 <= x < len(groups) and x != i]
        j = min(near, key=lambda x: abs(groups[x][1].mean - groups[i][1].mean))
        groups[j] = (groups[j][0] + groups[i][0], Stat.of(groups[j][1], groups[i][1]))
        del groups[i]
    return groups


def min_effect(stat: Stat) -> float:
    """The smallest change in the average worth reporting, in the value's own units."""
    if stat.ss is not None and stat.n:
        sd = math.sqrt(max(0.0, stat.ss / stat.n - stat.mean ** 2))
        if sd > 0:
            return MIN_EFFECT_SD * sd
    return MIN_EFFECT_SHARE * abs(stat.mean)


@dataclass
class Split:
    dim: str
    p: Optional[float]
    effect: float
    groups: list[tuple[list, Stat]]


def best_split(cells: list[tuple[tuple, Stat]], dims: list[str], usable: list[str],
               ordered: set[str], rating: bool, eff: float) -> Optional[Split]:
    """The most significant split of these cells that is real (p < ALPHA), big enough to
    care about (minimum effect) and has no tiny branch; None when nothing qualifies."""
    parent = Stat(ss=0.0 if rating else None)
    for _, st in cells:
        parent.add(st)
    # 500 rows per branch, or a tenth of the node on small data — never under MIN_BRANCH
    # rows, where one or two odd rows would decide the split
    min_rows = max(MIN_CHILD_SHARE * parent.n, min(MIN_CHILD_ROWS, 0.1 * parent.n), MIN_BRANCH)
    best = None
    for d in usable:
        i = dims.index(d)
        raw = _group(cells, i, rating)
        if len(raw) < 2:
            continue
        groups = _absorb_small(merge_categories(raw, d in ordered, eff), min_rows, d in ordered)
        if len(groups) < 2:
            continue
        effect = max(abs(g.mean - parent.mean) for _, g in groups)
        if effect < eff:
            continue
        p = anova_p([g for _, g in groups])
        if p is not None:
            p = min(1.0, p * bonferroni(len(raw), len(groups), d in ordered))
            if p >= ALPHA:
                continue
        cand = Split(d, p, effect, groups)
        rank = lambda s: (s.p if s.p is not None else 1.0, -s.effect)
        if best is None or rank(cand) < rank(best):
            best = cand
    return best


# A dimension that alone explains this much of the value's variation restates it
# (a review's comment title next to its score) rather than explaining it.
RESTATES = float(os.environ.get("TREE_RESTATES", "0.8"))


def restating(cells: Cells, candidates: list[str]) -> dict[str, str]:
    """{dimension: why} for candidates that restate the measure: share of the total
    variation between their members (η²) ≥ RESTATES. Needs the value's distribution."""
    if not cells.rating:
        return {}
    total = cells.stat()
    sst = total.ss - total.n * total.mean ** 2 if total.ss is not None else 0.0
    out = {}
    for d in candidates:
        groups = _group(cells.rows, cells.dims.index(d), True).values()
        ssb = sum(g.n * (g.mean - total.mean) ** 2 for g in groups)
        if sst > 0 and ssb / sst >= RESTATES:
            out[d] = (f"restates the measure itself (alone explains {ssb / sst:.0%} of it), "
                      "so it can't explain it")
    return out


def grow(cells: Cells, candidates: list[str], ordered: set[str], depth: int = MAX_DEPTH) -> Node:
    """The driver tree over `candidates` (a subset of cells.dims)."""
    eff = min_effect(cells.stat())   # one threshold for the whole tree, from the root

    def build(rows, used: list[str], level: int) -> Node:
        st = Stat(ss=0.0 if cells.rating else None)
        for _, s in rows:
            st.add(s)
        node = Node(st.n, st.mean)
        if level >= depth:
            return node
        sp = best_split(rows, cells.dims, [c for c in candidates if c not in used], ordered,
                        cells.rating, eff)
        if sp is None:
            return node
        node.split, node.p = sp.dim, sp.p
        i = cells.dims.index(sp.dim)
        for values, g in sorted(sp.groups, key=lambda vg: vg[1].mean):
            keep = set(map(str, values))
            sub = [(k, s) for k, s in rows if str(k[i]) in keep]
            node.branches.append(Branch(sorted(values, key=_sortable), build(sub, used + [sp.dim], level + 1)))
        return node
    return build(cells.rows, [], 0)


# ── why one group differs (mix vs rate, on cells) ────────────────────────────

def make_up(cells: Cells, compare_dim: str, segment: list, others: Optional[list],
            candidates: list[str]) -> dict:
    """Split the gap between the segment's average and the others' into make-up (the
    segment has more of some members) and within-member differences, per candidate —
    the symmetric Kitagawa split, as in drill.rank_average_gap, but on cells."""
    ci = cells.dims.index(compare_dim)
    seg = {str(v) for v in segment}
    oth = {str(v) for v in others} if others else None
    in_seg = [(k, s) for k, s in cells.rows if str(k[ci]) in seg]
    in_oth = [(k, s) for k, s in cells.rows if (str(k[ci]) in oth if oth else str(k[ci]) not in seg)]

    def total(rows):
        t = Stat(ss=None)
        for _, s in rows:
            t.add(s)
        return t

    a, b = total(in_oth), total(in_seg)
    gap = b.mean - a.mean
    out = {"avg_segment": b.mean, "avg_others": a.mean, "gap": gap,
           "rows_segment": b.n, "rows_others": a.n, "by_dimension": []}
    for d in candidates:
        i = cells.dims.index(d)
        ga, gb = _group(in_oth, i, False), _group(in_seg, i, False)
        mix = rate = 0.0
        members = []
        for k in set(ga) | set(gb):
            sa, sb = ga.get(k, Stat()), gb.get(k, Stat())
            pa, pb = sa.n / (a.n or 1), sb.n / (b.n or 1)
            ra = sa.mean if sa.n else sb.mean
            rb = sb.mean if sb.n else sa.mean
            m, r = (pb - pa) * (ra + rb) / 2, (rb - ra) * (pa + pb) / 2
            mix, rate = mix + m, rate + r
            members.append({"value": k, "share_segment": pb, "share_others": pa,
                            "avg_segment": rb if sb.n else None, "avg_others": ra if sa.n else None,
                            "part_of_gap": m + r})
        members.sort(key=lambda x: -abs(x["part_of_gap"]))
        out["by_dimension"].append({"dimension": d, "explained_by_make_up": abs(mix) / (abs(gap) or 1.0),
                                    "make_up_part": mix, "within_member_part": rate, "members": members[:5]})
    out["by_dimension"].sort(key=lambda x: -x["explained_by_make_up"])
    if not out["by_dimension"] or out["by_dimension"][0]["explained_by_make_up"] < drill.MIX_EXPLAINS:
        out["note"] = ("no dimension's make-up explains much of the gap: the segment differs "
                       "across the board, not because of a different mix")
    return out


# ── choosing what to fetch ────────────────────────────────────────────────────

def value_dimension(measure: str, view_meta: list[dict], configs: list[dict]) -> Optional[str]:
    """The per-row dimension holding what an avg measure averages (review_score for
    avg_review_score): same cube, same SQL."""
    view = measure.split(".")[0]
    ms = drill._view_members(view_meta, view, "measures")
    alias = next((m.get("aliasMember") for m in ms if m["name"] == measure), None)
    if not alias:
        return None
    cube, cm = alias.split(".", 1)
    data = next(((c.get("data") or {}) for c in configs
                 if ((c.get("data") or {}).get("name") or c.get("name")) == cube), {})
    sql = ((data.get("measures") or {}).get(cm) or {}).get("sql")
    col = next((k for k, d in (data.get("dimensions") or {}).items() if sql and d.get("sql") == sql), None)
    if not col:
        return None
    return next((d["name"] for d in drill._view_members(view_meta, view, "dimensions")
                 if d.get("aliasMember") == f"{cube}.{col}"), None)


def sample_share(total_rows: float) -> int:
    """Percent of rows to keep (1-100): all of them up to SAMPLE_ABOVE, else ≈SAMPLE_TARGET."""
    if total_rows <= SAMPLE_ABOVE:
        return 100
    return max(1, min(100, math.ceil(100 * SAMPLE_TARGET / total_rows)))


def cube_title(cube: str, view_meta: list[dict], configs: list[dict]) -> str:
    """What one row of `cube` is called ("Order Items"): the view's carried cube titles,
    then the config's title, then the name."""
    for v in view_meta:
        for c in ((v.get("meta") or {}).get("cubes") or []):
            if c.get("name") == cube and c.get("title"):
                return c["title"]
    data = next(((c.get("data") or {}) for c in configs
                 if ((c.get("data") or {}).get("name") or c.get("name")) == cube), {})
    return data.get("title") or cube


def count_measure(view: str, cube: str, view_meta: list[dict]) -> Optional[str]:
    return next((m["name"] for m in drill._view_members(view_meta, view, "measures")
                 if m.get("aggType") == "count" and drill._source_cube(m) == cube), None)


def pick_candidates(ordered_dims: list[str], view_meta: list[dict], safe: set[str], fans: dict[str, str],
                    stats: Optional[dict], extra_cells: int) -> tuple[list[str], dict[str, str]]:
    """Safe dimensions in the given order, as many as fit the cell budget. -> (picked,
    {left-out dimension: why})."""
    by_name = {d["name"]: d for c in view_meta for d in c.get("dimensions", [])}
    picked, skipped, est = [], {}, max(1, extra_cells)
    for name in ordered_dims:
        cube = drill._source_cube(by_name[name]) if name in by_name else None
        if cube in fans:
            skipped[name] = f"would count rows more than once ({fans[cube]} → {cube} is one-to-many)"
            continue
        if cube not in safe:
            continue
        distinct = getattr((stats or {}).get(name), "distinct", None) or 10   # no stats yet: assume 10
        if len(picked) >= MAX_CANDIDATES or est * distinct > CELL_BUDGET:
            skipped[name] = "left out to keep the data small (others ranked first)"
            continue
        picked.append(name)
        est *= distinct
    return picked, skipped
