# Plan: driver tree for averages

Status: draft for review — nothing built yet. Replaces the flat version on this branch
(`compare_segments`, `key_drivers`), keeping its maths and tests where they still apply.

## Goal

Answer "what explains this average?" automatically, the way an analyst would: find the
dimension that splits the data most, then the next split inside each part, and stop when
nothing meaningful is left. The agent and the engine choose the dimensions; the user gives
no input beyond the question.

Questions it should answer:
- "What explains the low ratings?" — what an average depends on, within one group.
- "Why does electronics rate lower than the rest?" — what explains a gap between two groups.
- "When did the low ratings come in?" — time as one of the candidate splits.

Example output the agent narrates:

```
All reviews        4.05★  (933k)
├── delivery: late      2.90★ (110k)
│   └── seller state: SP   2.60★ (60k)
└── delivery: on time   4.20★ (823k)   — nothing else moves it
```

## How it works

1. **Pick candidate dimensions** from the data model (see "Candidates").
2. **One Cube query** groups by all candidates at once and returns, per combination (a
   *cell*), the row count and the sum — plus, for ratings, the count per score.
3. **Keep the cells in memory** so follow-up questions need no new queries.
4. **Grow a CHAID-style tree** on the cells: pick the most significant split, merge
   categories that don't differ, repeat inside each branch, stop when nothing is
   significant.
5. **Hand the agent a compact tree** to narrate, with a chart query for the top split.

An average is sum ÷ count, and both add up, so any group's average is exact: add its
cells' sums and counts. Splitting on cells weighted by their counts gives the same splits
as running on every raw row.

## Candidates

### Safe joins only (no double counting)

Start at the measure's cube and walk the joins declared in the cube configs (library
`joins.*.relationship`). A dimension is safe when every step on the way is `many_to_one`
or `one_to_one` — each row has exactly one of it. After the first `one_to_many` step,
every dimension further on counts each row several times, so it is left out.

us_ecom, for review ratings:

| Path | Relationship | Use? |
|---|---|---|
| reviews → orders | many_to_one | yes |
| orders → customers | many_to_one | yes |
| orders → order_items → products, sellers | one_to_many | no — 2.2 items per order |
| orders → payments | one_to_many | no |

When a relationship is missing, keep the dimension but rely on the existing check (do the
parts add up to the whole?) to drop it if it double counts.

When a dimension is excluded for this reason, the result says so ("product category:
a review's order can hold items of several categories") and the per-item mode below can
answer the question instead.

### Order and filters

- Nearest cubes first (existing `cube_distances`), each cube's best few first (existing
  `PER_CUBE`).
- Column stats (existing): at least 70% filled, 2–50 distinct values, no ids or JSON.
- Only category-like dimensions (string, boolean). Numbers need bands first — out of scope
  for v1 (the per-row dimensions from PR #50 make bands possible later).
- Time: one time bucket as a candidate, at the granularity that gives at most ~24 buckets
  over the data's range (e.g. month for two years). This answers "when".

### Size cap

The cells query must stay under ~50k rows. Estimate rows as the product of the candidates'
distinct counts (from stats); drop the furthest / most-distinct candidates until the
estimate fits. Query with `limit 50001`; if it comes back full, drop one more and retry
once. Typically 6–8 candidates fit.

## The cells query

```
measures:   count, sum of the averaged column*    (*see below)
dimensions: the candidates  (+ the score itself, for ratings)
filters:    the question's filters
```

Cube gives us averages, not sums. For an `avg` measure we use the same cube's `count`
(already found by `weight_measure`) and recover sum = avg × count per cell — exact per cell.

For ratings (a small set of values, e.g. 1–5) the score itself is added as a grouping
column, which gives the full distribution per cell — needed for the significance test.
For other averages (price, duration) there's no distribution; those fall back to minimum
size and minimum effect rules instead of a significance test.

## Memory

- Cells are cached in the UI process, keyed by (view, measure, filters, candidates).
- Last ~20 entries kept; dropped after 30 minutes or when a cube config changes / Cube
  reloads.
- At ~50k rows × ~10 columns, a few MB each.
- `shortcut:` per-process, like the existing `_results` store; fine for one user, not
  shared across server processes.

## The tree (CHAID-style)

At each node, for each candidate dimension not used yet on that path:

1. Group the node's cells by the dimension → each category's count, mean, variance.
2. **Merge** categories whose means don't differ significantly (pairwise test, most
   similar first), so "SP, RJ" can share a branch.
3. **Test** the merged split: one-way ANOVA F-test (between vs within variance), with a
   Bonferroni adjustment for the number of ways the categories could have been merged.

Pick the candidate with the smallest adjusted p-value. Split if p < 0.05 and every child
has at least 1% of the node's rows (and at least 500). Otherwise the node is a leaf.

Stop at depth 3. Also stop when the best split changes the average by less than a
minimum effect (default 0.05 for ratings, 2% of the mean otherwise), so a big dataset
doesn't produce "significant" but meaningless splits.

Implemented in `agent/graph/drill.py` (~150 lines). No new dependency: scikit-learn's
trees only split in two and aren't installed; the Python CHAID packages are unmaintained.

## Gap between two groups

"Why is electronics lower than the rest?" uses the same cells:

- The gap's split by mix vs rate (Kitagawa, from the flat version) per candidate: does the
  group have a different make-up, or is it lower within every member?
- The tree is grown on the group's cells to show where it is lowest.

If no mix explains the gap, the answer is "lower across the board" — a finding.

In us_ecom, category comes through order items, so "electronics vs the rest" runs in
per-item mode automatically (next section).

## Per-item mode (v1)

A review is about a whole order; product category, product and seller belong to each
item. Per review, those dimensions have no single value, so they can't be splits. Per
item, they can: each item carries its order's rating.

Example — order #42 rated 2★, with headphones (electronics), a phone case (electronics)
and a lamp (home): per item that is three rows, all 2★; electronics gets the 2★ twice,
home once. Orders with more items weigh more. That is a different question from per
review, and the answer always says which one it is.

(For the record: Cube's default is neither. Asked for review count by category it uses
`count(distinct review_id)`, so each review counts once in every category its order
touches — 469,466 "electronics reviews" = reviews of orders with at least one electronics
item. The engine never uses that implicit definition.)

How it works — the grain moves to the item cube:

- **Weight** = the item cube's count measure (order_items.count) instead of review count.
- **Value** = the averaged column as a per-row dimension (`review_score`; for other
  averages, the per-row dimension PR #50 adds). If no such dimension exists, per-item mode
  isn't available for that measure and the answer says so.
- **Cells** = item count grouped by the candidates + the value → avg = Σ value × items ÷
  Σ items, and the value distribution for the significance test, as before.
- **Candidates** are picked from the item cube's side: items → orders, products, sellers
  (many_to_one) and orders → customers are safe; items → payments is not.
- The value's own path must not fan out either: items → orders → reviews is one_to_many
  in the config, but the data has at most one review per order (checked: max 1). The
  existing "parts add up to the whole" check verifies this on every run.

When it is used — decided by the engine, not the user:

- A comparison on a fan-out dimension ("electronics vs the rest" by product category) →
  per item.
- Otherwise per review; if the per-review tree excluded fan-out dimensions, the result
  lists them, and the agent may re-run per item to include them, saying so.
- Every result and every narrated number carries its unit: "per review" or "per item".

## Agent tools

Replace `compare_segments` and `key_drivers` with one tool:

```
explain_average(measure, filters=[], compare=None, per_item=None)
```

- `compare` = {dimension, values}: group vs the rest (or vs named values).
- `per_item`: None = the engine decides (see Per-item mode); True / False to force it.
- Returns: the tree (path, rows, average, children), excluded dimensions with reasons,
  and `chart_query` for the top split (the guard already accepts it, as for
  `explain_change`).
- Same build_query provenance guard as `explain_change`.

`explain_change` stays for totals over time. For averages over time it moves onto the same
cells later (period as the comparison); until then it keeps the flat mix/rate path.

Prompt: for "what explains / why lower / what affects" an average, call build_query then
`explain_average`; narrate the top branches with numbers; never call a difference below the
minimum effect meaningful; say "nothing checked explains it" when the root has no split.

## Reused / removed

- Reused: candidate ordering and stats, `cube_distances`, fan-out check, `weight_measure`,
  the mix/rate maths, the engine query limit, the Postgres `shm_size` bump.
- Removed: `compare_segments`, `key_drivers`, and their ~50-query fan-out.

## Tests

- Tree on synthetic cells: finds a planted split; merges two equal categories; makes no
  split on pure noise; respects minimum size and depth.
- Candidates: one_to_many paths excluded with a reason; missing relationship falls back to
  the empirical check.
- Per-item mode: weights by item count (the order #42 example gives electronics 2★ × 2,
  home 2★ × 1); picks candidates from the item side; chosen automatically for a fan-out
  comparison; refused with a reason when the value has no per-row dimension.
- Size cap: drops candidates until the estimate fits; retries once on a full result.
- Cache: same key reuses cells with no query; config change drops them.
- Tool through LangGraph (fake Cube): guard, output shape, chart_query.
- Live check, no LLM: us_ecom review ratings per review, and electronics vs the rest
  per item; timing and both trees printed.

## Steps

1. Candidates from joins + size cap.
2. Cells query + memory cache.
3. CHAID tree on cells.
3b. Per-item mode (grain switch, candidates from the item side).
4. `explain_average` tool + prompt; remove the flat tools.
5. Live check on us_ecom; PR.

## Open questions

1. ~~Per-item version: offer only, or v1?~~ Decided: supported in v1.
2. Time as a candidate in v1: yes as planned, or later?
3. Defaults: depth 3, p < 0.05, minimum effect 0.05★ / 2% — OK?
