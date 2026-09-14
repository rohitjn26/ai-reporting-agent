# Evals

Portable evals for the reporting agent. The guiding rule: **don't assert data
values, assert relationships to the schema.** A test that says "output must equal
`orders.total_revenue`" dies the moment the data changes. A test that says "every
field used exists in the live schema, and the query matches the one implied by the
prompt" works on any dataset, forever.

## How the cases are built (back-translation)

`generate.py` reads the **live** Cube schema (`/meta`) and works backwards:

1. Take a field, e.g. measure `orders.total_revenue`, title *"Total Revenue"*,
   description *"… Synonyms: revenue, sales, income, earnings …"*.
2. Build a prompt from it — `"total revenue by country"` (title) and
   `"revenue by country"`, `"sales by country"` (synonyms).
3. You already know the correct query, because you built the prompt from that
   field: `{measures:[orders.total_revenue], dimensions:[orders.country]}`.

Swap in completely different data tomorrow → re-run `generate.py` → new correct
cases, no code changes. The synonym prompts test whether the agent maps oblique
language ("sales", "earnings") onto the right field — the actual hard skill —
and they're auto-labelled because the label came from the schema.

Each case carries the expected **Cube query** and the expected **chart mapping**,
so one file grades both query construction and `save_graph` mapping.

## Files

| File | What |
|---|---|
| `generate.py` | Build cases from live metadata → `generated_queries.jsonl` |
| `grading.py`  | Pure structural graders + the "no hallucinated fields" invariant |

`generated_queries.jsonl` is a build artifact (git-ignored) — regenerate it.

## Usage

```bash
python evals/generate.py            # writes evals/generated_queries.jsonl
python evals/generate.py --print    # print cases, don't write
```

## Grading (portable by construction)

- `grade_query(actual, expected)` — measures/dimensions/time-dimensions match as
  sets; limit/order checked only when the case specifies them.
- `members_exist(actual, metadata)` — the invariant: nothing hallucinated. Works
  on any schema.
- `grade_chart_type` / `grade_mapping` — chart type is acceptable; save_graph
  mapping matches.

## The runner (next)

Not built yet. It will: load the cases, drive the real agent per prompt, extract
the actual `query_cube` / `create_chart` / `save_graph` tool calls from the
`astream_events` trajectory, and grade them with `grading.py`. Report pass-rate
**per template** (not one blended number) so you can see *where* it breaks when
you change a prompt or swap models. Because it calls the live LLM, it runs
separately from unit tests (money + non-determinism), not in the fast CI path.
```
