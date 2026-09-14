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

### LLM paraphrases (opt-in, `paraphrase.py`)

Synonyms cover the words *you* wrote into descriptions. To test genuinely messy,
real-user phrasing, `--paraphrase N` asks an LLM to reword each `title` case into
N natural variants ("which countries make us the most money?"). The safety
property: **the LLM only rewrites the wording — the expected query/mapping is
copied from the base case**, so paraphrases stay auto-labelled. Only
title-sourced `single_measure`/`measure_by_dimension`/`measure_over_time` cases
are paraphrased; `top_n`/`pivot` hinge on qualifiers ("top 5", "split by") a
rewrite might drop, which would invalidate the copied label. Paraphrases are a
notch lower-trust than title/synonym cases — spot-check them.

Each case carries the expected **Cube query** and the expected **chart mapping**,
so one file grades both query construction and `save_graph` mapping.

## Files

| File | What |
|---|---|
| `generate.py`   | Build cases from live metadata → `generated_queries.jsonl` |
| `paraphrase.py` | LLM reword of title cases into natural phrasings (opt-in) |
| `grading.py`    | Pure structural graders + the "no hallucinated fields" invariant |
| `run.py`        | Drive the live agent over cases and score it per aspect/template |

`generated_queries.jsonl` is a build artifact (git-ignored) — regenerate it.

## Usage

```bash
python evals/generate.py                 # deterministic: title + synonym cases
python evals/generate.py --print         # print cases, don't write
python evals/generate.py --paraphrase 3  # + 3 LLM paraphrases per title case
```

## Grading (portable by construction)

- `grade_query(actual, expected)` — measures/dimensions/time-dimensions match as
  sets; limit/order checked only when the case specifies them.
- `members_exist(actual, metadata)` — the invariant: nothing hallucinated. Works
  on any schema.
- `grade_chart_type` / `grade_mapping` — chart type is acceptable; save_graph
  mapping matches.

## The runner (`run.py`)

Loads the cases, drives the real agent per prompt (routed exactly like production
via `pick_agent`), extracts the actual `query_cube` / `create_chart` / `save_graph`
tool calls from the `astream_events` trajectory, and grades them with `grading.py`.
Reports pass-rate **per aspect × per template/source** — not one blended number —
so you see *where* it breaks when you change a prompt or swap models.

```bash
make eval                                  # sample of 20 cases (regenerates first)
make eval ARGS="--limit 0"                 # all cases
make eval ARGS="--templates pivot top_n"   # focus on the hard ones
make eval ARGS="--source llm_paraphrase"   # robustness on natural phrasing
python evals/run.py --out evals/eval_results.json   # save per-case detail
```

It calls the live LLM (money + non-determinism) and needs the stack up +
`ANTHROPIC_API_KEY`, so it is **not** part of the fast unit suite / CI.

### Aspects scored

- `query`   — last `query_cube` args vs the case's expected query
- `members` — no hallucinated fields (invariant; any schema)
- `chart`   — `create_chart` type is acceptable (only when a chart was made)
- `mapping` — `save_graph` mapping matches (only when a graph was saved)

`chart`/`mapping` are only scored when the agent actually took that action; plain
"X by Y" prompts usually stop at the query (the agent asks which chart type), so
those columns show `—` unless the prompt drives a chart/save.
```
