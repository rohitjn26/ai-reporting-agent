# How it works — one-page overview

A natural-language reporting agent. A user asks *"revenue by country as a bar chart"*
in plain English; the agent figures out the right query, runs it through a governed
semantic layer, and renders a chart. It can also **evolve its own schema** at runtime
through a human-approved flow.

Diagrams: `architecture_agent.pdf` · `architecture_evals.pdf` · `architecture_config_edit.pdf`

---

## One request, end to end

*"bar chart of revenue by country"*

1. **UI (`ui.py`, FastAPI):** streams the turn over **Server-Sent Events** (live tokens +
   tool steps). Routes the message — config-edit verbs → **Sonnet**, everything else →
   **Haiku**. Both are LangGraph ReAct agents sharing one checkpointer.
2. **`build_query`:** one **structured LLM call** turns the request into a *validated*
   Cube query — `{measures:[orders.total_revenue], dimensions:[orders.country]}` — checking
   every field against the live schema and repairing once if a field doesn't exist.
3. **`query_cube` (over MCP):** hands the query to **Cube.js**, which compiles it to SQL,
   runs it on Postgres, and returns rows + the generated SQL.
4. **`create_chart`:** renders a Chart.js page in the preview panel.

The LLM never writes SQL or touches the database — it selects semantic fields and
orchestrates tools.

---

## Layers

| Layer | What | Why |
|---|---|---|
| UI / orchestration | `ui.py` + two LangGraph agents (Haiku/Sonnet), SSE streaming | responsive UX; cost follows risk |
| Tools (MCP) | `cube-mcp` (metadata, query), `library-mcp` (config CRUD, graphs, dashboards) | decoupled tool services, standard interface |
| Semantic layer | **Cube.js** — governed measures/dimensions | LLM picks metrics, doesn't author SQL |
| Library API | stores schema **as config data** + saved `GRAPH`/`DASHBOARD` | runtime-evolvable, no code deploy |
| Data | Postgres (business data) + Postgres (config/library) | separation of concerns |

---

## Key design decisions (the "why")

1. **Semantic layer over raw text-to-SQL.** The model selects from governed metrics, so
   "revenue" always means one thing and queries are safe by construction.
2. **Two models, routed.** Cheap/fast **Haiku** for the common query/chart path; stronger
   **Sonnet** for the rarer, higher-stakes schema edits. Cost follows risk.
3. **`build_query` as an isolated, validated step.** The riskiest part (NL→query) is a
   focused structured call with a validate→repair wrapper — testable alone, and it enforces
   a no-hallucinated-fields invariant at runtime.
4. **Config-as-data + human-in-the-loop.** The agent can add a measure, but never silently:
   it proposes → user reviews a pre-filled form (with Test-SQL) → previews a diff → commits.
   *The LLM proposes, the human commits.*
5. **Charts/dashboards as replayable configs, not images.** A saved graph stores the query +
   a mapping, so dashboards re-run live and stay current.

---

## Two mechanics they probe

- **Human-in-the-loop over SSE:** SSE is one-way, so a mid-task question uses a LangGraph
  `interrupt()` — the graph pauses, state is saved by `thread_id`, the stream ends, and the
  answer returns on a **new `/resume` request** that resumes from the checkpoint.
- **Context management:** long chats get summarized — a fast model compresses old turns into
  one system message so the window doesn't blow up.

---

## How I know it's any good — evals

- **Cases generated from the live schema** by back-translation (build the prompt from a field →
  the correct query is known by construction). Survives a data swap: regenerate, don't rewrite.
- **A runner drives the real agent**, captures the actual `query_cube` call, and grades it on
  **separate aspects** — right query · no hallucinated field · acceptable chart type · correct
  save-graph mapping — reported **per template**, so regressions localize.
- Mostly **deterministic checks** (set-match), so cheap and no judge needed; an LLM
  **paraphraser** adds natural-phrasing cases (expected answer *copied*, so labels don't drift;
  a round-trip **verifier** is the planned control against drift).
- It caught a real bug on the first run: for *"top 5 X by Y"* the agent grouped by the wrong
  field and dropped the sort — field-existence passed, but the query check failed.

---

## Soundbites

- *"The LLM chooses semantic fields; it never writes SQL or touches the DB."*
- *"Cost follows risk — cheap model for queries, strong model for schema edits."*
- *"The agent evolves its own schema, but every mutation is human-approved and previewed."*
- *"Charts are configs, not images — dashboards are live, not snapshots."*
- *"I score query accuracy per capability against schema-derived cases — it caught a real top-N bug."*
