"""
Eval runner — drive the live agent over generated cases and score it.

For each case it sends the prompt to the real agent (routed exactly like
production via pick_agent), captures the actual tool calls from the
astream_events trajectory, and grades them with grading.py:

  - query    : the last query_cube args vs the case's expected query
  - members  : no hallucinated fields (invariant, any schema)
  - chart    : create_chart's type is acceptable        (only if a chart was made)
  - mapping  : save_graph mapping matches               (only if a graph was saved)

Reports pass-rate per aspect, broken down by template and source, so a
regression shows up where it happened instead of as one blended number.

This calls the live LLM (costs money, non-deterministic), so it is NOT part of
the fast unit suite. Run it deliberately:

  python evals/run.py                       # sample of 20 cases
  python evals/run.py --limit 0             # all cases
  python evals/run.py --templates pivot top_n
  python evals/run.py --source llm_paraphrase
  python evals/run.py --out evals/eval_results.json

Prereqs: the stack up (cube, cube-mcp, library-mcp) and ANTHROPIC_API_KEY set.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import uuid
from pathlib import Path

# Make agent modules and sibling eval modules importable.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("CHART_OPEN_BROWSER", "false")  # never pop a browser during evals

import generate
import grading


DEFAULT_CASES = str(Path(__file__).resolve().parent / "generated_queries.jsonl")


# ── load ──────────────────────────────────────────────────────────────────────

def load_cases(path: str, limit: int, templates: list[str] | None, source: str | None) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"No cases at {path}. Generate them first:  python evals/generate.py")
    cases = [json.loads(line) for line in open(path) if line.strip()]
    if templates:
        cases = [c for c in cases if c["template"] in set(templates)]
    if source:
        cases = [c for c in cases if c["source"] == source]
    if limit and limit > 0:
        cases = cases[:limit]
    return cases


# ── drive the agent ───────────────────────────────────────────────────────────

async def run_case(case: dict) -> dict:
    """Send one prompt to the agent, return the tool calls it actually made."""
    from langchain_core.messages import HumanMessage
    from graph.agent import pick_agent

    agent = pick_agent(case["prompt"])   # same routing as production
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 30}

    last_query = None      # last query_cube args (the one that feeds the answer)
    chart_type = None
    mapping = None
    error = None
    try:
        async for ev in agent.astream_events(
            {"messages": [HumanMessage(content=case["prompt"])]}, config=config, version="v2"
        ):
            if ev["event"] != "on_tool_end":
                continue
            name = ev.get("name", "")
            inp = ev["data"].get("input") or {}
            if name == "query_cube":
                last_query = inp
            elif name == "create_chart":
                chart_type = inp.get("chart_type")
            elif name == "save_graph":
                mapping = inp.get("mapping")
    except Exception as e:
        error = str(e)

    return {"query": last_query, "chart_type": chart_type, "mapping": mapping, "error": error}


def grade_case(case: dict, observed: dict, metadata: list[dict]) -> dict:
    """Score one case across the aspects we could observe."""
    aspects: dict[str, dict] = {}
    actual_q = observed["query"]

    if actual_q is None:
        aspects["query"] = {"passed": False, "reason": observed["error"] or "agent never called query_cube"}
        aspects["members"] = {"passed": False, "reason": "no query"}
    else:
        aspects["query"] = grading.grade_query(actual_q, case["expected"])
        aspects["members"] = grading.members_exist(actual_q, metadata)

    if observed["chart_type"] is not None:
        aspects["chart"] = grading.grade_chart_type(observed["chart_type"], case["expected_chart_types"])
    if observed["mapping"] is not None:
        aspects["mapping"] = grading.grade_mapping(observed["mapping"], case["expected_mapping"])

    return {
        "id": case["id"], "template": case["template"], "source": case["source"],
        "prompt": case["prompt"], "expected": case["expected"],
        "actual_query": actual_q, "aspects": aspects,
    }


# ── run + report ──────────────────────────────────────────────────────────────

async def _bounded(sem, coro):
    async with sem:
        return await coro


async def run(cases: list[dict], metadata: list[dict], concurrency: int) -> list[dict]:
    from graph.agent import build_agent
    await build_agent()   # populates the sonnet/haiku agents pick_agent chooses from

    sem = asyncio.Semaphore(concurrency)
    # gather preserves order, so results line up with cases.
    observed_list = await asyncio.gather(*[_bounded(sem, run_case(c)) for c in cases])
    return [grade_case(c, o, metadata) for c, o in zip(cases, observed_list)]


def _pct(n: int, d: int) -> str:
    return f"{(100*n/d):.0f}% ({n}/{d})" if d else "—"


def report(results: list[dict]) -> None:
    # aspect -> template -> [passed, total]
    agg: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    src_agg: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in results:
        for aspect, res in r["aspects"].items():
            for bucket, key in ((agg, r["template"]), (src_agg, r["source"])):
                bucket[aspect][key][1] += 1
                if res.get("passed"):
                    bucket[aspect][key][0] += 1

    aspects = ["query", "members", "chart", "mapping"]
    print("\n" + "=" * 72)
    print(f"EVAL RESULTS — {len(results)} cases")
    print("=" * 72)

    print("\nBy TEMPLATE:")
    templates = sorted({r["template"] for r in results})
    print(f"  {'template':<22} " + "  ".join(f"{a:<12}" for a in aspects))
    for t in templates:
        cells = []
        for a in aspects:
            p, tot = agg[a][t]
            cells.append(f"{_pct(p, tot):<12}" if tot else f"{'—':<12}")
        print(f"  {t:<22} " + "  ".join(cells))

    print("\nBy SOURCE:")
    print(f"  {'source':<22} " + "  ".join(f"{a:<12}" for a in aspects))
    for s in sorted({r["source"] for r in results}):
        cells = []
        for a in aspects:
            p, tot = src_agg[a][s]
            cells.append(f"{_pct(p, tot):<12}" if tot else f"{'—':<12}")
        print(f"  {s:<22} " + "  ".join(cells))

    # overall
    print("\nOVERALL:")
    for a in aspects:
        tot = sum(v[1] for v in agg[a].values())
        p = sum(v[0] for v in agg[a].values())
        if tot:
            print(f"  {a:<10} {_pct(p, tot)}")

    # failures worth eyeballing
    fails = [r for r in results if not r["aspects"].get("query", {}).get("passed")]
    if fails:
        print(f"\nQUERY FAILURES ({len(fails)}):")
        for r in fails[:25]:
            q = r["aspects"]["query"]
            detail = q.get("reason") or f"checks={q.get('checks')}"
            print(f"  · [{r['template']}/{r['source']}] {r['prompt']!r}")
            print(f"      expected={json.dumps(r['expected'])}")
            print(f"      actual  ={json.dumps(r['actual_query'])}")
            print(f"      {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--limit", type=int, default=20, help="max cases to run (0 = all)")
    ap.add_argument("--templates", nargs="*", help="filter to these templates")
    ap.add_argument("--source", help="filter to one source (title|synonym|llm_paraphrase)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", help="write per-case results JSON here")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (source it from agent/.env).")

    cases = load_cases(args.cases, args.limit, args.templates, args.source)
    if not cases:
        sys.exit("No cases matched the filters.")
    print(f"Running {len(cases)} case(s) at concurrency {args.concurrency} …")

    metadata = generate.fetch_metadata()
    results = asyncio.run(run(cases, metadata, args.concurrency))
    report(results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote per-case results -> {args.out}")


if __name__ == "__main__":
    main()
