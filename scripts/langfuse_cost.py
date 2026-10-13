"""
Context size and dollars per question, from Langfuse.

Pulls every LLM call (GENERATION observation) from the Langfuse public API,
groups calls into questions by traceId, and reports per question:
  calls     LLM calls made while answering it
  max ctx   biggest single-call input (the context window it reached)
  input     input tokens summed over all its calls
  cache%    share of that input read from the prompt cache
  cost      dollars (Langfuse prices each call by model)

    python scripts/langfuse_cost.py                          # last 7 days, by day
    python scripts/langfuse_cost.py --since 2026-10-01 --split 2026-10-08   # before vs after
    python scripts/langfuse_cost.py --top 5                  # costliest questions + what bloated them

--top fetches each costly trace's steps and lists tool outputs by size, which is
how you find the tool that dumped 100k tokens into the context.

Needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and optionally LANGFUSE_HOST)
in the environment or the repo-root .env. Read-only; makes no LLM calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
from datetime import date, timedelta
from pathlib import Path

import httpx


def _creds() -> tuple[str, tuple[str, str]]:
    env = dict(os.environ)
    dotenv = Path(__file__).resolve().parent.parent / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if line.startswith("LANGFUSE_") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"'))
    try:
        auth = (env["LANGFUSE_PUBLIC_KEY"], env["LANGFUSE_SECRET_KEY"])
    except KeyError:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (env or repo-root .env).")
    return env.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/"), auth


def fetch_observations(host: str, auth, **params) -> list[dict]:
    """All observations matching params, following the cursor. The legacy
    /api/public/traces endpoint returns 410 for new orgs, so use v2 observations."""
    out, cursor = [], None
    with httpx.Client(timeout=60, auth=auth) as client:
        while True:
            q = {"limit": 100, **params}
            if cursor:
                q["cursor"] = cursor
            r = client.get(f"{host}/api/public/v2/observations", params=q)
            r.raise_for_status()
            body = r.json()
            out += body.get("data", [])
            cursor = (body.get("meta") or {}).get("cursor")
            if not cursor or not body.get("data"):
                return out


def questions(calls: list[dict]) -> list[dict]:
    """One row per trace (= one user question)."""
    by_trace = collections.defaultdict(list)
    for c in calls:
        by_trace[c["traceId"]].append(c)
    rows = []
    for tid, cs in by_trace.items():
        cs.sort(key=lambda c: c["startTime"])
        ins = [c.get("inputUsage") or 0 for c in cs]
        cache_read = sum((c.get("usageDetails") or {}).get("input_cache_read", 0) for c in cs)
        rows.append({
            "trace": tid, "start": cs[0]["startTime"][:16].replace("T", " "),
            "calls": len(cs), "max_ctx": max(ins), "input": sum(ins),
            "cache": cache_read / max(sum(ins), 1),
            "cost": sum(c.get("totalCost") or 0 for c in cs),
        })
    return sorted(rows, key=lambda r: r["start"])


def summary(name: str, rows: list[dict]) -> str:
    if not rows:
        return f"{name:<14} (no questions)"
    med = lambda k: statistics.median(r[k] for r in rows)
    cache = sum(r["cache"] * r["input"] for r in rows) / max(sum(r["input"] for r in rows), 1)
    return (f"{name:<14} {len(rows):>4} q   max ctx {med('max_ctx'):>8,.0f}   "
            f"input/q {med('input'):>9,.0f}   cache {cache:>4.0%}   "
            f"${med('cost'):.3f}/q median   ${sum(r['cost'] for r in rows):.2f} total")


def biggest_tool_outputs(host: str, auth, trace_id: str, n: int = 3) -> list[tuple[str, int]]:
    steps = fetch_observations(host, auth, traceId=trace_id, fields="core,basic,io")
    sizes = [(s.get("name") or "?", len(json.dumps(s.get("output"), default=str)))
             for s in steps if s.get("type") == "TOOL"]
    return sorted(sizes, key=lambda x: -x[1])[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=str(date.today() - timedelta(days=7)), help="YYYY-MM-DD (default: 7 days ago)")
    ap.add_argument("--until", help="YYYY-MM-DD, exclusive")
    ap.add_argument("--split", help="YYYY-MM-DD: compare questions before vs from this date")
    ap.add_argument("--top", type=int, default=0, help="list the N costliest questions and their biggest tool outputs")
    ap.add_argument("--each", action="store_true", help="print every question")
    args = ap.parse_args()

    host, auth = _creds()
    params = {"type": "GENERATION", "fields": "core,basic,model,usage,metrics",
              "fromStartTime": f"{args.since}T00:00:00Z"}
    if args.until:
        params["toStartTime"] = f"{args.until}T00:00:00Z"
    rows = questions(fetch_observations(host, auth, **params))
    if not rows:
        raise SystemExit(f"No LLM calls traced since {args.since}.")

    if args.each:
        for r in rows:
            print(f"{r['start']}  calls={r['calls']:>2}  max ctx={r['max_ctx']:>7,}  "
                  f"input={r['input']:>9,}  cache={r['cache']:>4.0%}  ${r['cost']:.3f}")
        print()

    if args.split:
        print(summary(f"before {args.split}", [r for r in rows if r["start"][:10] < args.split]))
        print(summary(f"from {args.split}", [r for r in rows if r["start"][:10] >= args.split]))
    else:
        for day in sorted({r["start"][:10] for r in rows}):
            print(summary(day, [r for r in rows if r["start"][:10] == day]))
        print(summary("all", rows))

    if args.top:
        print(f"\nCostliest {args.top} questions:")
        for r in sorted(rows, key=lambda r: -r["cost"])[:args.top]:
            print(f"  {r['start']}  ${r['cost']:.3f}  max ctx {r['max_ctx']:,}  trace {r['trace']}")
            for name, size in biggest_tool_outputs(host, auth, r["trace"]):
                print(f"      {name:<28} {size:>9,} chars out")


if __name__ == "__main__":
    main()
