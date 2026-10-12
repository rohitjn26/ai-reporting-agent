"""
Refusal eval — does build_query say no when the data can't answer?

Every case asks for something the schema doesn't have: a missing metric
(profit margin), a missing dimension (sales rep), a near miss that tempts a
silent substitute (region when only country exists), or a forecast. The right
outcome is a refusal; the failure is a clean-looking query that answers a
different question.

build_query counts as saying no when its result has `_validation_problems`,
`_view_error`, or no measures and no dimensions. Anything else is a force-fit,
and the report shows the query it built instead.

Each case lists `absent` terms; --dry-run checks they really don't appear in any
member's name/title/description (optionally only in `absent_in` views), so a
case can't quietly become answerable after a schema change.

Calls build_query directly (view routing + one structured LLM call per case,
no feedback examples) — small cost. Run it deliberately:

  python evals/refusal.py --dry-run
  python evals/refusal.py
  python evals/refusal.py --kinds near_miss --out evals/refusal_results.json

Prereqs: Cube up and ANTHROPIC_API_KEY set.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "agent"))

DEFAULT_CASES = str(Path(__file__).resolve().parent / "refusal_cases.jsonl")


def load_cases(path: str, kinds: list[str] | None = None) -> list[dict]:
    cases = [json.loads(line) for line in open(path) if line.strip()]
    return [c for c in cases if not kinds or c["kind"] in set(kinds)]


def _member_text(views: list[dict]) -> list[tuple[str, str]]:
    """(view, lowercase name+title+description) for every member."""
    out = []
    for v in views:
        for m in v.get("measures", []) + v.get("dimensions", []):
            text = " ".join(str(m.get(k) or "") for k in ("name", "title", "shortTitle", "description"))
            out.append((v["name"], text.lower()))
    return out


def check_cases(cases: list[dict], views: list[dict]) -> list[str]:
    """Cases whose 'absent' terms actually appear in the schema (so they may be answerable)."""
    members = _member_text(views)
    problems = []
    for c in cases:
        scope = set(c.get("absent_in") or [v["name"] for v in views])
        for term in c["absent"]:
            hits = sorted({v for v, text in members if v in scope and term.lower() in text})
            if hits:
                problems.append(f"{c['id']}: '{term}' appears in {hits}")
    return problems


def said_no(built: dict) -> bool:
    return bool(built.get("_validation_problems") or built.get("_view_error")
                or not (built.get("measures") or built.get("dimensions")))


def run_cases(cases: list[dict], build, concurrency: int = 4) -> list[dict]:
    """`build(request)` -> build_query result dict; an exception counts as a fail."""
    def one(case):
        try:
            built = build(case["prompt"])
            ok = said_no(built)
        except Exception as e:
            built, ok = {"_error": f"{type(e).__name__}: {e}"}, False
        return {"id": case["id"], "kind": case["kind"], "prompt": case["prompt"],
                "pass": ok, "built": built}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))


def summarise(results: list[dict]) -> dict:
    by_kind = collections.defaultdict(lambda: {"n": 0, "pass": 0})
    for r in results:
        by_kind[r["kind"]]["n"] += 1
        by_kind[r["kind"]]["pass"] += r["pass"]
    return {"by_kind": dict(by_kind),
            "total": {"n": len(results), "pass": sum(r["pass"] for r in results)}}


def print_report(results: list[dict], summary: dict) -> None:
    pct = lambda a, n: f"{100 * a / n:5.1f}%" if n else "   - "
    print(f"\n{'kind':<18}{'n':>4}{'said no':>10}")
    for kind, k in sorted(summary["by_kind"].items()):
        print(f"{kind:<18}{k['n']:>4}{pct(k['pass'], k['n']):>10}")
    t = summary["total"]
    print(f"{'TOTAL':<18}{t['n']:>4}{pct(t['pass'], t['n']):>10}")
    fails = [r for r in results if not r["pass"]]
    if fails:
        print("\nforce-fitted (answered a different question):")
        for r in fails:
            q = {k: r["built"].get(k) for k in ("measures", "dimensions", "_error") if r["built"].get(k)}
            print(f"  [{r['kind']}] {r['prompt']!r}\n      built={json.dumps(q)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="check cases against the live schema; no LLM")
    ap.add_argument("--out")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_ROOT / "agent" / ".env")
    load_dotenv(_ROOT / ".env")
    from graph.agent import _fetch_cube_metadata
    from graph import query_builder as qb

    metadata, cubes = asyncio.run(_fetch_cube_metadata())
    cases = load_cases(args.cases, args.kinds)
    problems = check_cases(cases, metadata)
    print(f"{len(metadata)} views, {len(cases)} cases")
    if problems:
        sys.exit("cases that may be answerable (fix the case or its absent terms):\n  " + "\n  ".join(problems))
    if args.dry_run:
        return

    # No examples_fn: past feedback shouldn't decide whether the builder refuses.
    build = lambda request: qb.build_query(request, metadata, cubes=cubes)
    results = run_cases(cases, build, args.concurrency)
    summary = summarise(results)
    print_report(results, summary)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
