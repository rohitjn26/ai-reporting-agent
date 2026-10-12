"""
View-routing eval — does build_query's first step pick the right view?

Runs only the routing step (query_builder.select_view: one small structured LLM
call per case) against the live view catalog, not the whole agent. Each case in
routing_cases.jsonl lists the views that count as correct, best first; an empty
list means the request spans views and the router should refuse (view=null).

Reports, per case kind:
  - pass     : the chosen view is one of `accept` (or null for span cases)
  - preferred: the chosen view is `accept[0]`
plus which views get confused with which, so a bad description shows up by name.

Calls the live LLM (small cost, non-deterministic) — run it deliberately:

  python evals/routing.py --dry-run           # check cases vs live views, no LLM
  python evals/routing.py                     # all cases
  python evals/routing.py --kinds span member_only
  python evals/routing.py --out evals/routing_results.json

Prereqs: Cube up (its /meta is the catalog) and ANTHROPIC_API_KEY set.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "agent"))

DEFAULT_CASES = str(Path(__file__).resolve().parent / "routing_cases.jsonl")


def load_cases(path: str, kinds: list[str] | None = None) -> list[dict]:
    cases = [json.loads(line) for line in open(path) if line.strip()]
    return [c for c in cases if not kinds or c["kind"] in set(kinds)]


def check_cases(cases: list[dict], view_names: set[str]) -> list[str]:
    """Problems with the case file itself: unknown views, duplicate ids."""
    problems, seen = [], set()
    for c in cases:
        if c["id"] in seen:
            problems.append(f"{c['id']}: duplicate id")
        seen.add(c["id"])
        for v in c["accept"]:
            if v not in view_names:
                problems.append(f"{c['id']}: unknown view '{v}'")
    return problems


def grade(case: dict, chosen: str | None) -> dict:
    accept = case["accept"]
    passed = chosen is None if not accept else chosen in accept
    preferred = chosen is None if not accept else chosen == accept[0]
    return {"id": case["id"], "kind": case["kind"], "prompt": case["prompt"],
            "accept": accept, "chosen": chosen, "pass": passed, "preferred": preferred}


def run_cases(cases: list[dict], metadata: list[dict], cubes: list[dict], select, concurrency: int = 4) -> list[dict]:
    """`select(request, metadata, cubes=...)` -> ViewSelection; errors count as a fail."""
    def one(case):
        try:
            sel = select(case["prompt"], metadata, cubes=cubes)
            r = grade(case, sel.view)
            r["reason"] = sel.reason
        except Exception as e:
            r = grade(case, "<error>")
            r["reason"] = f"{type(e).__name__}: {e}"
        return r
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))


def summarise(results: list[dict]) -> dict:
    by_kind = collections.defaultdict(lambda: {"n": 0, "pass": 0, "preferred": 0})
    confusions = collections.Counter()
    for r in results:
        k = by_kind[r["kind"]]
        k["n"] += 1
        k["pass"] += r["pass"]
        k["preferred"] += r["preferred"]
        if not r["pass"]:
            want = r["accept"][0] if r["accept"] else "<refuse>"
            confusions[(want, r["chosen"] or "<refuse>")] += 1
    total = {"n": len(results), "pass": sum(r["pass"] for r in results),
             "preferred": sum(r["preferred"] for r in results)}
    return {"by_kind": dict(by_kind), "total": total, "confusions": confusions.most_common()}


def print_report(results: list[dict], summary: dict) -> None:
    pct = lambda a, n: f"{100 * a / n:5.1f}%" if n else "   - "
    print(f"\n{'kind':<15}{'n':>4}{'pass':>9}{'preferred':>12}")
    for kind, k in sorted(summary["by_kind"].items()):
        print(f"{kind:<15}{k['n']:>4}{pct(k['pass'], k['n']):>9}{pct(k['preferred'], k['n']):>12}")
    t = summary["total"]
    print(f"{'TOTAL':<15}{t['n']:>4}{pct(t['pass'], t['n']):>9}{pct(t['preferred'], t['n']):>12}")
    if summary["confusions"]:
        print("\nconfusions (expected -> chosen):")
        for (want, got), n in summary["confusions"]:
            print(f"  {n}x  {want} -> {got}")
    fails = [r for r in results if not r["pass"]]
    if fails:
        print("\nfailures:")
        for r in fails:
            print(f"  [{r['kind']}] {r['prompt']!r}\n      accept={r['accept']} chosen={r['chosen']}\n      reason={r.get('reason', '')[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--kinds", nargs="*", help="only these case kinds")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="validate cases against live views; no LLM")
    ap.add_argument("--out", help="write per-case results JSON here")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_ROOT / "agent" / ".env")
    load_dotenv(_ROOT / ".env")
    from graph.agent import _fetch_cube_metadata
    from graph import query_builder as qb

    metadata, cubes = asyncio.run(_fetch_cube_metadata())
    cases = load_cases(args.cases, args.kinds)
    problems = check_cases(cases, {v["name"] for v in metadata})
    print(f"{len(metadata)} views, {len(cases)} cases, catalog "
          f"{len(qb.render_view_catalog(metadata, cubes))} chars, router model "
          f"{os.environ.get('VIEW_SELECTOR_MODEL', 'claude-haiku-4-5-20251001')}")
    if problems:
        sys.exit("case file problems:\n  " + "\n  ".join(problems))
    if args.dry_run:
        return

    results = run_cases(cases, metadata, cubes, qb.select_view, args.concurrency)
    summary = summarise(results)
    print_report(results, summary)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
