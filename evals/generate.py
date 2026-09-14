"""
Back-translation eval generator — build (prompt -> expected query) cases FROM the
live cube schema, so the eval set survives a data/schema swap.

The idea: don't hand-write "revenue by country -> orders.total_revenue". Instead
read a field from metadata, construct a prompt from its title (and its synonyms),
and you already know the correct answer because you built the prompt from that
exact field. Swap in entirely different data tomorrow and the generator produces
new, correct cases with no code changes — just re-run it.

Prompt sources per case:
  - "title"         : literal, built from the field's title       (easy mapping)
  - "synonym"       : built from a synonym in the description     (robustness)
  - "llm_paraphrase": natural rewording from an LLM (opt-in, see paraphrase.py;
                      the expected query is COPIED, so cases stay auto-labelled)

Cases carry both the expected Cube query AND the expected chart mapping, so the
same file can grade query construction and save_graph mapping later.

Usage:
  python evals/generate.py                      # -> evals/generated_queries.jsonl
  python evals/generate.py --out cases.jsonl
  python evals/generate.py --print              # print cases, don't write
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

CUBE_URL        = os.environ.get("CUBE_URL", "http://localhost:4000")
CUBE_API_SECRET = os.environ.get("CUBE_API_SECRET", "local-dev-secret")

# Dimensions we never group by (identifiers / free-scale numerics make poor axes).
_SKIP_DIM_TYPES = {"number"}


# ── metadata → normalized members ─────────────────────────────────────────────

def fetch_metadata() -> list[dict]:
    """Fetch the live Cube data model. Returns the list of cube dicts from /meta."""
    req = urllib.request.Request(
        f"{CUBE_URL}/cubejs-api/v1/meta",
        headers={"Authorization": f"Bearer {CUBE_API_SECRET}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read()).get("cubes", [])


def _title(member: dict) -> str:
    """Human label without the cube-name prefix Cube prepends to `title`."""
    return member.get("shortTitle") or member.get("title") or member["name"].split(".")[-1]


def _synonyms(member: dict) -> list[str]:
    """Pull synonyms out of a description written as '... Synonyms: a, b, c.'"""
    desc = member.get("description") or ""
    m = re.search(r"synonyms?:\s*(.+?)(?:\.|$)", desc, flags=re.IGNORECASE)
    if not m:
        return []
    out = []
    for term in m.group(1).split(","):
        t = term.strip().rstrip(".")
        if t and t.lower() not in (s.lower() for s in out):
            out.append(t)
    return out


# ── case construction (pure — unit tested) ────────────────────────────────────

def _case(cube, template, prompt, source, expected, chart_types, mapping):
    return {
        "id":                  f"{cube}::{template}::{source}::{re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')}"[:120],
        "cube":                cube,
        "template":            template,
        "source":              source,      # "title" | "synonym"
        "prompt":              prompt,
        "expected":            expected,    # Cube query the agent should build
        "expected_chart_types": chart_types, # any of these is acceptable
        "expected_mapping":    mapping,     # for save_graph grading (None for plain single measure)
    }


def generate_cases(cubes: list[dict], *, max_synonyms: int = 2) -> list[dict]:
    """Produce eval cases from cube metadata. Deterministic and offline."""
    cases: list[dict] = []

    for c in cubes:
        cube = c["name"]
        measures = c.get("measures", [])
        dims_all = c.get("dimensions", [])
        cat_dims  = [d for d in dims_all if d.get("type") == "string"]
        time_dims = [d for d in dims_all if d.get("type") == "time"]
        # Group-by-able dimensions (skip ids and raw numerics).
        _ = [d for d in dims_all if d.get("type") not in _SKIP_DIM_TYPES]

        for m in measures:
            m_name, m_title = m["name"], _title(m)
            m_syn = _synonyms(m)[:max_synonyms]

            # 1) single measure (aggregate, no grouping)
            cases.append(_case(
                cube, "single_measure", f"total {m_title.lower()}", "title",
                {"measures": [m_name]}, ["table"], None,
            ))

            # 2) measure by a categorical dimension  (+ synonym variants + top-N)
            for d in cat_dims:
                d_name, d_title = d["name"], _title(d)
                expected = {"measures": [m_name], "dimensions": [d_name]}
                mapping  = {"label_dimension": d_name, "series_measures": [m_name],
                            "series_dimension": None}
                cases.append(_case(
                    cube, "measure_by_dimension", f"{m_title.lower()} by {d_title.lower()}",
                    "title", expected, ["bar", "table"], mapping,
                ))
                for syn in m_syn:
                    cases.append(_case(
                        cube, "measure_by_dimension", f"{syn.lower()} by {d_title.lower()}",
                        "synonym", expected, ["bar", "table"], mapping,
                    ))
                cases.append(_case(
                    cube, "top_n", f"top 5 {d_title.lower()} by {m_title.lower()}", "title",
                    {**expected, "order": {m_name: "desc"}, "limit": 5}, ["bar", "table"], mapping,
                ))

            # 3) measure over time  (+ pivot: split by first categorical dim)
            for t in time_dims:
                t_name, t_title = t["name"], _title(t)
                expected_time = {
                    "measures": [m_name],
                    "time_dimensions": [{"dimension": t_name, "granularity": "month"}],
                }
                cases.append(_case(
                    cube, "measure_over_time", f"monthly {m_title.lower()}", "title",
                    expected_time, ["line", "bar"],
                    {"label_dimension": t_name, "series_measures": [m_name], "series_dimension": None},
                ))
                if cat_dims:
                    d = cat_dims[0]
                    d_name, d_title = d["name"], _title(d)
                    cases.append(_case(
                        cube, "pivot",
                        f"monthly {m_title.lower()} split by {d_title.lower()}", "title",
                        {**expected_time, "dimensions": [d_name]}, ["line", "bar"],
                        {"label_dimension": t_name, "series_measures": [m_name],
                         "series_dimension": d_name},
                    ))

    # Drop duplicate prompts (e.g. a synonym that equals the title). Keep the
    # first occurrence — title-sourced cases are generated before synonyms.
    seen, deduped = set(), []
    for c in cases:
        key = (c["cube"], c["template"], c["prompt"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


# ── CLI ───────────────────────────────────────────────────────────────────────

def _dedupe(cases: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cases:
        key = (c["cube"], c["template"], c["prompt"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "generated_queries.jsonl"))
    ap.add_argument("--print", dest="do_print", action="store_true", help="print cases instead of writing")
    ap.add_argument("--paraphrase", type=int, default=0, metavar="N",
                    help="add N LLM paraphrases per title case (needs ANTHROPIC_API_KEY; 0=off)")
    ap.add_argument("--paraphrase-model", default=None, help="override the paraphrase model id")
    args = ap.parse_args()

    cubes = fetch_metadata()
    cases = generate_cases(cubes)

    if args.paraphrase > 0:
        import paraphrase
        kw = {"model": args.paraphrase_model} if args.paraphrase_model else {}
        extra = paraphrase.paraphrase_cases(cases, n=args.paraphrase, **kw)
        cases = _dedupe(cases + extra)
        print(f"Added {len(extra)} LLM paraphrase case(s).")

    by_template: dict[str, int] = {}
    for c in cases:
        by_template[c["template"]] = by_template.get(c["template"], 0) + 1

    if args.do_print:
        for c in cases:
            print(json.dumps(c))
    else:
        with open(args.out, "w") as f:
            for c in cases:
                f.write(json.dumps(c) + "\n")
        print(f"Wrote {len(cases)} cases -> {args.out}")

    print("By template: " + ", ".join(f"{k}={v}" for k, v in sorted(by_template.items())))


if __name__ == "__main__":
    main()
