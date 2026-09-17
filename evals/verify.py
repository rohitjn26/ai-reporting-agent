"""
Paraphrase verifier — drift control for auto-labelled paraphrase cases.

`paraphrase.py` copies the expected query verbatim onto each reworded prompt, on
the promise that the LLM changed only the *wording*. But an LLM can quietly change
the *meaning* ("revenue by country" -> "profit by region"), and because the label
is copied, nothing downstream would notice — the eval would then assert a wrong
answer and blame the agent for getting a mislabelled case "wrong".

This module is the second, independent LLM the design calls for. It re-derives a
Cube query from the paraphrase ALONE (it never sees the copied label), then checks
whether that query round-trips to the case's expected query. A paraphrase is kept
only if an independent reader, given just the words, lands on the same query.
Ones that don't round-trip are dropped as drift — or as genuine ambiguity, which
is equally unsafe to auto-label.

Independence matters twice:
  - Different model from the paraphraser by default (EVAL_VERIFIER_MODEL), so the
    two don't share the same blind spot.
  - Generation-time gate, separate from run.py (the agent under test), so the
    label can never leak into what's being graded.

Reuses `query_builder.build_query` for the NL->query step (schema rendering +
member validation come free) and `grading.grade_query` for the round-trip check —
the exact structural comparison the eval itself uses.

Offline/testable: pass your own `verify_fn(prompt, metadata) -> dict` (a Cube
query); the default calls query_builder (Anthropic) only when you actually run it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Reuse the agent's NL->query component and the eval's structural grader.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import grading  # sibling module (evals/ is on sys.path)

# A DIFFERENT model from the paraphraser (haiku) by default, so the re-derivation
# doesn't inherit the same mistakes the paraphraser might make.
_DEFAULT_MODEL = os.environ.get("EVAL_VERIFIER_MODEL", "claude-sonnet-4-6")


def _build_query_verifier(prompt: str, metadata: list[dict], model: str) -> dict:
    """Independent NL->query: re-derive a Cube query from the paraphrase alone."""
    from graph.query_builder import build_query  # lazy — needs agent deps + a key
    return build_query(prompt, metadata, model=model)


def verify_case(case: dict, metadata: list[dict], *, verify_fn) -> dict:
    """Re-derive a query from the case prompt and grade it against the label.

    Returns {"round_trips": bool, "derived": <query>, "grade": <grade_query dict>}.
    The derived query is stripped of the validator's `_validation_problems`
    annotation before comparison so it compares cleanly."""
    derived = verify_fn(case["prompt"], metadata)
    derived_clean = {k: v for k, v in derived.items() if k != "_validation_problems"}
    grade = grading.grade_query(derived_clean, case["expected"])
    return {"round_trips": grade["passed"], "derived": derived_clean, "grade": grade}


def verify_cases(
    cases: list[dict],
    metadata: list[dict],
    *,
    verify_fn=None,
    model: str = _DEFAULT_MODEL,
    only_source: str = "llm_paraphrase",
) -> tuple[list[dict], list[dict]]:
    """Split cases into (kept, dropped).

    Only `only_source` cases are verified; everything else passes through into
    `kept` untouched (title/synonym cases are already correct by construction).
    A verified case is kept iff its prompt round-trips to the expected query.
    Dropped cases carry a `_drift` field (the derived query + the failing checks)
    so you can eyeball what the paraphraser actually changed.

    `verify_fn(prompt, metadata) -> query dict` is injectable for offline tests.
    """
    fn = verify_fn or (lambda p, md: _build_query_verifier(p, md, model))
    kept: list[dict] = []
    dropped: list[dict] = []
    for c in cases:
        if c.get("source") != only_source:
            kept.append(c)
            continue
        try:
            res = verify_case(c, metadata, verify_fn=fn)
        except Exception as e:  # a failed verify shouldn't silently keep an unchecked case
            dropped.append({**c, "_drift": {"error": str(e)}})
            print(f"  [warn] verify failed for {c['prompt']!r}: {e} — dropped")
            continue
        if res["round_trips"]:
            kept.append(c)
        else:
            dropped.append({**c, "_drift": {
                "derived": res["derived"], "checks": res["grade"]["checks"],
            }})
    return kept, dropped
