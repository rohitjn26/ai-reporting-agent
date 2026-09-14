"""
LLM paraphrase layer — expand eval cases with natural, oblique phrasings.

The deterministic generator (`generate.py`) builds prompts from field titles and
the synonyms written into descriptions. This module goes further: it asks an LLM
to rewrite a canonical prompt into the messy ways a real business user would ask
for the SAME thing ("which countries make us the most money?" for "revenue by
country").

Safety property that keeps cases auto-labelled: the LLM only rewrites the
*wording*. The `expected` query and mapping are copied verbatim from the base
case — the model never decides the answer, so a paraphrase can't silently change
the label. (It CAN drift the meaning, though — see the template allow-list below.)

Offline/testable: pass your own `generate_fn(prompt, n) -> list[str]`; the default
calls Anthropic only when you actually run paraphrasing.
"""
from __future__ import annotations

import json
import os
import re

import generate  # sibling module (evals/ is on sys.path when run)

_DEFAULT_MODEL = os.environ.get("EVAL_PARAPHRASE_MODEL", "claude-haiku-4-5-20251001")

# Only paraphrase templates whose meaning survives loose rewording. top_n and
# pivot hinge on qualifiers ("top 5", "split by") an LLM may drop — dropping them
# would invalidate the copied label — so we leave those to title/synonym cases.
_PARAPHRASABLE = {"single_measure", "measure_by_dimension", "measure_over_time"}

_SYSTEM = (
    "You generate test paraphrases for a data-analytics assistant. Given one "
    "canonical data request, produce natural, varied ways a real business user "
    "might ask for THE SAME THING.\n"
    "Rules:\n"
    "- Keep the exact same metric(s), grouping, and any time period. Do NOT add, "
    "drop, or change filters, limits, sortings, or dimensions.\n"
    "- Vary tone and vocabulary: use synonyms, questions, casual phrasing.\n"
    "- Each paraphrase must be self-contained and unambiguous.\n"
    "- Return ONLY a JSON array of strings. No prose, no numbering."
)


def _parse_array(text: str) -> list[str]:
    """Pull a JSON array of strings out of an LLM response, tolerating stray prose."""
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: one paraphrase per non-empty line.
    return [ln.strip(" -*\t").strip() for ln in text.splitlines() if ln.strip()]


def _anthropic_generate(prompt: str, n: int, model: str) -> list[str]:
    from langchain_anthropic import ChatAnthropic  # lazy — only when actually used
    llm = ChatAnthropic(model=model, temperature=1.0, max_tokens=500)
    resp = llm.invoke([
        ("system", _SYSTEM),
        ("human", f'Canonical request: "{prompt}"\nGive {n} paraphrases as a JSON array of strings.'),
    ])
    content = resp.content if isinstance(resp.content, str) else "".join(
        b.get("text", "") for b in resp.content if isinstance(b, dict)
    )
    return _parse_array(content)[:n]


def paraphrase_cases(
    base_cases: list[dict],
    n: int = 3,
    *,
    model: str = _DEFAULT_MODEL,
    generate_fn=None,
    templates: set[str] = _PARAPHRASABLE,
) -> list[dict]:
    """Return NEW cases (source='llm_paraphrase') — the base cases are untouched.

    Only `source='title'` cases in the allow-listed templates are paraphrased, so
    we don't paraphrase-a-paraphrase and don't risk meaning drift on top_n/pivot.
    `generate_fn(prompt, n)` is injectable for offline testing.
    """
    gen = generate_fn or (lambda p, k: _anthropic_generate(p, k, model))
    out: list[dict] = []
    for c in base_cases:
        if c["source"] != "title" or c["template"] not in templates:
            continue
        try:
            variants = gen(c["prompt"], n)
        except Exception as e:  # a failed call for one prompt shouldn't kill the batch
            print(f"  [warn] paraphrase failed for {c['prompt']!r}: {e}")
            continue
        for v in variants:
            v = v.strip()
            if not v or v.lower() == c["prompt"].lower():
                continue
            # Reuse the canonical builder so id format + dedup behave identically;
            # crucially, the expected query/mapping are copied from the base case.
            out.append(generate._case(
                c["cube"], c["template"], v, "llm_paraphrase",
                c["expected"], c["expected_chart_types"], c["expected_mapping"],
            ))
    return out
