#!/usr/bin/env python3
"""Canonical structural rules for finding risk/mitigation passages.

    GOLD_HEAD = (privacy impact analysis|are there any privacy risks|privacy risk analysis|
                 privacy risk management|privacy analysis and risk)
    RISK_MARK = \\b(?:privacy risk|risk)\\s*:
    MIT_MARK  = \\bmitigation\\s*:

HEADING_RULE generalises GOLD_HEAD — "are there any" dropped, `mitigate`, `what risks` and
`any risks` added. INLINE_RULE merges RISK_MARK and MIT_MARK.
"""

from __future__ import annotations

import re

RULES_VERSION = "structural-v1"

# Rule 1 — the unit's governing question or heading announces a risk/mitigation section.
HEADING_RULE = re.compile(
    r"(privacy risk"
    r"|privacy impact analysis"
    r"|risk analysis"
    r"|risk management"
    r"|mitigate"
    r"|what risks"
    r"|any risks)",
    re.I,
)

# Rule 2 — the unit's text carries an explicit inline marker the agency wrote itself.
INLINE_RULE = re.compile(r"\b(privacy risk|risk|mitigation)\s*:", re.I)


def unit_text(unit: dict) -> str:
    return " ".join(s["text"] for s in unit.get("sentences", []))


def unit_heading(unit: dict) -> str:
    return f"{unit.get('question') or ''} {unit.get('heading') or ''}".strip()


def has_risk_heading(unit: dict) -> bool:
    """Rule 1: does the governing question/heading announce risk or mitigation content?"""
    return bool(HEADING_RULE.search(unit_heading(unit)))


def has_inline_marker(unit: dict) -> bool:
    """Rule 2: does the text contain an explicit `Privacy Risk:` / `Mitigation:` marker?"""
    return bool(INLINE_RULE.search(unit_text(unit)))


def flags_content(unit: dict) -> bool:
    """The combined structural rule: either signal is enough."""
    return has_risk_heading(unit) or has_inline_marker(unit)


def inline_type(text: str) -> str | None:
    """The one type assignment the rules can make without judgement.

    Returns 'risk', 'mitigation' or None. Only ~27 of 73 documents use these markers, so this
    covers about a quarter of the content — but where present it needs no interpretation.
    """
    m = INLINE_RULE.search(text)
    if not m:
        return None
    return "mitigation" if m.group(1).lower() == "mitigation" else "risk"


if __name__ == "__main__":
    import json
    from pathlib import Path

    units = [json.loads(l) for l in (Path(__file__).parent / "inputs/units.jsonl").open()]
    h = sum(map(has_risk_heading, units))
    i = sum(map(has_inline_marker, units))
    both = sum(1 for u in units if flags_content(u))
    print(f"{RULES_VERSION} over {len(units)} units")
    print(f"  heading rule matches : {h}")
    print(f"  inline rule matches  : {i}")
    print(f"  either               : {both}")
