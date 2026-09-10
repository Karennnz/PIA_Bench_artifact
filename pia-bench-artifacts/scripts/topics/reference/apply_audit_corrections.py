#!/usr/bin/env python3
"""Apply conservative semantic corrections emitted by internal QA audits.

The correction files carry expected current-state guards and exact evidence.
Raw API outputs remain under ``assignments/raw_model``; the post-quote-repair,
pre-semantic-audit state is preserved under ``assignments/pre_audit``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = ROOT / "inputs"
ASSIGNMENTS = ROOT / "assignments"
VALIDATION = ROOT / "validation"
PLURAL = {"system": "systems", "risk": "risks", "mitigation": "mitigations"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_text(row: dict) -> str:
    return str(row.get("topic_input_text") or row.get("text") or row.get("descriptive_text") or "")


def apply_system(assignment: dict, correction: dict, source: str) -> dict:
    checks = {
        "primary": (assignment.get("primary_topic_id"), correction["current_primary_topic_id"]),
        "secondary": (assignment.get("secondary_topic_ids", []), correction["current_secondary_topic_ids"]),
        "ai_status": (assignment.get("ai_status"), correction["current_ai_status"]),
    }
    bad = {key: value for key, value in checks.items() if value[0] != value[1]}
    if bad:
        raise RuntimeError(f"System correction current-state mismatch for {assignment['record_id']}: {bad}")
    quote = correction["evidence_quote"]
    if quote not in source:
        raise RuntimeError(f"System correction evidence is not exact: {assignment['record_id']}")
    assignment["primary_topic_id"] = correction["recommended_primary_topic_id"]
    assignment["secondary_topic_ids"] = correction["recommended_secondary_topic_ids"]
    assignment["ai_status"] = correction["recommended_ai_status"]
    assignment["evidence_quote"] = quote
    assignment["evidence_exact"] = True
    assignment["confidence"] = correction.get("confidence", assignment.get("confidence", "high"))
    assignment["rationale"] = correction["rationale"]
    return assignment


def apply_content(assignment: dict, correction: dict, source: str) -> dict:
    current_ids = [item["topic_id"] for item in assignment.get("topics", [])]
    expected = correction["current"]
    if assignment.get("primary_topic_id") != expected["primary_topic_id"] or current_ids != expected["topic_ids"]:
        raise RuntimeError(
            f"Content correction current-state mismatch for {assignment['record_id']}: "
            f"found primary={assignment.get('primary_topic_id')} topics={current_ids}, expected={expected}"
        )
    proposed = correction["proposed"]
    evidence_by_topic = proposed.get("topic_evidence_quotes", {})
    items = []
    for topic_id in proposed["topic_ids"]:
        quote = evidence_by_topic.get(topic_id, correction["evidence_quote"])
        if not quote or quote not in source:
            raise RuntimeError(
                f"Content correction evidence is not exact for {assignment['record_id']} topic={topic_id}"
            )
        items.append({"topic_id": topic_id, "evidence_quote": quote, "evidence_exact": True})
    assignment["primary_topic_id"] = proposed["primary_topic_id"]
    assignment["topics"] = items
    assignment["confidence"] = correction.get("audit_confidence", "high")
    assignment["rationale"] = correction["reasoning"]
    return assignment


def main() -> None:
    staged: dict[Path, list[dict]] = {}
    applied: list[dict] = []
    for kind in PLURAL:
        correction_path = VALIDATION / f"{kind}_assignment_corrections.jsonl"
        corrections = read_jsonl(correction_path)
        if not corrections:
            continue
        assignment_path = ASSIGNMENTS / f"{PLURAL[kind]}.jsonl"
        rows = read_jsonl(assignment_path)
        by_id = {row["record_id"]: row for row in rows}
        source_by_id = {
            row["record_id"]: row
            for row in read_jsonl(INPUTS / f"{PLURAL[kind]}.jsonl")
        }
        for correction in corrections:
            rid = correction["record_id"]
            if rid not in by_id or rid not in source_by_id:
                raise RuntimeError(f"Unknown correction record: {kind}:{rid}")
            before_primary = by_id[rid]["primary_topic_id"]
            source = model_text(source_by_id[rid])
            if kind == "system":
                by_id[rid] = apply_system(by_id[rid], correction, source)
            else:
                by_id[rid] = apply_content(by_id[rid], correction, source)
            by_id[rid]["audit_correction"] = {
                "source": correction_path.name,
                "applied_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "model_assisted_internal_qa_not_human_validation",
                "previous_primary_topic_id": before_primary,
                "correction_type": correction["correction_type"],
            }
            applied.append(
                {
                    "kind": kind,
                    "record_id": rid,
                    "previous_primary_topic_id": before_primary,
                    "final_primary_topic_id": by_id[rid]["primary_topic_id"],
                    "correction_source": correction_path.name,
                    "correction_type": correction["correction_type"],
                }
            )
        staged[assignment_path] = rows

    backup_dir = ASSIGNMENTS / "pre_audit"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in staged.items():
        backup = backup_dir / path.name
        if not backup.exists():
            shutil.copy2(path, backup)
        write_jsonl(path, rows)
    write_jsonl(VALIDATION / "applied_assignment_corrections.jsonl", applied)
    print(f"Applied {len(applied)} guarded semantic corrections across {len(staged)} assignment files")


if __name__ == "__main__":
    main()
