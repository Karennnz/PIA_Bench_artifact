#!/usr/bin/env python3
"""Build an auditable released-reference topic view (451 risks; 831 mitigations).

This is a presentation/analysis layer. It never edits the frozen canonical inputs
or assignments. Canonical special codes are re-audited, residual outcomes are
collapsed to OTHER, and the 17 overlap-derived released spans are restored by
inheriting the final displayed topic of their retained counterpart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import textwrap
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
WORKSPACE = RUN_ROOT.parent
SOURCE_ROOT = WORKSPACE / "2026-09-06_risk_mitigation_topics_73"
SNAPSHOT = SOURCE_ROOT / "snapshot" / "content_analysis_73"
REVIEW = RUN_ROOT / "review"
OUTPUT = RUN_ROOT / "outputs"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

INPUT_PATHS = {
    "risk": SNAPSHOT / "inputs" / "risks.jsonl",
    "mitigation": SNAPSHOT / "inputs" / "mitigations.jsonl",
}
ASSIGNMENT_PATHS = {
    "risk": SOURCE_ROOT / "work" / "assignments" / "risks.jsonl",
    "mitigation": SOURCE_ROOT / "work" / "assignments" / "mitigations.jsonl",
}
REVIEW_PATHS = {
    "risk": REVIEW / "risk_special_reaudit.jsonl",
    "mitigation": REVIEW / "mitigation_special_reaudit.jsonl",
}
EXCLUSION_PATH = SNAPSHOT / "validation" / "canonical_exclusions.jsonl"
OVERLAP_PATH = REVIEW / "restored_overlap_assignments.jsonl"
OVERRIDE_PATH = REVIEW / "final_adjudication_overrides.jsonl"
TERMINOLOGY_PATH = SOURCE_ROOT / "presentation" / "presentation_terminology.json"

EXPECTED_CANONICAL = {"risk": 444, "mitigation": 821}
EXPECTED_RESTORED = {"risk": 7, "mitigation": 10}
EXPECTED_RELEASED = {"risk": 451, "mitigation": 831}
EXPECTED_SPECIAL = {"risk": 92, "mitigation": 89}
EXPECTED_DOCUMENTS = 73

TOPIC_IDS = {
    "risk": [f"R{i:02d}" for i in range(1, 9)] + ["R_OTHER"],
    "mitigation": [f"M{i:02d}" for i in range(1, 9)] + ["M_OTHER"],
}
NAMED_IDS = {
    "risk": set(TOPIC_IDS["risk"][:-1]),
    "mitigation": set(TOPIC_IDS["mitigation"][:-1]),
}
SPECIAL_IDS = {
    "risk": {"R_OTHER", "R_UNCLEAR", "R_NOISE"},
    "mitigation": {"M_OTHER", "M_UNCLEAR", "M_NOISE"},
}
OTHER_IDS = {"risk": "R_OTHER", "mitigation": "M_OTHER"}

RISK_COLOURS = {
    topic: colour for topic, colour in zip(
        TOPIC_IDS["risk"][:-1],
        ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A", "#7A6652"],
    )
}
MITIGATION_COLOURS = {
    topic: colour for topic, colour in zip(
        TOPIC_IDS["mitigation"][:-1],
        ["#009E73", "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#CC79A7", "#6A3D9A", "#7A6652"],
    )
}
OTHER_COLOUR = "#9CA3AF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row[key]
        if value in indexed:
            raise AssertionError(f"Duplicate {label} {key}: {value}")
        indexed[value] = row
    return indexed


def outcome(topic_id: str) -> str:
    for value in ("OTHER", "UNCLEAR", "NOISE"):
        if topic_id.endswith(f"_{value}"):
            return value
    return "NAMED"


def record_id_from_exclusion(row: dict[str, Any]) -> str:
    return f"{row['chunk_id']}#span#{int(row['span_id']):04d}"


def evidence_from_assignment(assignment: dict[str, Any]) -> str:
    topics = assignment.get("topics", [])
    if not topics:
        return ""
    if topics[0].get("topic_id") != assignment["primary_topic_id"]:
        raise AssertionError(f"Primary topic is not first for {assignment['record_id']}")
    return topics[0].get("evidence_quote", "")


def load_terminology() -> dict[str, Any]:
    terminology = json.loads(TERMINOLOGY_PATH.read_text(encoding="utf-8"))
    terminology["risk"]["R_OTHER"] = {
        "label": "Other or non-specific risk text",
        "short_label": "Other / non-specific",
        "definition": (
            "Residual released-reference spans that do not defensibly instantiate R01–R08. "
            "This reporting bucket includes a small number of other substantive risks as well as "
            "generic claims, headings, bare ratings, null statements, and extraction fragments."
        ),
    }
    terminology["mitigation"]["M_OTHER"] = {
        "label": "Other or non-specific control text",
        "short_label": "Other / non-specific",
        "definition": (
            "Residual released-reference spans that do not defensibly instantiate M01–M08. "
            "This reporting bucket includes other substantive controls as well as generic adequacy "
            "claims, reference-only text, mixed mechanisms, and extraction fragments."
        ),
    }
    return terminology


def validate_and_load() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, str],
]:
    immutable_paths = list(INPUT_PATHS.values()) + list(ASSIGNMENT_PATHS.values()) + [EXCLUSION_PATH, TERMINOLOGY_PATH]
    hashes_before = {str(path.relative_to(WORKSPACE)): sha256_file(path) for path in immutable_paths}

    inputs: dict[str, list[dict[str, Any]]] = {}
    assignments: dict[str, list[dict[str, Any]]] = {}
    reviews: dict[str, dict[str, dict[str, Any]]] = {}

    for kind in ("risk", "mitigation"):
        inputs[kind] = read_jsonl(INPUT_PATHS[kind])
        assignments[kind] = read_jsonl(ASSIGNMENT_PATHS[kind])
        if len(inputs[kind]) != EXPECTED_CANONICAL[kind]:
            raise AssertionError(f"Unexpected canonical {kind} inputs: {len(inputs[kind])}")
        if len(assignments[kind]) != EXPECTED_CANONICAL[kind]:
            raise AssertionError(f"Unexpected canonical {kind} assignments: {len(assignments[kind])}")

        input_index = index_unique(inputs[kind], "record_id", f"{kind} input")
        assignment_index = index_unique(assignments[kind], "record_id", f"{kind} assignment")
        if set(input_index) != set(assignment_index):
            raise AssertionError(f"{kind} input/assignment ID mismatch")

        special_records = {
            record_id for record_id, assignment in assignment_index.items()
            if assignment["primary_topic_id"] in SPECIAL_IDS[kind]
        }
        review_rows = read_jsonl(REVIEW_PATHS[kind])
        review_index = index_unique(review_rows, "record_id", f"{kind} review")
        if len(review_rows) != EXPECTED_SPECIAL[kind] or set(review_index) != special_records:
            raise AssertionError(f"{kind} review does not exactly cover all special assignments")
        for record_id, review in review_index.items():
            assignment = assignment_index[record_id]
            source = input_index[record_id]
            if review["original_topic_id"] != assignment["primary_topic_id"]:
                raise AssertionError(f"Review original topic mismatch: {record_id}")
            if review["proposed_topic_id"] not in set(TOPIC_IDS[kind]):
                raise AssertionError(f"Invalid review target for {record_id}: {review['proposed_topic_id']}")
            quote = review.get("evidence_quote", "")
            if quote and quote not in source["text"]:
                raise AssertionError(f"Non-exact review quote: {record_id}")
        reviews[kind] = review_index

    all_inputs = {
        record_id: (kind, row)
        for kind in ("risk", "mitigation")
        for record_id, row in index_unique(inputs[kind], "record_id", kind).items()
    }
    if len(all_inputs) != sum(EXPECTED_CANONICAL.values()):
        raise AssertionError("Record IDs collide across risk and mitigation inputs")

    overrides: dict[str, dict[str, Any]] = {}
    if OVERRIDE_PATH.exists():
        for row in read_jsonl(OVERRIDE_PATH):
            record_id = row["record_id"]
            if record_id in overrides:
                raise AssertionError(f"Duplicate final override: {record_id}")
            if record_id not in all_inputs:
                raise AssertionError(f"Override record does not exist: {record_id}")
            kind, source = all_inputs[record_id]
            if row["kind"] != kind:
                raise AssertionError(f"Override kind mismatch: {record_id}")
            if record_id not in reviews[kind]:
                raise AssertionError(f"Override is not a reviewed special record: {record_id}")
            if row["first_pass_topic_id"] != reviews[kind][record_id]["proposed_topic_id"]:
                raise AssertionError(f"Override first-pass mismatch: {record_id}")
            if row["final_topic_id"] not in set(TOPIC_IDS[kind]):
                raise AssertionError(f"Invalid final override topic: {record_id}")
            quote = row.get("evidence_quote", "")
            if quote and quote not in source["text"]:
                raise AssertionError(f"Non-exact override quote: {record_id}")
            overrides[record_id] = row

    exclusions = read_jsonl(EXCLUSION_PATH)
    span_exclusions = [row for row in exclusions if row.get("record_type") == "span"]
    excluded_index = index_unique(
        [{**row, "record_id": record_id_from_exclusion(row)} for row in span_exclusions],
        "record_id",
        "span exclusion",
    )
    overlaps = read_jsonl(OVERLAP_PATH)
    overlap_index = index_unique(overlaps, "record_id", "restored overlap")
    if set(overlap_index) != set(excluded_index):
        raise AssertionError("Restoration rows do not exactly cover the released span exclusions")
    overlap_counts = Counter(row["kind"] for row in overlaps)
    if dict(overlap_counts) != EXPECTED_RESTORED:
        raise AssertionError(f"Unexpected overlap counts: {dict(overlap_counts)}")

    assignment_indexes = {
        kind: index_unique(assignments[kind], "record_id", f"{kind} assignment")
        for kind in ("risk", "mitigation")
    }
    for row in overlaps:
        excluded = excluded_index[row["record_id"]]
        identity = row["source_excluded_identity"]
        for key in ("agency", "doc_id", "chunk_id", "span_id", "span_type", "overlap_unit_id", "source_gold_file"):
            if identity[key] != excluded[key]:
                raise AssertionError(f"Overlap identity mismatch ({key}): {row['record_id']}")
        if row["text"] != excluded["text"]:
            raise AssertionError(f"Overlap text mismatch: {row['record_id']}")
        counterpart = assignment_indexes[row["kind"]].get(row["counterpart_record_id"])
        if counterpart is None:
            raise AssertionError(f"Missing overlap counterpart: {row['record_id']}")
        if counterpart["primary_topic_id"] != row["inherited_topic_id"]:
            raise AssertionError(f"Inherited topic mismatch: {row['record_id']}")

    terminology = load_terminology()
    return inputs, assignments, reviews, overlaps, overrides, terminology, hashes_before


def build_rows(
    inputs: dict[str, list[dict[str, Any]]],
    assignments: dict[str, list[dict[str, Any]]],
    reviews: dict[str, dict[str, dict[str, Any]]],
    overlaps: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
    terminology: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_by_id: dict[str, dict[str, Any]] = {}

    for kind in ("risk", "mitigation"):
        input_index = index_unique(inputs[kind], "record_id", f"{kind} input")
        for assignment in assignments[kind]:
            record_id = assignment["record_id"]
            source = input_index[record_id]
            original_topic = assignment["primary_topic_id"]
            first_pass_topic = original_topic
            displayed_topic = original_topic
            decision_source = "frozen_named_assignment_unchanged"
            rationale = assignment.get("rationale", "")
            evidence_quote = evidence_from_assignment(assignment)

            if original_topic in SPECIAL_IDS[kind]:
                review = reviews[kind][record_id]
                first_pass_topic = review["proposed_topic_id"]
                displayed_topic = first_pass_topic
                decision_source = f"special_reaudit:{review['decision']}"
                rationale = review["rationale"]
                evidence_quote = review.get("evidence_quote", "")
            if record_id in overrides:
                override = overrides[record_id]
                displayed_topic = override["final_topic_id"]
                decision_source = f"final_adjudication:{override['decision']}"
                rationale = override["rationale"]
                evidence_quote = override.get("evidence_quote", "")

            if displayed_topic not in TOPIC_IDS[kind]:
                raise AssertionError(f"Disallowed displayed topic: {record_id} -> {displayed_topic}")
            if evidence_quote and evidence_quote not in source["text"]:
                raise AssertionError(f"Non-exact final evidence quote: {record_id}")

            row = {
                "record_id": record_id,
                "kind": kind,
                "agency": source["agency"],
                "doc_id": source["doc_id"],
                "chunk_id": source["chunk_id"],
                "span_id": source["span_id"],
                "released_reference_status": "canonical_retained",
                "is_overlap_restoration": False,
                "counterpart_record_id": "",
                "original_primary_topic_id": original_topic,
                "original_coding_outcome": outcome(original_topic),
                "first_pass_reaudit_topic_id": first_pass_topic,
                "displayed_topic_id": displayed_topic,
                "displayed_topic_label": terminology[kind][displayed_topic]["label"],
                "decision_source": decision_source,
                "confidence": assignment.get("confidence", ""),
                "evidence_quote": evidence_quote,
                "rationale": rationale,
                "text": source["text"],
                "text_sha256": source.get("text_sha256", hashlib.sha256(source["text"].encode()).hexdigest()),
                "source_gold_file": source.get("source_gold_file", ""),
            }
            rows.append(row)
            canonical_by_id[record_id] = row

    for overlap in overlaps:
        kind = overlap["kind"]
        counterpart = canonical_by_id[overlap["counterpart_record_id"]]
        identity = overlap["source_excluded_identity"]
        displayed_topic = counterpart["displayed_topic_id"]
        first_pass_topic = counterpart["first_pass_reaudit_topic_id"]
        inherited_changed = displayed_topic != overlap["inherited_topic_id"]
        rows.append({
            "record_id": overlap["record_id"],
            "kind": kind,
            "agency": identity["agency"],
            "doc_id": identity["doc_id"],
            "chunk_id": identity["chunk_id"],
            "span_id": identity["span_id"],
            "released_reference_status": "restored_overlap_derived_span",
            "is_overlap_restoration": True,
            "counterpart_record_id": overlap["counterpart_record_id"],
            "original_primary_topic_id": overlap["inherited_topic_id"],
            "original_coding_outcome": outcome(overlap["inherited_topic_id"]),
            "first_pass_reaudit_topic_id": first_pass_topic,
            "displayed_topic_id": displayed_topic,
            "displayed_topic_label": terminology[kind][displayed_topic]["label"],
            "decision_source": (
                "overlap_inherited_from_reaudited_counterpart" if inherited_changed
                else "overlap_inherited_from_frozen_counterpart"
            ),
            "confidence": "derived",
            "evidence_quote": "",
            "rationale": overlap["matching_rationale"] + (
                f" Final displayed topic follows the counterpart's adjudicated {displayed_topic} label."
                if inherited_changed else ""
            ),
            "text": overlap["text"],
            "text_sha256": hashlib.sha256(overlap["text"].encode()).hexdigest(),
            "source_gold_file": identity["source_gold_file"],
        })

    if len({row["record_id"] for row in rows}) != len(rows):
        raise AssertionError("Released assignment IDs are not unique")
    for kind in ("risk", "mitigation"):
        selected = [row for row in rows if row["kind"] == kind]
        if len(selected) != EXPECTED_RELEASED[kind]:
            raise AssertionError(f"Released {kind} total is {len(selected)}, not {EXPECTED_RELEASED[kind]}")
        if any(outcome(row["displayed_topic_id"]) in {"NOISE", "UNCLEAR"} for row in selected):
            raise AssertionError(f"Displayed {kind} view still contains NOISE/UNCLEAR")
    if len({row["doc_id"] for row in rows}) != EXPECTED_DOCUMENTS:
        raise AssertionError("Released view does not cover exactly 73 documents")
    return sorted(rows, key=lambda row: (row["kind"], row["record_id"]))


def topic_statistics(rows: list[dict[str, Any]], terminology: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for kind in ("risk", "mitigation"):
        selected = [row for row in rows if row["kind"] == kind]
        denominator = len(selected)
        for order, topic_id in enumerate(TOPIC_IDS[kind], 1):
            topic_rows = [row for row in selected if row["displayed_topic_id"] == topic_id]
            canonical_count = sum(not row["is_overlap_restoration"] for row in topic_rows)
            restored_count = sum(row["is_overlap_restoration"] for row in topic_rows)
            reworked_count = sum(
                not row["is_overlap_restoration"]
                and row["original_primary_topic_id"] in SPECIAL_IDS[kind]
                and row["displayed_topic_id"] != row["original_primary_topic_id"]
                for row in topic_rows
            )
            output.append({
                "kind": kind,
                "topic_order": order,
                "topic_id": topic_id,
                "topic_label": terminology[kind][topic_id]["label"],
                "topic_definition": terminology[kind][topic_id]["definition"],
                "is_named_topic": topic_id in NAMED_IDS[kind],
                "released_span_count": len(topic_rows),
                "released_span_denominator": denominator,
                "released_span_pct": round(100 * len(topic_rows) / denominator, 1),
                "canonical_display_count": canonical_count,
                "restored_overlap_count": restored_count,
                "canonical_special_reworked_into_topic": reworked_count,
                "primary_document_count": len({row["doc_id"] for row in topic_rows}),
                "document_denominator": EXPECTED_DOCUMENTS,
                "primary_document_pct": round(100 * len({row["doc_id"] for row in topic_rows}) / EXPECTED_DOCUMENTS, 1),
            })
        if sum(row["released_span_count"] for row in output if row["kind"] == kind) != denominator:
            raise AssertionError(f"Topic statistics do not sum for {kind}")
    return output


def transition_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        provenance = "restored_overlap" if row["is_overlap_restoration"] else "canonical"
        counts[(row["kind"], provenance, row["original_primary_topic_id"], row["displayed_topic_id"])] += 1
    return [
        {
            "kind": kind,
            "provenance": provenance,
            "original_or_inherited_topic_id": original,
            "displayed_topic_id": displayed,
            "span_count": count,
        }
        for (kind, provenance, original, displayed), count in sorted(counts.items())
    ]


def residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row["displayed_topic_id"] != OTHER_IDS[row["kind"]]:
            continue
        selected.append({
            "record_id": row["record_id"],
            "kind": row["kind"],
            "agency": row["agency"],
            "doc_id": row["doc_id"],
            "is_overlap_restoration": row["is_overlap_restoration"],
            "counterpart_record_id": row["counterpart_record_id"],
            "original_primary_topic_id": row["original_primary_topic_id"],
            "original_coding_outcome": row["original_coding_outcome"],
            "decision_source": row["decision_source"],
            "evidence_quote": row["evidence_quote"],
            "rationale": row["rationale"],
            "text": row["text"],
        })
    return selected


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def build_latex_table(stats: list[dict[str, Any]]) -> str:
    lines = [
        r"% Generated by build_full_reference_view.py; requires booktabs and threeparttable.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Primary topic distribution in the agency-written released reference set.}",
        r"\label{tab:released-reference-topic-distribution}",
        r"\small",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llp{9.4cm}rr}",
        r"\toprule",
        r"Type & Code & Theme & $n$ & \% \\",
        r"\midrule",
    ]
    for kind in ("risk", "mitigation"):
        kind_rows = [row for row in stats if row["kind"] == kind]
        for index, row in enumerate(kind_rows):
            label = row["topic_label"]
            lines.append(
                f"{kind.title() if index == 0 else ''} & {tex_escape(row['topic_id'])} & "
                f"{tex_escape(label)} & {row['released_span_count']} & {row['released_span_pct']:.1f} \\\\"
            )
        if kind == "risk":
            lines.append(r"\midrule")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item Percentages use all released-reference spans as denominators: 451 risks and 831 mitigations. Each span has one exclusive displayed primary topic.",
        r"\item The released totals restore 7 risk and 10 mitigation spans from overlapping source chunks. The de-duplicated analytical view remains 444 risks and 821 mitigations.",
        r"\item Other is a residual reporting bucket formed by merging the former Other, Unclear, and Noise outcomes after conservative re-audit; it is not a coherent substantive topic.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def topic_colour(kind: str, topic_id: str) -> str:
    if topic_id == OTHER_IDS[kind]:
        return OTHER_COLOUR
    return (RISK_COLOURS if kind == "risk" else MITIGATION_COLOURS)[topic_id]


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_distribution_figure(stats: list[dict[str, Any]], terminology: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.7))
    for axis, kind in zip(axes, ("risk", "mitigation")):
        kind_rows = [row for row in stats if row["kind"] == kind]
        labels = [
            f"{row['topic_id']}  {textwrap.fill(terminology[kind][row['topic_id']]['short_label'], 28)}"
            for row in kind_rows
        ]
        counts = [row["released_span_count"] for row in kind_rows]
        positions = list(range(len(kind_rows)))[::-1]
        bars = axis.barh(
            positions, counts,
            color=[topic_colour(kind, row["topic_id"]) for row in kind_rows],
            edgecolor="white", linewidth=0.8,
        )
        axis.set_yticks(positions, labels)
        axis.set_xlabel("Released-reference primary spans")
        axis.set_title(f"{kind.title()} themes ($n={EXPECTED_RELEASED[kind]}$)", fontweight="bold")
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        maximum = max(counts)
        axis.set_xlim(0, maximum * 1.23)
        for bar, row in zip(bars, kind_rows):
            axis.text(
                bar.get_width() + maximum * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{row['released_span_count']}  ({row['released_span_pct']:.1f}%)",
                va="center", fontsize=9,
            )
    fig.suptitle("Released-reference primary-topic distribution", fontsize=17, fontweight="bold", y=1.01)
    fig.text(
        0.5, -0.015,
        "Exclusive span-level labels. OTHER consolidates residual Other, Unclear, and Noise after re-audit; restored overlap spans are included.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(wspace=0.57)
    save_figure(fig, FIGURES / "released_reference_topic_distribution")


def add_wedge(axis: plt.Axes, start: float, extent: float, radius: float, width: float, colour: str) -> None:
    axis.add_patch(Wedge(
        (0, 0), radius, start - extent, start, width=width,
        facecolor=colour, edgecolor="white", linewidth=1.2,
    ))


def make_sunburst_figure(stats: list[dict[str, Any]], terminology: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 10.5))
    for axis, kind in zip(axes, ("risk", "mitigation")):
        kind_rows = [row for row in stats if row["kind"] == kind]
        total = sum(row["released_span_count"] for row in kind_rows)
        named_total = sum(row["released_span_count"] for row in kind_rows if row["is_named_topic"])
        other_total = total - named_total

        start = 90.0
        for label, count, colour in (("Named", named_total, "#475569"), ("OTHER", other_total, OTHER_COLOUR)):
            extent = 360 * count / total
            add_wedge(axis, start, extent, 0.61, 0.27, colour)
            angle = math.radians(start - extent / 2)
            if extent > 18:
                axis.text(0.47 * math.cos(angle), 0.47 * math.sin(angle), f"{label}\n{count}", ha="center", va="center", fontsize=8)
            start -= extent

        start = 90.0
        for row in kind_rows:
            count = row["released_span_count"]
            extent = 360 * count / total
            add_wedge(axis, start, extent, 1.0, 0.36, topic_colour(kind, row["topic_id"]))
            angle = math.radians(start - extent / 2)
            if extent >= 7:
                axis.text(0.82 * math.cos(angle), 0.82 * math.sin(angle), f"{row['topic_id']}\n{count}", ha="center", va="center", fontsize=7.4)
            start -= extent

        axis.text(0, 0, f"{kind.title()} spans\n$n={total}$", ha="center", va="center", fontsize=12, fontweight="bold")
        axis.set_title(f"{kind.title()} composition", fontweight="bold", pad=10)
        axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal")
        axis.axis("off")
        handles = [Rectangle((0, 0), 1, 1, facecolor=topic_colour(kind, row["topic_id"])) for row in kind_rows]
        labels = [
            f"{row['topic_id']}  {terminology[kind][row['topic_id']]['short_label']} ({row['released_span_count']})"
            for row in kind_rows
        ]
        axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=7.4)
    fig.suptitle("Released-reference topic composition", fontsize=17, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.015,
        "Area represents released-reference spans (451 risks; 831 mitigations). Inner rings distinguish named themes from the residual OTHER bucket.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(bottom=0.26, wspace=0.22)
    save_figure(fig, FIGURES / "released_reference_topic_sunburst")


def report_markdown(rows: list[dict[str, Any]], stats: list[dict[str, Any]], overrides: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Released-reference risk and mitigation topic rework",
        "",
        "## Answer and reconciliation",
        "",
        "Yes. The released-reference presentation view now sums exactly to **451 risk spans and 831 mitigation spans**:",
        "",
        "| Span type | De-duplicated canonical view | Restored overlap-derived spans | Released-reference view |",
        "|---|---:|---:|---:|",
        "| Risk | 444 | 7 | **451** |",
        "| Mitigation | 821 | 10 | **831** |",
        "",
        "The two views answer different questions. Use 451/831 when matching the cardinality of the released agency-written reference set. Use 444/821 for de-duplicated analytical claims, because the additional 17 spans arise from overlapping source chunks rather than 17 independent passages.",
        "",
        "## What changed in the special categories",
        "",
    ]
    for kind in ("risk", "mitigation"):
        canonical_special = [
            row for row in rows
            if row["kind"] == kind and not row["is_overlap_restoration"]
            and row["original_primary_topic_id"] in SPECIAL_IDS[kind]
        ]
        named = sum(row["displayed_topic_id"] in NAMED_IDS[kind] for row in canonical_special)
        other = sum(row["displayed_topic_id"] == OTHER_IDS[kind] for row in canonical_special)
        original_counts = Counter(row["original_primary_topic_id"] for row in canonical_special)
        original_text = ", ".join(f"{topic}={count}" for topic, count in sorted(original_counts.items()))
        support_verb = "supports" if named == 1 else "support"
        lines.append(
            f"- **{kind.title()}:** re-audited {len(canonical_special)} spans ({original_text}). "
            f"{named} {support_verb} a named theme; {other} lack a defensible named primary mechanism and are displayed as `{OTHER_IDS[kind]}`."
        )
    lines.extend([
        "",
        f"A second-review/adjudication layer reviewed {len(overrides)} contested first-pass decisions; "
        f"{sum(row['first_pass_topic_id'] != row['final_topic_id'] for row in overrides.values())} were changed. "
        "The row-level files preserve the frozen source label, first-pass re-audit label, final displayed label, evidence, rationale, and overlap provenance.",
        "",
        "`OTHER` is intentionally a **residual reporting bucket**, not a new coherent topic. It merges the former `OTHER`, `UNCLEAR`, and `NOISE` display outcomes as requested. It can therefore contain: (i) a substantive mechanism outside the eight-theme taxonomy; (ii) a genuine but underspecified or evenly mixed mechanism; or (iii) non-substantive text such as a heading, bare severity rating, assurance, null statement, cross-reference, or extraction fragment. These distinctions remain recoverable through `original_coding_outcome` and `decision_source`.",
        "",
        "## Released-reference distribution",
        "",
        "| Type | Code | Theme | Spans | % of released spans | Documents |",
        "|---|---|---|---:|---:|---:|",
    ])
    for kind in ("risk", "mitigation"):
        for row in [item for item in stats if item["kind"] == kind]:
            lines.append(
                f"| {kind.title()} | {row['topic_id']} | {row['topic_label']} | "
                f"{row['released_span_count']} | {row['released_span_pct']:.1f}% | {row['primary_document_count']}/73 |"
            )
    lines.extend([
        "",
        "Percentages are exclusive primary-topic shares and sum to 100% within rounding for each span type. Document counts are non-exclusive because a PIA can express several themes.",
        "",
        "## Interpretation safeguards",
        "",
        "- The unit is a labelled span, not an observed privacy incident, a unique system practice, or proof that a control is effective.",
        "- Released-reference frequencies are not independent: overlap restoration and repeated template language can repeat the same mechanism. In particular, repeated SSP/CSAM mapping language contributes multiple M08 spans.",
        "- A named reassignment was made only when the exact span states a defensible mechanism. Headings plus bare severity ratings were left in `OTHER` after independent cross-review.",
        "- Link/Sankey figures were not silently updated: the current request reconciles span topics, while a released link view would separately require restoring and validating the ten excluded overlap links.",
        "",
        "## Output guide",
        "",
        "- `full_reference_assignments.jsonl` / `.csv`: every released span with frozen, first-pass, and final topic fields.",
        "- `full_reference_topic_counts.csv`: table-ready counts, percentages, document breadth, canonical counts, and overlap additions.",
        "- `topic_transition_counts.csv`: complete frozen/inherited-to-displayed reconciliation.",
        "- `residual_other_audit.csv`: every residual `OTHER` row with its reason and exact text.",
        "- `full_reference_topic_table.tex`: paper-ready LaTeX table.",
        "- `figures/`: updated distribution and sunburst figures in PNG and PDF.",
        "- `validation_report.json`: machine-readable assertions and source hashes.",
        "",
    ])
    return "\n".join(lines)


def output_manifest(paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": "trace-full-reference-topic-view-v1",
        "files": [
            {
                "path": str(path.relative_to(RUN_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(paths)
        ],
    }


def make_zip() -> Path:
    archive = WORKSPACE / f"{RUN_ROOT.name}.zip"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(RUN_ROOT.rglob("*")):
            if (
                path.is_file()
                and not path.name.startswith(".")
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                handle.write(path, arcname=str(Path(RUN_ROOT.name) / path.relative_to(RUN_ROOT)))
    os.replace(temporary, archive)
    return archive


def main() -> None:
    inputs, assignments, reviews, overlaps, overrides, terminology, hashes_before = validate_and_load()
    rows = build_rows(inputs, assignments, reviews, overlaps, overrides, terminology)
    stats = topic_statistics(rows, terminology)
    transitions = transition_statistics(rows)
    residual = residual_rows(rows)

    assignment_jsonl = OUTPUT / "full_reference_assignments.jsonl"
    assignment_csv = OUTPUT / "full_reference_assignments.csv"
    count_csv = TABLES / "full_reference_topic_counts.csv"
    transition_csv = TABLES / "topic_transition_counts.csv"
    residual_csv = TABLES / "residual_other_audit.csv"
    latex_path = TABLES / "full_reference_topic_table.tex"
    report_path = OUTPUT / "REPORT.md"

    write_jsonl(assignment_jsonl, rows)
    write_csv(assignment_csv, rows)
    write_csv(count_csv, stats)
    write_csv(transition_csv, transitions)
    write_csv(residual_csv, residual)
    atomic_text(latex_path, build_latex_table(stats))
    atomic_text(report_path, report_markdown(rows, stats, overrides))
    make_distribution_figure(stats, terminology)
    make_sunburst_figure(stats, terminology)

    immutable_paths = list(INPUT_PATHS.values()) + list(ASSIGNMENT_PATHS.values()) + [EXCLUSION_PATH, TERMINOLOGY_PATH]
    hashes_after = {str(path.relative_to(WORKSPACE)): sha256_file(path) for path in immutable_paths}
    if hashes_after != hashes_before:
        raise AssertionError("One or more frozen source files changed during the build")

    topic_counts = {
        kind: {
            row["topic_id"]: row["released_span_count"]
            for row in stats if row["kind"] == kind
        }
        for kind in ("risk", "mitigation")
    }
    special_summary: dict[str, Any] = {}
    for kind in ("risk", "mitigation"):
        canonical_special = [
            row for row in rows
            if row["kind"] == kind and not row["is_overlap_restoration"]
            and row["original_primary_topic_id"] in SPECIAL_IDS[kind]
        ]
        special_summary[kind] = {
            "reviewed": len(canonical_special),
            "reassigned_to_named": sum(row["displayed_topic_id"] in NAMED_IDS[kind] for row in canonical_special),
            "reported_as_other": sum(row["displayed_topic_id"] == OTHER_IDS[kind] for row in canonical_special),
            "original_outcomes": dict(sorted(Counter(row["original_primary_topic_id"] for row in canonical_special).items())),
        }

    validation = {
        "schema_version": "trace-full-reference-topic-validation-v1",
        "status": "PASS",
        "view": "released_reference_primary_topic_presentation",
        "checks": {
            "canonical_input_totals": EXPECTED_CANONICAL,
            "restored_overlap_totals": EXPECTED_RESTORED,
            "released_output_totals": {
                kind: sum(row["kind"] == kind for row in rows) for kind in ("risk", "mitigation")
            },
            "expected_released_totals": EXPECTED_RELEASED,
            "unique_released_record_ids": len({row["record_id"] for row in rows}),
            "document_union": len({row["doc_id"] for row in rows}),
            "displayed_noise_count": sum(row["displayed_topic_id"].endswith("_NOISE") for row in rows),
            "displayed_unclear_count": sum(row["displayed_topic_id"].endswith("_UNCLEAR") for row in rows),
            "topic_sums": {kind: sum(topic_counts[kind].values()) for kind in ("risk", "mitigation")},
            "frozen_sources_unchanged": hashes_before == hashes_after,
        },
        "topic_counts": topic_counts,
        "special_reaudit": special_summary,
        "final_adjudication_reviewed_count": len(overrides),
        "final_adjudication_changed_count": sum(
            row["first_pass_topic_id"] != row["final_topic_id"] for row in overrides.values()
        ),
        "residual_other_counts": {
            kind: sum(row["kind"] == kind for row in residual) for kind in ("risk", "mitigation")
        },
        "source_sha256": hashes_after,
        "methodological_note": (
            "The 451/831 released-reference view includes 17 overlap-derived spans. "
            "Use the 444/821 canonical view for de-duplicated analytical claims."
        ),
    }
    validation_path = OUTPUT / "validation_report.json"
    atomic_json(validation_path, validation)

    produced = [
        assignment_jsonl, assignment_csv, count_csv, transition_csv, residual_csv,
        latex_path, report_path, validation_path,
        FIGURES / "released_reference_topic_distribution.png",
        FIGURES / "released_reference_topic_distribution.pdf",
        FIGURES / "released_reference_topic_sunburst.png",
        FIGURES / "released_reference_topic_sunburst.pdf",
    ]
    manifest_path = OUTPUT / "manifest.json"
    atomic_json(manifest_path, output_manifest(produced))
    archive = make_zip()

    print(json.dumps({
        "status": "PASS",
        "released_totals": EXPECTED_RELEASED,
        "topic_counts": topic_counts,
        "special_reaudit": special_summary,
        "archive": str(archive),
    }, indent=2))


if __name__ == "__main__":
    main()
