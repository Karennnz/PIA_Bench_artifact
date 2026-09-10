#!/usr/bin/env python3
"""Build provenance-preserving inputs for the TRACE 73-PIA content analysis.

This script is deliberately model-free.  It constructs four separate text corpora
(systems, risks, mitigations, and null findings), a canonical relation table, and
an inventory with structural validation results.

The released gold inventory is retained in the inventory.  The analysis corpora
use a canonical view which removes spans wholly confined to an actually repeated
leading unit in a later chunk, and links incident on those spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "documents": 73,
    "chunks": 77,
    "units": 3_934,
    "unit_sentences": 11_545,
    "selected_system_units": 258,
    # Two additional selected sentences duplicate gold sentence text elsewhere
    # in the same document.  They are conservatively removed along with the 188
    # directly referenced gold sentence instances.
    "system_sentences_after_masking": 2_454,
    "system_descriptive_characters_sum": 460_627,
    "released": {
        "risk": 451,
        "mitigation": 831,
        "null_finding": 95,
        "links": 829,
    },
    "canonical": {
        "risk": 444,
        "mitigation": 821,
        "null_finding": 95,
        "links": 819,
    },
}

SPAN_TYPES = ("risk", "mitigation", "null_finding")

# This is only presentation logic applied to the curated document slug.  It is
# not document-content extraction, and the source is recorded on every row.
ACRONYMS = {
    "abco": "ABCO",
    "acfo": "ACFO",
    "aff": "AFF",
    "casp": "CASP",
    "cfpb": "CFPB",
    "cms": "CMS",
    "cres": "CRES",
    "cs": "CS",
    "devsecops": "DevSecOps",
    "dms": "DMS",
    "drms": "DRMS",
    "edalr": "eDALR",
    "epacs": "EPACS",
    "fbp": "FBP",
    "fdis": "FDIS",
    "flaas": "FLAAS",
    "flas": "FLAS",
    "fld": "FLD",
    "flprac": "FLPRAC",
    "fms": "FMS",
    "fns": "FNS",
    "foia": "FOIA",
    "fpac": "FPAC",
    "fpds": "FPDS",
    "fpiss": "FPISS",
    "fps": "FPS",
    "fscoms": "FSCOMS",
    "fsfoss": "FSFOSS",
    "fsims": "FSIMS",
    "fspb": "FSPB",
    "fss": "FSS",
    "fssss": "FSSSS",
    "gis": "GIS",
    "gsa": "GSA",
    "hmda": "HMDA",
    "hud": "HUD",
    "hva": "HVA",
    "icam": "ICAM",
    "kb": "KB",
    "midas": "MIDAS",
    "ml": "ML",
    "nad": "NAD",
    "nfc": "NFC",
    "nr": "NR",
    "ocx": "OCX",
    "ola": "OLA",
    "omwi": "OMWI",
    "opm": "OPM",
    "oscar": "OSCAR",
    "pdf": "PDF",
    "pia": "PIA",
    "rma": "RMA",
    "rovr": "ROVR",
    "sap": "SAP",
    "sbl": "SBL",
    "ses": "SES",
    "snap": "SNAP",
    "usda": "USDA",
    "voc": "VoC",
    "webta": "WebTA",
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_submission = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=default_submission,
        help="Path to 2026-08-31_infpriv_submission (default: inferred).",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
            records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sentence_key(unit_id: str, local_sid: int) -> str:
    return f"{unit_id}#sentence#{local_sid:04d}"


def span_record_id(chunk_id: str, span_id: int) -> str:
    return f"{chunk_id}#span#{span_id:04d}"


def human_title_from_doc_id(doc_id: str) -> str:
    match = re.fullmatch(r"\d+_([a-z0-9]+)_(.+)", doc_id)
    require(match is not None, f"Unexpected doc_id format: {doc_id}")
    slug = match.group(2)
    words = []
    for token in slug.split("-"):
        if token in ACRONYMS:
            words.append(ACRONYMS[token])
        elif token in {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "under", "with"} and words:
            words.append(token)
        elif token == "v":
            words.append("v")
        elif token.isdigit():
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def directory_manifest(paths: list[Path], base: Path) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(base)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]
    manifest_payload = "\n".join(
        f"{item['path']}\t{item['sha256']}\t{item['bytes']}" for item in files
    )
    return {
        "file_count": len(files),
        "manifest_sha256": sha256_text(manifest_payload),
        "files": files,
    }


def main() -> None:
    args = parse_args()
    submission_dir = args.submission_dir.resolve()
    workspace_dir = submission_dir.parent
    analysis_dir = submission_dir / "content_analysis_73"
    inputs_dir = analysis_dir / "inputs"
    validation_dir = analysis_dir / "validation"

    repo_dir = submission_dir / "trace_opensource_repo"
    units_path = repo_dir / "data" / "units" / "units.jsonl"
    documents_path = repo_dir / "data" / "units" / "documents.jsonl"
    gold_dir = repo_dir / "data" / "gold" / "spans"
    attributes_dir = (
        workspace_dir
        / "2026-08-04_meeting_followup"
        / "gold_extraction"
        / "outputs_attributes"
    )

    for required_path in (units_path, documents_path, gold_dir, attributes_dir):
        require(required_path.exists(), f"Missing source: {required_path}")

    units = read_jsonl(units_path)
    chunks = read_jsonl(documents_path)
    gold_paths = sorted(gold_dir.glob("*.json"))
    attribute_paths = sorted(attributes_dir.glob("*.json"))

    require(len(units) == EXPECTED["units"], f"Expected 3,934 units; found {len(units)}")
    require(len(chunks) == EXPECTED["chunks"], f"Expected 77 chunks; found {len(chunks)}")
    require(len(gold_paths) == EXPECTED["chunks"], f"Expected 77 gold files; found {len(gold_paths)}")
    require(len(attribute_paths) == EXPECTED["documents"], f"Expected 73 attribute files; found {len(attribute_paths)}")

    units_by_id: dict[str, dict[str, Any]] = {}
    units_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    doc_agency: dict[str, str] = {}
    total_unit_sentences = 0
    for unit in units:
        unit_id = unit["unit_id"]
        doc_id = unit["doc_id"]
        require(unit_id not in units_by_id, f"Duplicate unit_id: {unit_id}")
        require(unit_id.startswith(f"{doc_id}#unit#"), f"Unit/doc mismatch: {unit_id}")
        require(unit["n_sentences"] == len(unit["sentences"]), f"Bad n_sentences: {unit_id}")
        require(
            unit["n_chars"]
            == len(" ".join(item["text"] for item in unit["sentences"])),
            f"Bad n_chars: {unit_id}",
        )
        local_sids = [item["sid"] for item in unit["sentences"]]
        require(len(local_sids) == len(set(local_sids)), f"Duplicate local sid: {unit_id}")
        units_by_id[unit_id] = unit
        units_by_doc[doc_id].append(unit)
        total_unit_sentences += len(unit["sentences"])
        if doc_id in doc_agency:
            require(doc_agency[doc_id] == unit["agency"], f"Agency mismatch: {doc_id}")
        doc_agency[doc_id] = unit["agency"]

    doc_ids = sorted(units_by_doc)
    require(len(doc_ids) == EXPECTED["documents"], f"Expected 73 documents; found {len(doc_ids)}")
    require(total_unit_sentences == EXPECTED["unit_sentences"], f"Expected 11,545 unit sentences; found {total_unit_sentences}")

    chunks_by_id: dict[str, dict[str, Any]] = {}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_sentence_to_source: dict[tuple[str, int], dict[str, Any]] = {}
    chunk_observed_unit_ids: dict[str, list[str]] = {}
    mapping_checks = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]
        require(chunk_id not in chunks_by_id, f"Duplicate chunk_id: {chunk_id}")
        require(doc_id in units_by_doc, f"Chunk has unknown doc_id: {chunk_id}")
        require(chunk["agency"] == doc_agency[doc_id], f"Chunk agency mismatch: {chunk_id}")
        require(chunk["n_sentences"] == len(chunk["sentences"]), f"Bad chunk n_sentences: {chunk_id}")

        observed_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ordered_unit_ids: list[str] = []
        for item in chunk["sentences"]:
            unit_id = item["unit_id"]
            require(unit_id in units_by_id, f"Chunk references unknown unit: {chunk_id}/{unit_id}")
            if not observed_by_unit[unit_id]:
                ordered_unit_ids.append(unit_id)
            observed_by_unit[unit_id].append(item)

        for unit_id in ordered_unit_ids:
            observed = observed_by_unit[unit_id]
            source_sentences = units_by_id[unit_id]["sentences"]
            require(len(observed) == len(source_sentences), f"Partial unit in chunk: {chunk_id}/{unit_id}")
            for chunk_sentence, source_sentence in zip(observed, source_sentences):
                require(chunk_sentence["text"] == source_sentence["text"], f"Sentence text mismatch: {chunk_id}/{unit_id}")
                key = (chunk_id, chunk_sentence["sid"])
                require(key not in chunk_sentence_to_source, f"Duplicate chunk sentence id: {key}")
                chunk_sentence_to_source[key] = {
                    "sentence_key": sentence_key(unit_id, source_sentence["sid"]),
                    "unit_id": unit_id,
                    "local_sid": source_sentence["sid"],
                    "text": source_sentence["text"],
                }
                mapping_checks += 1

        chunks_by_id[chunk_id] = chunk
        chunks_by_doc[doc_id].append(chunk)
        chunk_observed_unit_ids[chunk_id] = ordered_unit_ids

    require(sorted(chunks_by_doc) == doc_ids, "Document mismatch between units and chunks")

    # Identify only an actually repeated leading content-bearing unit.  A later
    # chunk whose first observed unit was not present earlier is not canonicalized.
    overlap_unit_by_chunk: dict[str, str] = {}
    chunk_order_by_doc: dict[str, list[str]] = {}
    for doc_id, doc_chunks in chunks_by_doc.items():
        ordered_chunks = sorted(doc_chunks, key=lambda item: item["chunk_id"])
        seen_units: set[str] = set()
        chunk_order_by_doc[doc_id] = [item["chunk_id"] for item in ordered_chunks]
        for position, chunk in enumerate(ordered_chunks):
            chunk_id = chunk["chunk_id"]
            observed_units = chunk_observed_unit_ids[chunk_id]
            require(observed_units, f"Chunk has no content-bearing unit: {chunk_id}")
            leading_unit = observed_units[0]
            if position > 0 and leading_unit in seen_units:
                overlap_unit_by_chunk[chunk_id] = leading_unit
            seen_units.update(observed_units)

    gold_by_chunk: dict[str, dict[str, Any]] = {}
    released_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    validation_problem_count = 0
    all_span_keys: set[tuple[str, int]] = set()
    gold_sentence_keys: set[str] = set()
    gold_sentence_keys_by_doc: dict[str, set[str]] = defaultdict(set)
    gold_sentence_texts_by_doc: dict[str, set[str]] = defaultdict(set)
    canonical_spans: dict[str, list[dict[str, Any]]] = {span_type: [] for span_type in SPAN_TYPES}
    exclusion_rows: list[dict[str, Any]] = []
    canonical_links: list[dict[str, Any]] = []
    released_link_keys: set[tuple[str, int, int]] = set()
    canonical_link_keys: set[tuple[str, int, int]] = set()
    endpoint_checks = 0
    exact_span_text_checks = 0
    exact_span_text_matches = 0
    span_text_reconstruction_mismatches: list[dict[str, Any]] = []

    for gold_path in gold_paths:
        gold = read_json(gold_path)
        chunk_id = gold["chunk_id"]
        doc_id = gold["doc_id"]
        require(chunk_id in chunks_by_id, f"Gold has unknown chunk: {gold_path}")
        require(chunk_id not in gold_by_chunk, f"Duplicate gold chunk: {chunk_id}")
        require(doc_id == chunks_by_id[chunk_id]["doc_id"], f"Gold doc mismatch: {chunk_id}")
        require(gold["agency"] == doc_agency[doc_id], f"Gold agency mismatch: {chunk_id}")
        require(gold["n_sentences"] == chunks_by_id[chunk_id]["n_sentences"], f"Gold sentence-count mismatch: {chunk_id}")
        validation_problem_count += len(gold.get("validation_problems", []))

        spans_by_id: dict[int, dict[str, Any]] = {}
        dropped_span_ids: set[int] = set()
        overlap_unit = overlap_unit_by_chunk.get(chunk_id)

        for span in gold["spans"]:
            span_id = span["span_id"]
            span_type = span["type"]
            key = (chunk_id, span_id)
            require(span_type in SPAN_TYPES, f"Unknown span type: {chunk_id}/{span_id}/{span_type}")
            require(key not in all_span_keys, f"Duplicate span key: {key}")
            require(span_id not in spans_by_id, f"Duplicate span_id within chunk: {key}")
            require(span["sentence_ids"], f"Span without sentences: {key}")
            require(span["unit_ids"], f"Span without units: {key}")

            mapped_sentences = []
            for chunk_sid in span["sentence_ids"]:
                sentence_map_key = (chunk_id, chunk_sid)
                require(sentence_map_key in chunk_sentence_to_source, f"Unknown gold sentence: {sentence_map_key}")
                mapped = chunk_sentence_to_source[sentence_map_key]
                mapped_sentences.append(mapped)
                gold_sentence_keys.add(mapped["sentence_key"])
                gold_sentence_keys_by_doc[doc_id].add(mapped["sentence_key"])
                gold_sentence_texts_by_doc[doc_id].add(mapped["text"])

            mapped_unit_ids = list(dict.fromkeys(item["unit_id"] for item in mapped_sentences))
            require(mapped_unit_ids == span["unit_ids"], f"Gold unit provenance mismatch: {key}")
            source_text = " ".join(item["text"] for item in mapped_sentences)
            exact_span_text_checks += 1
            if source_text == span["text"]:
                exact_span_text_matches += 1
            else:
                # A small number of released span texts contain deterministic OCR
                # cleanup relative to the sentence source (for example "T here"
                # -> "There").  Preserve the released span verbatim, retain its
                # sentence keys, and expose the difference rather than silently
                # rewriting either representation.
                span_text_reconstruction_mismatches.append(
                    {
                        "chunk_id": chunk_id,
                        "span_id": span_id,
                        "span_type": span_type,
                        "doc_id": doc_id,
                        "gold_span_text": span["text"],
                        "source_sentences_space_joined": source_text,
                        "sentence_keys": [item["sentence_key"] for item in mapped_sentences],
                    }
                )

            released_counts[span_type] += 1
            spans_by_id[span_id] = span
            all_span_keys.add(key)

            is_overlap_duplicate = (
                overlap_unit is not None
                and set(span["unit_ids"]) == {overlap_unit}
            )
            if is_overlap_duplicate:
                dropped_span_ids.add(span_id)
                exclusion_rows.append(
                    {
                        "record_type": "span",
                        "reason": "confined_to_repeated_leading_unit_of_later_chunk",
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "agency": gold["agency"],
                        "overlap_unit_id": overlap_unit,
                        "span_id": span_id,
                        "span_type": span_type,
                        "text": span["text"],
                        "source_gold_file": str(gold_path.relative_to(workspace_dir)),
                    }
                )
                continue

            record_id = span_record_id(chunk_id, span_id)
            record = {
                "record_id": record_id,
                "key": {"chunk_id": chunk_id, "span_id": span_id},
                "chunk_id": chunk_id,
                "span_id": span_id,
                "span_type": span_type,
                "doc_id": doc_id,
                "agency": gold["agency"],
                "text": span["text"],
                "topic_input_text": span["text"],
                "text_sha256": sha256_text(span["text"]),
                "sentence_ids": span["sentence_ids"],
                "sentence_keys": [item["sentence_key"] for item in mapped_sentences],
                "unit_ids": span["unit_ids"],
                "section": span.get("section"),
                "explanation": span.get("explanation"),
                "confidence": span.get("confidence"),
                "source_gold_file": str(gold_path.relative_to(workspace_dir)),
                "canonicalization": "retained",
                "topic_input_policy": "verbatim_span_text_only; section_and_explanation_are_provenance_not_model_input",
            }
            canonical_spans[span_type].append(record)
            canonical_counts[span_type] += 1

        for link in gold["links"]:
            risk_id = link["risk_span_id"]
            mitigation_id = link["mitigation_span_id"]
            link_key = (chunk_id, risk_id, mitigation_id)
            require(link_key not in released_link_keys, f"Duplicate released link: {link_key}")
            require(risk_id in spans_by_id, f"Link has unknown risk endpoint: {link_key}")
            require(mitigation_id in spans_by_id, f"Link has unknown mitigation endpoint: {link_key}")
            require(spans_by_id[risk_id]["type"] == "risk", f"Link risk endpoint has wrong type: {link_key}")
            require(spans_by_id[mitigation_id]["type"] == "mitigation", f"Link mitigation endpoint has wrong type: {link_key}")
            released_link_keys.add(link_key)
            released_counts["links"] += 1
            endpoint_checks += 2

            if risk_id in dropped_span_ids or mitigation_id in dropped_span_ids:
                exclusion_rows.append(
                    {
                        "record_type": "link",
                        "reason": "incident_on_excluded_overlap_span",
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "agency": gold["agency"],
                        "risk_span_id": risk_id,
                        "mitigation_span_id": mitigation_id,
                        "confidence": link.get("confidence"),
                        "source_gold_file": str(gold_path.relative_to(workspace_dir)),
                    }
                )
                continue

            require(link_key not in canonical_link_keys, f"Duplicate canonical link: {link_key}")
            canonical_link_keys.add(link_key)
            canonical_counts["links"] += 1
            canonical_links.append(
                {
                    "record_id": f"{chunk_id}#link#risk-{risk_id:04d}#mitigation-{mitigation_id:04d}",
                    "key": {
                        "chunk_id": chunk_id,
                        "risk_span_id": risk_id,
                        "mitigation_span_id": mitigation_id,
                    },
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "agency": gold["agency"],
                    "risk_key": {"chunk_id": chunk_id, "span_id": risk_id},
                    "mitigation_key": {"chunk_id": chunk_id, "span_id": mitigation_id},
                    "risk_record_id": span_record_id(chunk_id, risk_id),
                    "mitigation_record_id": span_record_id(chunk_id, mitigation_id),
                    "confidence": link.get("confidence"),
                    "source_gold_file": str(gold_path.relative_to(workspace_dir)),
                    "canonicalization": "retained",
                }
            )

        gold_by_chunk[chunk_id] = gold

    require(sorted(gold_by_chunk) == sorted(chunks_by_id), "Gold/chunk coverage mismatch")
    require(dict(released_counts) == EXPECTED["released"], f"Released inventory mismatch: {dict(released_counts)}")
    require(dict(canonical_counts) == EXPECTED["canonical"], f"Canonical inventory mismatch: {dict(canonical_counts)}")

    # Load and reconcile the eight-attribute unit assignments.
    attributes_by_unit: dict[str, dict[str, Any]] = {}
    attribute_docs: set[str] = set()
    attribute_prompt_versions: Counter[str] = Counter()
    attribute_units_not_returned = 0
    for attribute_path in attribute_paths:
        result = read_json(attribute_path)
        doc_id = result["document_id"]
        require(doc_id not in attribute_docs, f"Duplicate attribute document: {doc_id}")
        require(doc_id in units_by_doc, f"Attributes have unknown document: {doc_id}")
        attribute_docs.add(doc_id)
        attribute_prompt_versions[result.get("prompt_version", "unknown")] += 1
        attribute_units_not_returned += len(result.get("units_not_returned", []))
        for assignment in result["units"]:
            unit_id = assignment["unit_id"]
            require(unit_id in units_by_id, f"Attributes reference unknown unit: {unit_id}")
            require(units_by_id[unit_id]["doc_id"] == doc_id, f"Attribute unit/doc mismatch: {unit_id}")
            require(unit_id not in attributes_by_unit, f"Duplicate attribute unit: {unit_id}")
            require(len(assignment["attributes"]) == len(set(assignment["attributes"])), f"Duplicate attribute label: {unit_id}")
            attributes_by_unit[unit_id] = assignment

    require(attribute_docs == set(doc_ids), "Attribute document coverage mismatch")
    require(len(attributes_by_unit) == len(units_by_id), f"Attribute unit coverage mismatch: {len(attributes_by_unit)}/{len(units_by_id)}")
    require(attribute_units_not_returned == 0, f"Attribute units_not_returned is nonzero: {attribute_units_not_returned}")

    selected_unit_ids = {
        unit_id
        for unit_id, assignment in attributes_by_unit.items()
        if {"context", "purpose"}.issubset(set(assignment["attributes"]))
    }
    require(len(selected_unit_ids) == EXPECTED["selected_system_units"], f"Expected 258 selected units; found {len(selected_unit_ids)}")

    system_records: list[dict[str, Any]] = []
    total_system_sentences_before = 0
    total_system_sentences_after = 0
    total_system_chars_sum = 0
    total_system_chars_space_joined = 0
    selected_gold_sentence_keys: set[str] = set()
    selected_duplicate_gold_text_sentence_keys: set[str] = set()
    system_docs_with_masking = 0
    profile_text_gold_sentence_key_leaks: list[str] = []
    profile_exact_gold_text_duplicates: list[dict[str, Any]] = []

    for doc_id in doc_ids:
        title = human_title_from_doc_id(doc_id)
        source_units: list[dict[str, Any]] = []
        descriptive_sentences: list[dict[str, Any]] = []
        masked_keys_for_doc: list[str] = []
        directly_masked_keys_for_doc: list[str] = []
        duplicate_text_masked_keys_for_doc: list[str] = []

        for unit in units_by_doc[doc_id]:
            unit_id = unit["unit_id"]
            if unit_id not in selected_unit_ids:
                continue
            assignment = attributes_by_unit[unit_id]
            retained_for_unit = []
            masked_for_unit = []
            directly_masked_for_unit = []
            duplicate_text_masked_for_unit = []
            for source_sentence in unit["sentences"]:
                stable_key = sentence_key(unit_id, source_sentence["sid"])
                total_system_sentences_before += 1
                is_direct_gold_sentence = stable_key in gold_sentence_keys
                is_duplicate_gold_text = (
                    not is_direct_gold_sentence
                    and source_sentence["text"] in gold_sentence_texts_by_doc[doc_id]
                )
                if is_direct_gold_sentence or is_duplicate_gold_text:
                    if is_direct_gold_sentence:
                        selected_gold_sentence_keys.add(stable_key)
                        directly_masked_keys_for_doc.append(stable_key)
                        directly_masked_for_unit.append(stable_key)
                    else:
                        selected_duplicate_gold_text_sentence_keys.add(stable_key)
                        duplicate_text_masked_keys_for_doc.append(stable_key)
                        duplicate_text_masked_for_unit.append(stable_key)
                    masked_keys_for_doc.append(stable_key)
                    masked_for_unit.append(stable_key)
                    continue
                sentence_row = {
                    "sentence_key": stable_key,
                    "unit_id": unit_id,
                    "sid": source_sentence["sid"],
                    "text": source_sentence["text"],
                }
                descriptive_sentences.append(sentence_row)
                retained_for_unit.append(stable_key)

            source_units.append(
                {
                    "unit_id": unit_id,
                    "attributes": assignment["attributes"],
                    "attribute_confidence": assignment.get("confidence"),
                    "unit_type": unit["unit_type"],
                    "source_heading": unit.get("heading"),
                    "source_question": unit.get("question"),
                    "retained_sentence_keys": retained_for_unit,
                    "masked_sentence_keys": masked_for_unit,
                    "directly_referenced_gold_sentence_keys": directly_masked_for_unit,
                    "duplicate_gold_text_sentence_keys": duplicate_text_masked_for_unit,
                }
            )

        require(source_units, f"No context AND purpose units for {doc_id}")
        require(descriptive_sentences, f"No descriptive sentences after masking for {doc_id}")
        if masked_keys_for_doc:
            system_docs_with_masking += 1

        descriptive_text = " ".join(item["text"] for item in descriptive_sentences)
        topic_input_text = f"{title}\n\n{descriptive_text}"
        retained_keys = {item["sentence_key"] for item in descriptive_sentences}
        leaked_keys = retained_keys.intersection(gold_sentence_keys_by_doc[doc_id])
        profile_text_gold_sentence_key_leaks.extend(sorted(leaked_keys))
        retained_keys_by_text: dict[str, list[str]] = defaultdict(list)
        for item in descriptive_sentences:
            retained_keys_by_text[item["text"]].append(item["sentence_key"])
        duplicate_gold_texts = set(retained_keys_by_text).intersection(
            gold_sentence_texts_by_doc[doc_id]
        )
        for duplicate_text in sorted(duplicate_gold_texts):
            profile_exact_gold_text_duplicates.append(
                {
                    "doc_id": doc_id,
                    "text": duplicate_text,
                    "retained_sentence_keys": retained_keys_by_text[duplicate_text],
                }
            )

        n_char_sum = sum(len(item["text"]) for item in descriptive_sentences)
        total_system_sentences_after += len(descriptive_sentences)
        total_system_chars_sum += n_char_sum
        total_system_chars_space_joined += len(descriptive_text)

        system_records.append(
            {
                "record_id": doc_id,
                "doc_id": doc_id,
                "agency": doc_agency[doc_id],
                "title": title,
                "title_source": "derived_from_doc_id",
                "selection_rule": "unit attributes include both context AND purpose",
                "masking_rule": "remove every source sentence referenced by any released risk, mitigation, or null-finding span, plus any selected sentence that exactly duplicates gold sentence text elsewhere in the same document",
                "topic_input_policy": "title plus retained sentence text only; source headings and questions are provenance and are not in topic_input_text",
                "topic_input_text": topic_input_text,
                "topic_input_sha256": sha256_text(topic_input_text),
                "descriptive_text": descriptive_text,
                "descriptive_text_sha256": sha256_text(descriptive_text),
                "n_selected_units": len(source_units),
                "n_sentences_before_masking": len(descriptive_sentences) + len(masked_keys_for_doc),
                "n_sentences_masked": len(masked_keys_for_doc),
                "n_directly_referenced_gold_sentences_masked": len(directly_masked_keys_for_doc),
                "n_duplicate_gold_text_sentences_masked": len(duplicate_text_masked_keys_for_doc),
                "n_sentences": len(descriptive_sentences),
                "n_descriptive_characters_sum": n_char_sum,
                "n_descriptive_characters_space_joined": len(descriptive_text),
                "selected_unit_ids": [item["unit_id"] for item in source_units],
                "masked_sentence_keys": masked_keys_for_doc,
                "directly_referenced_gold_sentence_keys": directly_masked_keys_for_doc,
                "duplicate_gold_text_sentence_keys": duplicate_text_masked_keys_for_doc,
                "sentences": descriptive_sentences,
                "source_units": source_units,
            }
        )

    require(len(system_records) == EXPECTED["documents"], f"Expected 73 system records; found {len(system_records)}")
    require(total_system_sentences_after == EXPECTED["system_sentences_after_masking"], f"Expected 2,454 post-mask system sentences; found {total_system_sentences_after}")
    require(
        total_system_chars_sum == EXPECTED["system_descriptive_characters_sum"],
        f"Expected 460,627 descriptive source characters; found {total_system_chars_sum}",
    )
    require(
        total_system_chars_space_joined
        == total_system_chars_sum + total_system_sentences_after - len(system_records),
        "Unexpected descriptive-text joining arithmetic",
    )
    require(not profile_text_gold_sentence_key_leaks, f"Gold sentence keys remain in profiles: {profile_text_gold_sentence_key_leaks[:5]}")
    require(
        not profile_exact_gold_text_duplicates,
        "A retained system sentence exactly duplicates gold sentence text: "
        f"{profile_exact_gold_text_duplicates[:3]}",
    )

    # Every canonical link must resolve through the composite (chunk_id, span_id)
    # key in the canonical risk and mitigation files.
    canonical_risk_keys = {(item["chunk_id"], item["span_id"]) for item in canonical_spans["risk"]}
    canonical_mitigation_keys = {(item["chunk_id"], item["span_id"]) for item in canonical_spans["mitigation"]}
    for link in canonical_links:
        require((link["chunk_id"], link["risk_key"]["span_id"]) in canonical_risk_keys, f"Unresolved canonical risk link: {link['record_id']}")
        require((link["chunk_id"], link["mitigation_key"]["span_id"]) in canonical_mitigation_keys, f"Unresolved canonical mitigation link: {link['record_id']}")

    output_paths = {
        "systems": inputs_dir / "systems.jsonl",
        "risks": inputs_dir / "risks.jsonl",
        "mitigations": inputs_dir / "mitigations.jsonl",
        "null_findings": inputs_dir / "null_findings.jsonl",
        "links": inputs_dir / "links.jsonl",
        "canonical_exclusions": validation_dir / "canonical_exclusions.jsonl",
        "span_text_reconstruction_mismatches": validation_dir / "span_text_reconstruction_mismatches.jsonl",
    }
    write_jsonl(output_paths["systems"], system_records)
    write_jsonl(output_paths["risks"], canonical_spans["risk"])
    write_jsonl(output_paths["mitigations"], canonical_spans["mitigation"])
    write_jsonl(output_paths["null_findings"], canonical_spans["null_finding"])
    write_jsonl(output_paths["links"], canonical_links)
    write_jsonl(output_paths["canonical_exclusions"], exclusion_rows)
    write_jsonl(output_paths["span_text_reconstruction_mismatches"], span_text_reconstruction_mismatches)

    output_manifest = {
        name: {
            "path": str(path.relative_to(workspace_dir)),
            "records": sum(1 for _ in path.open("r", encoding="utf-8")),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in output_paths.items()
    }

    overlap_exclusion_counts = Counter(
        item.get("span_type", "links") for item in exclusion_rows
    )
    agency_document_counts = Counter(doc_agency.values())
    agency_system_unit_counts = Counter()
    agency_system_sentence_counts = Counter()
    for record in system_records:
        agency_system_unit_counts[record["agency"]] += record["n_selected_units"]
        agency_system_sentence_counts[record["agency"]] += record["n_sentences"]

    source_manifest = {
        "units": directory_manifest([units_path], workspace_dir),
        "documents": directory_manifest([documents_path], workspace_dir),
        "gold": directory_manifest(gold_paths, workspace_dir),
        "attributes": directory_manifest(attribute_paths, workspace_dir),
    }

    inventory = {
        "schema_version": "trace-content-analysis-inputs-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "73 cleaned TRACE PIA documents only",
        "model_calls": 0,
        "source_manifests": source_manifest,
        "source_inventory": {
            "documents": len(doc_ids),
            "chunks": len(chunks),
            "units": len(units),
            "unit_sentences": total_unit_sentences,
            "chunk_sentence_occurrences_including_overlap": mapping_checks,
            "gold_files": len(gold_paths),
            "attribute_files": len(attribute_paths),
            "attribute_assignments": len(attributes_by_unit),
            "attribute_prompt_versions": dict(sorted(attribute_prompt_versions.items())),
            "attribute_units_not_returned": attribute_units_not_returned,
            "gold_validation_problems": validation_problem_count,
            "documents_by_agency": dict(sorted(agency_document_counts.items())),
        },
        "released_inventory": {
            "risks": released_counts["risk"],
            "mitigations": released_counts["mitigation"],
            "null_findings": released_counts["null_finding"],
            "links": released_counts["links"],
            "description": "Counts in the released TRACE gold files before overlap canonicalization.",
        },
        "canonical_inventory": {
            "risks": canonical_counts["risk"],
            "mitigations": canonical_counts["mitigation"],
            "null_findings": canonical_counts["null_finding"],
            "links": canonical_counts["links"],
            "description": "Counts used for non-duplicated topic and relation analysis.",
        },
        "canonicalization": {
            "rule": "For a later chunk, identify its first content-bearing unit; only if that unit occurred in an earlier chunk of the same document, exclude spans confined wholly to that unit and links incident on those spans.",
            "overlap_unit_by_chunk": dict(sorted(overlap_unit_by_chunk.items())),
            "excluded_records": len(exclusion_rows),
            "excluded_by_type": dict(sorted(overlap_exclusion_counts.items())),
            "diagnostic_file": str(output_paths["canonical_exclusions"].relative_to(workspace_dir)),
        },
        "system_profiles": {
            "records": len(system_records),
            "selection_rule": "context AND purpose",
            "selected_units": len(selected_unit_ids),
            "sentences_before_masking": total_system_sentences_before,
            "unique_gold_sentences_in_release": len(gold_sentence_keys),
            "unique_gold_sentences_intersecting_selected_units": len(selected_gold_sentence_keys),
            "selected_sentences_removed_for_duplicate_gold_text": len(selected_duplicate_gold_text_sentence_keys),
            "documents_with_at_least_one_masked_gold_sentence": system_docs_with_masking,
            "sentences_after_masking": total_system_sentences_after,
            "descriptive_characters_sum": total_system_chars_sum,
            "descriptive_characters_space_joined": total_system_chars_space_joined,
            "selected_units_by_agency": dict(sorted(agency_system_unit_counts.items())),
            "sentences_after_masking_by_agency": dict(sorted(agency_system_sentence_counts.items())),
            "title_source": "derived_from_doc_id",
            "topic_input_construction": "human-readable derived title + two newlines + retained sentence texts joined by one space",
            "headings_and_questions_in_topic_input": False,
        },
        "provenance_checks": {
            "chunk_sentence_to_unit_sentence_mappings_checked": mapping_checks,
            "exact_span_text_reconstructions_checked": exact_span_text_checks,
            "exact_span_text_reconstructions_matched": exact_span_text_matches,
            "span_text_reconstruction_mismatches": len(span_text_reconstruction_mismatches),
            "link_endpoints_checked": endpoint_checks,
            "composite_span_key": ["chunk_id", "span_id"],
            "all_system_gold_sentence_keys_removed": not profile_text_gold_sentence_key_leaks,
            "no_retained_system_sentence_exactly_duplicates_gold_sentence_text": not profile_exact_gold_text_duplicates,
            "all_canonical_links_resolve": True,
            "all_73_documents_have_system_profiles": len(system_records) == 73,
            "all_units_have_attribute_assignments": len(attributes_by_unit) == len(units_by_id),
        },
        "output_manifest": output_manifest,
    }

    inventory_path = validation_dir / "inventory.json"
    write_json(inventory_path, inventory)
    print(json.dumps({
        "inventory": str(inventory_path),
        "released": inventory["released_inventory"],
        "canonical": inventory["canonical_inventory"],
        "system_profiles": inventory["system_profiles"],
        "outputs": output_manifest,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
