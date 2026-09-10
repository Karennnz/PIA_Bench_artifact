#!/usr/bin/env python3
"""Assign frozen topics to generated PIA statements with TopicGPT.

This is TopicGPT's *assignment* stage. It converts each frozen JSON codebook
to the hierarchy format consumed by ``topicgpt_python.assign_topics``, converts
deduplicated generated statements to TopicGPT's ``{"id", "text"}`` JSONL
format, calls the official TopicGPT package, and validates the raw responses.
It deliberately does not regenerate or refine the frozen topics.

TopicGPT is prompt based and therefore always needs an assignment prompt. If
``--prompts-dir`` is omitted, this script uses a versioned PIA-specific prompt
embedded below. An external prompt must contain ``{tree}`` and ``{Document}``
and must request the line-oriented JSON format documented by ``--help``.

Use ``--prepare-only`` first to materialize and inspect the exact TopicGPT
inputs, hierarchy, and prompt without importing TopicGPT or making API calls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import string
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal


Kind = Literal["risk", "mitigation"]
TopicGPTAssigner = Callable[..., Any]

PROMPT_FILES: dict[str, str] = {
    "risk": "topic_assignment_risk.txt",
    "mitigation": "topic_assignment_mitigation.txt",
}
PREDICTION_FIELDS: tuple[tuple[str, Kind], ...] = (
    ("risks", "risk"),
    ("mitigations", "mitigation"),
)
BATCH_MAX_RECORDS = 40
BATCH_MAX_CHARS = 20_000
CACHE_FORMAT_VERSION = "assign-generated-topicgpt-v3"
OUTPUT_FORMAT_VERSION = "generated-topic-assignment-v2"
BUILTIN_PROMPT_VERSION = "pia-topicgpt-assignment-v1"
BUILTIN_CORRECTION_PROMPT_VERSION = "pia-topicgpt-correction-v1"
BUILTIN_EXAMPLE_TOPIC_IDS: dict[Kind, set[str]] = {
    "risk": {"R01", "R02", "R05"},
    "mitigation": {"M01", "M02", "M05"},
}
CONFIDENCE_VALUES = {"high", "medium", "low"}
TOPIC_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
RESPONSE_LINE_PATTERN = re.compile(
    r"^\[1\]\s+(?P<topic_id>[A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*"
    r"(?P<payload>\{.*\})\s*$"
)


# The double braces are intentional: TopicGPT applies str.format() to the
# prompt, so JSON examples must escape their literal braces.
BUILTIN_PROMPTS: dict[Kind, str] = {
    "risk": """You will receive one generated privacy-risk statement and a frozen topic hierarchy. Assign the statement to the most relevant topics in that hierarchy. Do not invent topics or evidence.

[Topic hierarchy]
{tree}

[PIA examples]
Example 1
Document: The service may collect more personal details than are necessary for the stated purpose.
Assignment:
[1] R01: {{"confidence":"high","rationale":"The statement explicitly describes collection beyond necessity.","evidence_quote":"collect more personal details than are necessary"}}

Example 2
Document: An attacker who steals an administrator credential could view the case records.
Assignment:
[1] R02: {{"confidence":"high","rationale":"The statement describes access by an unauthorized attacker.","evidence_quote":"An attacker who steals an administrator credential could view the case records"}}

Example 3
Document: Records may remain in the system after the approved retention period ends.
Assignment:
[1] R05: {{"confidence":"high","rationale":"The statement describes retention beyond an approved period.","evidence_quote":"remain in the system after the approved retention period ends"}}

[Instructions]
1. Assign one primary topic and, only when independently supported, at most two secondary topics.
2. Put the primary topic on the first line. Put each secondary topic on a later line.
3. Use only exact topic IDs from the hierarchy. Never invent, rename, or combine IDs.
4. For every topic, copy one non-empty, contiguous evidence quote exactly from the document, preserving spelling, capitalization, and punctuation.
5. Use confidence high, medium, or low. Give a short statement-specific rationale.
6. Use an OTHER, UNCLEAR, or NOISE topic from the hierarchy when its definition applies; do not force a substantive topic.
7. Return only assignment lines. Each line must be valid JSON after the colon and must use exactly this schema:
[1] TOPIC_ID: {{"confidence":"high|medium|low","rationale":"short reason","evidence_quote":"exact document substring"}}

[Document]
{Document}

Double-check every topic ID against the hierarchy and every quote against the document.
""",
    "mitigation": """You will receive one generated privacy-mitigation statement and a frozen topic hierarchy. Assign the statement to the most relevant topics in that hierarchy. Do not invent topics or evidence.

[Topic hierarchy]
{tree}

[PIA examples]
Example 1
Document: Only the minimum fields needed for the service are collected.
Assignment:
[1] M01: {{"confidence":"high","rationale":"The control minimizes the fields collected.","evidence_quote":"Only the minimum fields needed for the service are collected"}}

Example 2
Document: Multi-factor authentication and role-based access restrict records to authorized staff.
Assignment:
[1] M02: {{"confidence":"high","rationale":"The control verifies users and restricts access by role.","evidence_quote":"Multi-factor authentication and role-based access restrict records to authorized staff"}}

Example 3
Document: Records are automatically deleted when the approved retention period expires.
Assignment:
[1] M05: {{"confidence":"high","rationale":"The control deletes records at the end of the retention period.","evidence_quote":"automatically deleted when the approved retention period expires"}}

[Instructions]
1. Assign one primary topic and, only when independently supported, at most two secondary topics.
2. Put the primary topic on the first line. Put each secondary topic on a later line.
3. Use only exact topic IDs from the hierarchy. Never invent, rename, or combine IDs.
4. For every topic, copy one non-empty, contiguous evidence quote exactly from the document, preserving spelling, capitalization, and punctuation.
5. Use confidence high, medium, or low. Give a short statement-specific rationale.
6. Use an OTHER, UNCLEAR, or NOISE topic from the hierarchy when its definition applies; do not force a substantive topic.
7. Return only assignment lines. Each line must be valid JSON after the colon and must use exactly this schema:
[1] TOPIC_ID: {{"confidence":"high|medium|low","rationale":"short reason","evidence_quote":"exact document substring"}}

[Document]
{Document}

Double-check every topic ID against the hierarchy and every quote against the document.
""",
}


def builtin_correction_prompt(kind: Kind) -> str:
    """Return a TopicGPT prompt for correcting a rejected assignment."""
    noun = "privacy-risk" if kind == "risk" else "privacy-mitigation"
    return f"""A previous TopicGPT assignment for one {noun} statement failed deterministic validation. Reassign it using the frozen hierarchy and the validation feedback in the correction packet.

[Topic hierarchy]
{{tree}}

[Instructions]
1. The correction packet contains ORIGINAL STATEMENT, PREVIOUS RESPONSE, and VALIDATION ERRORS sections.
2. Classify only the ORIGINAL STATEMENT. The other sections are feedback, not source evidence.
3. Return one primary topic first and at most two independently supported secondary topics.
4. Use only exact topic IDs from the hierarchy.
5. Every evidence_quote must be a non-empty, contiguous, character-for-character substring of the ORIGINAL STATEMENT.
6. Return only assignment lines in this exact form, with valid JSON after the colon:
[1] TOPIC_ID: {{{{"confidence":"high|medium|low","rationale":"short reason","evidence_quote":"exact substring of ORIGINAL STATEMENT"}}}}

[Correction packet]
{{Document}}

Check the IDs, number of topics, JSON syntax, and exact quotes before answering.
"""


def norm(text: Any) -> str:
    """Normalize whitespace exactly as used when constructing record IDs."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def temperature_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 2.0:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
    return parsed


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"record in {path} line {line_number} is not an object")
            rows.append(row)
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_text_if_changed(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def validate_topicgpt_prompt(prompt: str, source: str) -> None:
    if not prompt.strip():
        raise ValueError(f"TopicGPT prompt is empty: {source}")
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(prompt)
            if field_name is not None
        }
    except ValueError as exc:
        raise ValueError(f"invalid braces in TopicGPT prompt {source}: {exc}") from exc
    required = {"tree", "Document"}
    if not required.issubset(fields):
        missing = ", ".join(sorted(required - fields))
        raise ValueError(f"TopicGPT prompt {source} is missing placeholder(s): {missing}")
    unknown = fields - required
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"TopicGPT prompt {source} has unsupported placeholder(s): {names}")
    for token in ('"confidence"', '"rationale"', '"evidence_quote"'):
        if token not in prompt:
            raise ValueError(
                f"TopicGPT prompt {source} must request the parser field {token}; "
                "use the built-in prompt as a template"
            )
    try:
        prompt.format(tree="[1] T01: Example", Document="Example document")
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            f"TopicGPT prompt {source} cannot be formatted; escape literal JSON braces "
            f"as '{{{{' and '}}}}': {exc}"
        ) from exc


def load_prompt(prompts_dir: Path | None, kind: Kind) -> tuple[str, str, str]:
    if prompts_dir is None:
        prompt = BUILTIN_PROMPTS[kind]
        source = f"builtin:{BUILTIN_PROMPT_VERSION}:{kind}"
    else:
        path = prompts_dir / PROMPT_FILES[kind]
        try:
            prompt = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError(f"prompt file not found: {path}") from exc
        source = f"external:{path.name}"
    validate_topicgpt_prompt(prompt, source)
    return prompt, sha256_text(prompt), source


def load_codebook(codebooks_dir: Path, kind: Kind) -> tuple[list[dict[str, Any]], str]:
    path = codebooks_dir / f"{kind}.json"
    book = read_json(path)
    if not isinstance(book, dict) or not isinstance(book.get("topics"), list):
        raise ValueError(f"{path} must contain a JSON object with a topics list")

    topics = book["topics"]
    topic_ids: list[str] = []
    for index, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            raise ValueError(f"topic {index} in {path} is not an object")
        topic_id = topic.get("topic_id")
        if not isinstance(topic_id, str) or not TOPIC_ID_PATTERN.fullmatch(topic_id):
            raise ValueError(
                f"topic {index} in {path} has an invalid TopicGPT-safe topic_id; "
                "use letters, numbers, underscores, or hyphens"
            )
        topic_ids.append(topic_id)
    if not topic_ids:
        raise ValueError(f"{path} contains no topics")
    if len(topic_ids) != len(set(topic_ids)):
        raise ValueError(f"duplicate topic_id values in {path}")

    canonical = json.dumps(topics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return topics, sha256_text(canonical)


def topicgpt_hierarchy(topics: list[dict[str, Any]]) -> str:
    """Convert a flat frozen codebook to TopicGPT's count-bearing Markdown."""
    def rules(value: Any) -> list[str]:
        if isinstance(value, str):
            return [norm(value)] if norm(value) else []
        if isinstance(value, list):
            return [norm(item) for item in value if norm(item)]
        return []

    lines: list[str] = []
    for topic in topics:
        pieces = [
            f"Name: {norm(topic.get('name'))}",
            f"Definition: {norm(topic.get('definition'))}",
        ]
        inclusion = rules(topic.get("inclusion_rule"))
        exclusion = rules(topic.get("exclusion_rule"))
        if inclusion:
            pieces.append("Include: " + "; ".join(inclusion))
        if exclusion:
            pieces.append("Exclude: " + "; ".join(exclusion))
        distinction = norm(topic.get("distinction_rule"))
        if distinction:
            pieces.append(f"Distinction: {distinction}")
        description = " ".join(piece for piece in pieces if not piece.endswith(": "))
        lines.append(f"[1] {topic['topic_id']} (Count: 1): {description}")
    return "\n".join(lines) + "\n"


def collect_records(prediction_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Return one record per kind, document ID, and normalized description."""
    records: dict[tuple[str, str, str], dict[str, Any]] = {}

    for path in prediction_paths:
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError(f"predictions file not found: {path}") from exc

        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"record in {path} line {line_number} is not an object")
                if "doc_id" not in row:
                    raise ValueError(f"record in {path} line {line_number} has no doc_id")
                doc_id = str(row["doc_id"])
                if not doc_id:
                    raise ValueError(f"record in {path} line {line_number} has an empty doc_id")
                missing = [name for name in ("model", "condition", "seed") if name not in row]
                if missing:
                    raise ValueError(
                        f"record in {path} line {line_number} lacks occurrence field(s): "
                        + ", ".join(missing)
                    )

                predicted = row.get("predicted") or {}
                if not isinstance(predicted, dict):
                    raise ValueError(f"predicted in {path} line {line_number} is not an object")
                for field, kind in PREDICTION_FIELDS:
                    items = predicted.get(field) or []
                    if not isinstance(items, list):
                        raise ValueError(f"predicted.{field} in {path} line {line_number} is not a list")
                    for rank, item in enumerate(items, start=1):
                        if not isinstance(item, dict):
                            raise ValueError(
                                f"predicted.{field}[{rank - 1}] in {path} line {line_number} "
                                "is not an object"
                            )
                        text = norm(item.get("description"))
                        if not text:
                            continue
                        key = (kind, doc_id, text)
                        record = records.get(key)
                        if record is None:
                            record_id = "gen_" + sha256_text("|".join(key))[:20]
                            record = records[key] = {
                                "record_id": record_id,
                                "kind": kind,
                                "doc_id": doc_id,
                                "agency": row.get("agency", ""),
                                "text": text,
                                "occurrences": [],
                            }
                        record["occurrences"].append(
                            {
                                "model": row["model"],
                                "condition": row["condition"],
                                "seed": row["seed"],
                                "rank": rank,
                            }
                        )

    output: dict[str, list[dict[str, Any]]] = {"risk": [], "mitigation": []}
    for key in sorted(records):
        output[key[0]].append(records[key])
    return output


def make_batches(
    rows: list[dict[str, Any]],
    max_records: int = BATCH_MAX_RECORDS,
    max_chars: int = BATCH_MAX_CHARS,
) -> list[list[dict[str, Any]]]:
    """Chunk work for caching/concurrency; TopicGPT still prompts per document."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for row in rows:
        size = len(row["text"])
        if current and (len(current) >= max_records or current_chars + size > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def parse_topicgpt_response(
    response: str,
    source_text: str,
    valid_topic_ids: set[str],
    prompted_source_text: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse and strictly validate the PIA-specific TopicGPT response."""
    errors: list[str] = []
    parsed_topics: list[dict[str, str]] = []
    nonempty_lines = [line.strip() for line in response.splitlines() if line.strip()]
    content_lines = [line for line in nonempty_lines if not line.startswith("```")]

    for line_number, line in enumerate(content_lines, start=1):
        match = RESPONSE_LINE_PATTERN.fullmatch(line)
        if match is None:
            errors.append(f"line {line_number} does not match the required assignment format")
            continue
        topic_id = match.group("topic_id")
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number} has invalid JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number} payload is not an object")
            continue
        expected_fields = {"confidence", "rationale", "evidence_quote"}
        if set(payload) != expected_fields:
            errors.append(f"line {line_number} must contain exactly {sorted(expected_fields)}")
            continue
        confidence = payload.get("confidence")
        rationale = payload.get("rationale")
        quote = payload.get("evidence_quote")
        if topic_id not in valid_topic_ids:
            errors.append(f"line {line_number} uses unknown topic_id {topic_id!r}")
        confidence_valid = isinstance(confidence, str) and confidence in CONFIDENCE_VALUES
        if not confidence_valid:
            errors.append(f"line {line_number} has invalid confidence {confidence!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"line {line_number} has an empty rationale")
        if not isinstance(quote, str) or not quote:
            errors.append(f"line {line_number} has an empty evidence quote")
        elif quote not in source_text:
            errors.append(f"line {line_number} evidence quote is not an exact substring")
        elif prompted_source_text is not None and quote not in prompted_source_text:
            errors.append(
                f"line {line_number} evidence quote was not present in the text shown to TopicGPT"
            )
        if (
            topic_id in valid_topic_ids
            and confidence_valid
            and isinstance(rationale, str)
            and rationale.strip()
            and isinstance(quote, str)
            and bool(quote)
            and quote in source_text
            and (prompted_source_text is None or quote in prompted_source_text)
        ):
            parsed_topics.append(
                {
                    "topic_id": topic_id,
                    "confidence": confidence,
                    "rationale": rationale.strip(),
                    "evidence_quote": quote,
                }
            )

    if not content_lines:
        errors.append("response is empty")
    if not 1 <= len(content_lines) <= 3:
        errors.append("response must contain one primary and at most two secondary topics")
    topic_ids = [topic["topic_id"] for topic in parsed_topics]
    if len(topic_ids) != len(set(topic_ids)):
        errors.append("response repeats a topic_id")
    if errors:
        return None, list(dict.fromkeys(errors))

    primary = parsed_topics[0]
    return (
        {
            "primary_topic_id": primary["topic_id"],
            "topics": [
                {
                    "topic_id": topic["topic_id"],
                    "evidence_quote": topic["evidence_quote"],
                    "evidence_exact": True,
                }
                for topic in parsed_topics
            ],
            "confidence": primary["confidence"],
            "rationale": "; ".join(
                f"{topic['topic_id']}: {topic['rationale']}" for topic in parsed_topics
            ),
        },
        [],
    )


def topicgpt_package() -> tuple[TopicGPTAssigner, str]:
    """Load the official package lazily so --prepare-only needs no install."""
    try:
        module = importlib.import_module("topicgpt_python.assignment")
        assigner = getattr(module, "assign_topics")
    except Exception as exc:
        raise RuntimeError(
            "unable to import topicgpt_python. Install the official TopicGPT package "
            "in a separate environment (`python -m pip install topicgpt_python`) and "
            "run this script with that environment's Python. A separate environment "
            "avoids this repository's sentence-transformers version conflict."
        ) from exc
    try:
        version = importlib.metadata.version("topicgpt_python")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (source checkout)"
    try:
        package_dir = Path(module.__file__).resolve().parent
        source_hasher = hashlib.sha256()
        for filename in ("assignment.py", "utils.py"):
            source_hasher.update(filename.encode("utf-8"))
            source_hasher.update((package_dir / filename).read_bytes())
        implementation = f"{version}+source.{source_hasher.hexdigest()[:16]}"
    except (OSError, TypeError):
        implementation = version
    return assigner, implementation


def cache_key(
    kind: Kind,
    package_version: str,
    api: str,
    model: str,
    provider_identity_sha256: str,
    prompt_sha256: str,
    codebook_sha256: str,
    hierarchy_sha256: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    context_len: int | None,
    input_rows: list[dict[str, Any]],
) -> str:
    raw = json.dumps(
        {
            "cache_format": CACHE_FORMAT_VERSION,
            "framework": "topicgpt_python.assign_topics",
            "kind": kind,
            "package_version": package_version,
            "api": api,
            "model": model,
            "provider_identity_sha256": provider_identity_sha256,
            "prompt_sha256": prompt_sha256,
            "codebook_sha256": codebook_sha256,
            "hierarchy_sha256": hierarchy_sha256,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "context_len": context_len,
            "input": [{"id": row["record_id"], "text": row["text"]} for row in input_rows],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(raw)[:24]


def validate_transport(
    raw_rows: list[dict[str, Any]], input_rows: list[dict[str, Any]], source: Path
) -> dict[str, dict[str, Any]]:
    expected = {row["record_id"]: row["text"] for row in input_rows}
    output: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        record_id = str(row.get("id", ""))
        if record_id not in expected:
            raise ValueError(f"TopicGPT output {source} contains unknown id {record_id!r}")
        if record_id in output:
            raise ValueError(f"TopicGPT output {source} repeats id {record_id!r}")
        if row.get("text") != expected[record_id]:
            raise ValueError(f"TopicGPT output {source} changed input text for {record_id}")
        response = row.get("responses")
        if response is None:
            # Let semantic validation route an empty/refused completion through
            # correction and, if necessary, quarantine instead of killing the
            # whole cache chunk.
            row = {**row, "responses": ""}
        elif not isinstance(response, str):
            raise ValueError(f"TopicGPT output {source} has no string response for {record_id}")
        if not isinstance(row.get("prompted_docs"), str):
            raise ValueError(f"TopicGPT output {source} has no prompted text for {record_id}")
        output[record_id] = row
    missing = sorted(set(expected) - set(output))
    if missing:
        raise ValueError(f"TopicGPT output {source} is missing {len(missing)} input id(s)")
    return output


def has_transient_topicgpt_error(rows: dict[str, dict[str, Any]]) -> bool:
    """Detect TopicGPT's sentinel after its provider retries are exhausted."""
    return any(row["responses"].strip().casefold() in {"", "error"} for row in rows.values())


def invoke_topicgpt(
    *,
    assigner: TopicGPTAssigner,
    package_version: str,
    kind: Kind,
    input_rows: list[dict[str, Any]],
    prompt_path: Path,
    prompt_sha256: str,
    topic_path: Path,
    codebook_sha256: str,
    hierarchy_sha256: str,
    cache_dir: Path,
    api: str,
    model: str,
    provider_identity_sha256: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    context_len: int | None,
    verbose: bool,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Run official assign_topics or load its complete raw output cache."""
    key = cache_key(
        kind=kind,
        package_version=package_version,
        api=api,
        model=model,
        provider_identity_sha256=provider_identity_sha256,
        prompt_sha256=prompt_sha256,
        codebook_sha256=codebook_sha256,
        hierarchy_sha256=hierarchy_sha256,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        context_len=context_len,
        input_rows=input_rows,
    )
    cache_path = cache_dir / f"assign_generated_{kind}_topicgpt_{key}.jsonl"
    if cache_path.exists():
        try:
            cached = validate_transport(read_jsonl(cache_path), input_rows, cache_path)
            if not has_transient_topicgpt_error(cached):
                return cached, True
        except ValueError:
            # A truncated/corrupt cache is never accepted. The fresh result is
            # atomically installed below after full transport validation.
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_dir,
        prefix=".topicgpt_input.",
        suffix=".jsonl",
        delete=False,
    ) as input_handle:
        input_path = Path(input_handle.name)
        for row in input_rows:
            input_handle.write(
                json.dumps({"id": row["record_id"], "text": row["text"]}, ensure_ascii=False)
                + "\n"
            )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_dir,
        prefix=".topicgpt_output.",
        suffix=".jsonl",
        delete=False,
    ) as output_handle:
        output_path = Path(output_handle.name)

    kwargs: dict[str, Any] = {
        "api": api,
        "model": model,
        "data": str(input_path),
        "prompt_file": str(prompt_path),
        "out_file": str(output_path),
        "topic_file": str(topic_path),
        "verbose": verbose,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        parameters = inspect.signature(assigner).parameters
        if context_len is not None:
            if "context_len" not in parameters:
                raise RuntimeError(
                    "installed TopicGPT does not support context_len; "
                    "upgrade it or omit --context-len"
                )
            kwargs["context_len"] = context_len
        assigner(**kwargs)
        raw_rows = read_jsonl(output_path)
        validated = validate_transport(raw_rows, input_rows, output_path)
        # Official TopicGPT writes the literal string "Error" after exhausting
        # provider retries. It can be corrected during this run, but must not
        # become a permanent cached failure on future runs.
        if not has_transient_topicgpt_error(validated):
            output_path.replace(cache_path)
        return validated, False
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def correction_packet(source_text: str, response: str, errors: list[str]) -> str:
    return (
        "--- ORIGINAL STATEMENT ---\n"
        + source_text
        + "\n--- END ORIGINAL STATEMENT ---\n"
        + "--- PREVIOUS RESPONSE ---\n"
        + response
        + "\n--- VALIDATION ERRORS ---\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n--- END CORRECTION PACKET ---"
    )


def prompted_original_text(prompted_text: str, correction_round: bool) -> str:
    """Return only source text that TopicGPT actually saw."""
    if not correction_round:
        return prompted_text
    start_marker = "--- ORIGINAL STATEMENT ---\n"
    end_marker = "\n--- END ORIGINAL STATEMENT ---"
    if start_marker not in prompted_text or end_marker not in prompted_text:
        return ""
    return prompted_text.split(start_marker, 1)[1].split(end_marker, 1)[0]


def call_batch(
    *,
    batch_index: int,
    batch: list[dict[str, Any]],
    kind: Kind,
    valid_topic_ids: set[str],
    assignment_prompt_path: Path,
    assignment_prompt_sha256: str,
    correction_prompt_path: Path,
    correction_prompt_sha256: str,
    topic_path: Path,
    codebook_sha256: str,
    hierarchy_sha256: str,
    cache_dir: Path,
    assigner: TopicGPTAssigner,
    package_version: str,
    api: str,
    model: str,
    provider_identity_sha256: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    context_len: int | None,
    max_corrections: int,
    verbose: bool,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Assign one cache chunk and correct every deterministically invalid row."""
    source_by_id = {row["record_id"]: row for row in batch}
    pending = list(batch)
    latest_response: dict[str, str] = {}
    latest_errors: dict[str, list[str]] = {}
    accepted: list[dict[str, Any]] = []
    cache_hits = 0
    calls = 0

    for round_index in range(max_corrections + 1):
        if round_index == 0:
            call_rows = pending
            prompt_path = assignment_prompt_path
            prompt_sha256 = assignment_prompt_sha256
        else:
            call_rows = [
                {
                    **source_by_id[row["record_id"]],
                    "text": correction_packet(
                        source_by_id[row["record_id"]]["text"],
                        latest_response[row["record_id"]],
                        latest_errors[row["record_id"]],
                    ),
                }
                for row in pending
            ]
            prompt_path = correction_prompt_path
            prompt_sha256 = correction_prompt_sha256

        raw_by_id, cache_hit = invoke_topicgpt(
            assigner=assigner,
            package_version=package_version,
            kind=kind,
            input_rows=call_rows,
            prompt_path=prompt_path,
            prompt_sha256=prompt_sha256,
            topic_path=topic_path,
            codebook_sha256=codebook_sha256,
            hierarchy_sha256=hierarchy_sha256,
            cache_dir=cache_dir,
            api=api,
            model=model,
            provider_identity_sha256=provider_identity_sha256,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            context_len=context_len,
            verbose=verbose,
        )
        calls += 1
        cache_hits += int(cache_hit)

        call_text_by_id = {row["record_id"]: row["text"] for row in call_rows}
        rejected: list[dict[str, Any]] = []
        for row in pending:
            record_id = row["record_id"]
            raw_row = raw_by_id[record_id]
            response = raw_row["responses"]
            prompted = raw_row["prompted_docs"]
            parsed, errors = parse_topicgpt_response(
                response,
                source_by_id[record_id]["text"],
                valid_topic_ids,
                prompted_original_text(prompted, correction_round=round_index > 0),
            )
            latest_response[record_id] = response
            latest_errors[record_id] = errors
            if parsed is None:
                rejected.append(source_by_id[record_id])
                continue
            accepted.append(
                {
                    "record_id": record_id,
                    **parsed,
                    "topicgpt_response": response,
                    "topicgpt_attempts": round_index + 1,
                    "topicgpt_input_truncated": (
                        isinstance(prompted, str) and prompted != call_text_by_id[record_id]
                    ),
                }
            )
        pending = rejected
        if not pending:
            break

    quarantined = [
        {
            "record_id": row["record_id"],
            "kind": kind,
            "doc_id": row["doc_id"],
            "agency": row["agency"],
            "text": row["text"],
            "occurrences": row["occurrences"],
            "attempts": max_corrections + 1,
            "last_response": latest_response.get(row["record_id"], ""),
            "validation_errors": latest_errors.get(row["record_id"], ["no response"]),
        }
        for row in pending
    ]
    return batch_index, accepted, quarantined, {"calls": calls, "cache_hits": cache_hits}


def prepare_resources(
    *,
    kind: Kind,
    rows: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    prompt: str,
    prompt_sha256: str,
    prompt_source: str,
    codebook_sha256: str,
    target_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    hierarchy = topicgpt_hierarchy(topics)
    input_path = target_dir / f"topicgpt_{kind}_input{suffix}.jsonl"
    topic_path = target_dir / f"topicgpt_{kind}_topics{suffix}.md"
    prompt_path = target_dir / f"topicgpt_{kind}_assignment_prompt{suffix}.txt"
    correction_prompt = builtin_correction_prompt(kind)
    validate_topicgpt_prompt(
        correction_prompt, f"builtin:{BUILTIN_CORRECTION_PROMPT_VERSION}:{kind}"
    )
    correction_path = target_dir / f"topicgpt_{kind}_correction_prompt{suffix}.txt"
    write_jsonl_atomic(
        input_path, [{"id": row["record_id"], "text": row["text"]} for row in rows]
    )
    write_text_if_changed(topic_path, hierarchy)
    write_text_if_changed(prompt_path, prompt)
    write_text_if_changed(correction_path, correction_prompt)
    return {
        "kind": kind,
        "n_records": len(rows),
        "input_file": input_path.name,
        "topic_file": topic_path.name,
        "assignment_prompt_file": prompt_path.name,
        "assignment_prompt_source": prompt_source,
        "assignment_prompt_sha256": prompt_sha256,
        "correction_prompt_file": correction_path.name,
        "correction_prompt_version": BUILTIN_CORRECTION_PROMPT_VERSION,
        "correction_prompt_sha256": sha256_text(correction_prompt),
        "codebook_sha256": codebook_sha256,
        "topic_hierarchy_sha256": sha256_text(hierarchy),
    }


def run_kind(
    *,
    kind: Kind,
    rows: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    codebook_sha256: str,
    prompt: str,
    prompt_sha256: str,
    prompt_source: str,
    cache_dir: Path,
    output_dir: Path,
    assigner: TopicGPTAssigner,
    package_version: str,
    api: str,
    model: str,
    provider_identity_sha256: str,
    workers: int,
    batch_size: int,
    limit: int | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    context_len: int | None,
    max_corrections: int,
    verbose: bool,
    prediction_file_count: int,
) -> int:
    hierarchy = topicgpt_hierarchy(topics)
    hierarchy_sha256 = sha256_text(hierarchy)
    correction_prompt = builtin_correction_prompt(kind)
    validate_topicgpt_prompt(
        correction_prompt, f"builtin:{BUILTIN_CORRECTION_PROMPT_VERSION}:{kind}"
    )
    correction_prompt_sha256 = sha256_text(correction_prompt)

    topic_path = cache_dir / f"topicgpt_{kind}_topics_{codebook_sha256[:16]}.md"
    prompt_path = cache_dir / f"topicgpt_{kind}_assignment_{prompt_sha256[:16]}.txt"
    correction_path = (
        cache_dir / f"topicgpt_{kind}_correction_{correction_prompt_sha256[:16]}.txt"
    )
    write_text_if_changed(topic_path, hierarchy)
    write_text_if_changed(prompt_path, prompt)
    write_text_if_changed(correction_path, correction_prompt)

    # Official TopicGPT initializes the local vLLM model inside assign_topics;
    # keep one chunk per kind so the model is not reloaded for every cache chunk.
    all_batches = (
        ([rows] if rows else [])
        if api == "vllm"
        else make_batches(rows, max_records=batch_size)
    )
    batches = all_batches if limit is None else all_batches[:limit]
    selected_rows = [row for batch in batches for row in batch]
    partial = len(selected_rows) < len(rows)
    suffix = "_partial" if partial else ""
    print(
        f"[{kind}] {len(selected_rows)}/{len(rows)} records -> {len(batches)} TopicGPT "
        f"cache chunks; approximately {len(selected_rows)} initial model requests",
        flush=True,
    )

    results: list[
        tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]
    ] = []
    if batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    call_batch,
                    batch_index=index,
                    batch=batch,
                    kind=kind,
                    valid_topic_ids={topic["topic_id"] for topic in topics},
                    assignment_prompt_path=prompt_path,
                    assignment_prompt_sha256=prompt_sha256,
                    correction_prompt_path=correction_path,
                    correction_prompt_sha256=correction_prompt_sha256,
                    topic_path=topic_path,
                    codebook_sha256=codebook_sha256,
                    hierarchy_sha256=hierarchy_sha256,
                    cache_dir=cache_dir,
                    assigner=assigner,
                    package_version=package_version,
                    api=api,
                    model=model,
                    provider_identity_sha256=provider_identity_sha256,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    context_len=context_len,
                    max_corrections=max_corrections,
                    verbose=verbose,
                )
                for index, batch in enumerate(batches, start=1)
            ]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                results.append(future.result())
                if completed % 25 == 0 or completed == len(futures):
                    print(f"[{kind}] {completed}/{len(futures)} chunks", flush=True)

    results.sort(key=lambda result: result[0])
    assignments = [item for _, items, _, _ in results for item in items]
    quarantined = [item for _, _, items, _ in results for item in items]
    source_by_id = {row["record_id"]: row for row in selected_rows}
    assignment_by_id = {row["record_id"]: row for row in assignments}
    kept: list[dict[str, Any]] = []
    for source in selected_rows:
        assignment = assignment_by_id.get(source["record_id"])
        if assignment is None:
            continue
        kept.append(
            {
                "record_id": source["record_id"],
                "kind": kind,
                "doc_id": source["doc_id"],
                "agency": source["agency"],
                "text": source["text"],
                "primary_topic_id": assignment["primary_topic_id"],
                "topics": assignment["topics"],
                "confidence": assignment["confidence"],
                "rationale": assignment["rationale"],
                "occurrences": source["occurrences"],
                "topicgpt_response": assignment["topicgpt_response"],
                "topicgpt_attempts": assignment["topicgpt_attempts"],
                "topicgpt_input_truncated": assignment["topicgpt_input_truncated"],
            }
        )

    # All emitted rows have passed strict validation; failed rows are kept in a
    # separate quarantine and force a non-zero process exit in main().
    output_path = output_dir / f"generated_{kind}_topics{suffix}.jsonl"
    quarantine_path = output_dir / f"generated_{kind}_topics{suffix}_quarantine.jsonl"
    write_jsonl_atomic(output_path, kept)
    write_jsonl_atomic(quarantine_path, quarantined)
    corrected = sum(row["topicgpt_attempts"] > 1 for row in kept)
    truncated = sum(bool(row["topicgpt_input_truncated"]) for row in kept)
    calls = sum(batch_meta["calls"] for _, _, _, batch_meta in results)
    cache_hits = sum(batch_meta["cache_hits"] for _, _, _, batch_meta in results)
    missing_ids = sorted(set(source_by_id) - set(assignment_by_id))
    meta = {
        "kind": kind,
        "output_format": OUTPUT_FORMAT_VERSION,
        "cache_format": CACHE_FORMAT_VERSION,
        "partial_run": partial,
        "assignment_scope": "model-generated statements; TopicGPT assignment stage only",
        "topicgpt": {
            "implementation": "official topicgpt_python.assign_topics",
            "package_version": package_version,
            "api": api,
            "provider_identity_sha256": provider_identity_sha256,
            "requested_model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "context_len": context_len,
            "max_corrections": max_corrections,
            "model_requests": "one per statement per attempt (not exposed by package)",
            "token_usage": None,
        },
        "prompt_version": prompt_source,
        "prompt_sha256": prompt_sha256,
        "correction_prompt_version": BUILTIN_CORRECTION_PROMPT_VERSION,
        "correction_prompt_sha256": correction_prompt_sha256,
        "codebook_sha256": codebook_sha256,
        "topic_hierarchy_sha256": hierarchy_sha256,
        "prediction_file_count": prediction_file_count,
        # Compatibility aliases retained for consumers of the v1 metadata.
        "requested_model": model,
        "returned_models": [],
        "n_records": len(rows),
        "n_batches": len(batches),
        "n_missing": len(quarantined),
        "usage": {
            "available": False,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
        },
        "n_records_total": len(rows),
        "n_records_selected": len(selected_rows),
        "n_batches_total": len(all_batches),
        "n_batches_processed": len(batches),
        "batch_limit": limit,
        "n_assigned": len(kept),
        "n_corrected": corrected,
        "n_quarantined": len(quarantined),
        "n_topicgpt_inputs_truncated": truncated,
        "missing_sample": missing_ids[:20],
        "topicgpt_chunk_calls": calls,
        "topicgpt_chunk_cache_hits": cache_hits,
        "integrity": {
            "strict_topic_ids": True,
            "one_primary_max_two_secondary": True,
            "unique_topic_ids": True,
            "exact_nonempty_evidence": True,
            "invalid_rows_emitted": 0,
        },
    }
    meta_path = output_dir / f"generated_{kind}_topics{suffix}_meta.json"
    write_json_atomic(meta_path, meta)
    print(
        f"[{kind}] wrote {output_path}: assigned={len(kept)} corrected={corrected} "
        f"quarantined={len(quarantined)}",
        flush=True,
    )
    return len(quarantined)


def configure_provider(api: str, api_key_env: str) -> None:
    if api == "openai":
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"environment variable {api_key_env!r} is not set or is empty")
        # TopicGPT's OpenAI client specifically reads OPENAI_API_KEY.
        os.environ["OPENAI_API_KEY"] = api_key
    elif api == "azure":
        missing = [
            name
            for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError("missing Azure environment variable(s): " + ", ".join(missing))
    elif api == "vertex":
        missing = [
            name for name in ("VERTEX_PROJECT", "VERTEX_LOCATION") if not os.environ.get(name)
        ]
        if missing:
            raise ValueError("missing Vertex environment variable(s): " + ", ".join(missing))
    elif api == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("environment variable 'GEMINI_API_KEY' is not set or is empty")


def provider_identity_sha256(api: str) -> str:
    """Hash non-secret endpoint/deployment identity for cache isolation."""
    identity: dict[str, str] = {"api": api}
    if api == "openai":
        identity["base_url"] = os.environ.get("OPENAI_BASE_URL", "official-openai-default")
    elif api == "azure":
        identity["endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    elif api == "vertex":
        identity["project"] = os.environ.get("VERTEX_PROJECT", "")
        identity["location"] = os.environ.get("VERTEX_LOCATION", "")
    elif api == "gemini":
        identity["endpoint"] = "official-gemini-default"
    elif api == "vllm":
        identity["backend"] = "local-vllm"
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="one or more generation prediction JSONL files",
    )
    parser.add_argument("--codebooks-dir", type=Path, required=True)
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help=(
            "optional directory containing topic_assignment_risk.txt and/or "
            "topic_assignment_mitigation.txt; otherwise use the embedded PIA prompts"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--api",
        choices=("openai", "azure", "vertex", "gemini", "vllm"),
        default="openai",
        help="provider backend passed to official TopicGPT",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="assignment model; it must be supported by TopicGPT's selected provider",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="OpenAI API-key environment variable; the value is never logged",
    )
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=BATCH_MAX_RECORDS,
        help="records per resumable cache chunk (TopicGPT still sends one prompt per record)",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=600)
    parser.add_argument("--temperature", type=temperature_value, default=0.0)
    parser.add_argument("--top-p", type=unit_interval, default=1.0)
    parser.add_argument(
        "--context-len",
        type=positive_int,
        default=None,
        help="TopicGPT context budget; set explicitly for newer or custom models",
    )
    parser.add_argument(
        "--max-corrections",
        type=nonnegative_int,
        default=1,
        help="bounded TopicGPT reassignments for rows that fail deterministic validation",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=tuple(PROMPT_FILES),
        default=["risk", "mitigation"],
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="maximum cache chunks per kind; writes *_partial outputs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write inspectable TopicGPT input/topic/prompt files without importing TopicGPT",
    )
    parser.add_argument("--verbose", action="store_true", help="enable TopicGPT logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        parser.error("--api-key-env must be a valid environment-variable name")
    if args.api == "vllm" and args.workers != 1:
        parser.error("use --workers 1 with vLLM; TopicGPT batches vLLM inputs internally")

    cache_dir: Path = args.cache_dir
    output_dir: Path = args.output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_kinds: list[Kind] = list(dict.fromkeys(args.kinds))

    try:
        records = collect_records(args.predictions)
        resources: dict[Kind, tuple[list[dict[str, Any]], str, str, str, str]] = {}
        for kind in selected_kinds:
            topics, codebook_sha256 = load_codebook(args.codebooks_dir, kind)
            if args.prompts_dir is None:
                topic_ids = {topic["topic_id"] for topic in topics}
                missing_examples = BUILTIN_EXAMPLE_TOPIC_IDS[kind] - topic_ids
                if missing_examples:
                    raise ValueError(
                        f"the built-in {kind} prompt refers to topic IDs absent from the "
                        f"codebook ({', '.join(sorted(missing_examples))}); supply a "
                        "matching prompt with --prompts-dir"
                    )
            prompt, prompt_sha256, prompt_source = load_prompt(args.prompts_dir, kind)
            resources[kind] = (
                topics,
                codebook_sha256,
                prompt,
                prompt_sha256,
                prompt_source,
            )
    except ValueError as exc:
        parser.error(str(exc))

    all_batches = {
        kind: (
            ([records[kind]] if records[kind] else [])
            if args.api == "vllm"
            else make_batches(records[kind], max_records=args.batch_size)
        )
        for kind in selected_kinds
    }
    selected_rows = {
        kind: [
            row
            for batch in (
                all_batches[kind] if args.limit is None else all_batches[kind][: args.limit]
            )
            for row in batch
        ]
        for kind in selected_kinds
    }
    partial_run = any(
        len(selected_rows[kind]) < len(records[kind]) for kind in selected_kinds
    )
    suffix = "_partial" if partial_run else ""
    index = (
        {kind: len(rows) for kind, rows in records.items()}
        if not partial_run
        else {
            kind: {
                "total": len(records[kind]),
                "selected": len(selected_rows.get(kind, [])),
            }
            for kind in records
        }
    )
    write_json_atomic(output_dir / f"generated_records_index{suffix}.json", index)

    if args.prepare_only:
        manifests = []
        for kind in selected_kinds:
            topics, codebook_sha256, prompt, prompt_sha256, prompt_source = resources[kind]
            manifests.append(
                prepare_resources(
                    kind=kind,
                    rows=selected_rows[kind],
                    topics=topics,
                    prompt=prompt,
                    prompt_sha256=prompt_sha256,
                    prompt_source=prompt_source,
                    codebook_sha256=codebook_sha256,
                    target_dir=output_dir,
                    suffix=(
                        "_partial"
                        if len(selected_rows[kind]) < len(records[kind])
                        else ""
                    ),
                )
            )
        write_json_atomic(
            output_dir / f"topicgpt_preparation_manifest{suffix}.json",
            {
                "framework_stage": "TopicGPT assignment",
                "api_calls_made": False,
                "partial": partial_run,
                "resources": manifests,
            },
        )
        print(f"prepared TopicGPT files in {output_dir}; no API calls made", flush=True)
        return

    try:
        configure_provider(args.api, args.api_key_env)
        backend_sha256 = provider_identity_sha256(args.api)
        assigner, package_version = topicgpt_package()
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    total_quarantined = 0
    for kind in selected_kinds:
        topics, codebook_sha256, prompt, prompt_sha256, prompt_source = resources[kind]
        total_quarantined += run_kind(
            kind=kind,
            rows=records[kind],
            topics=topics,
            codebook_sha256=codebook_sha256,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            prompt_source=prompt_source,
            cache_dir=cache_dir,
            output_dir=output_dir,
            assigner=assigner,
            package_version=package_version,
            api=args.api,
            model=args.model,
            provider_identity_sha256=backend_sha256,
            workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            context_len=args.context_len,
            max_corrections=args.max_corrections,
            verbose=args.verbose,
            prediction_file_count=len(args.predictions),
        )
    if total_quarantined:
        print(
            f"TopicGPT assignment finished with {total_quarantined} quarantined row(s); "
            "see *_quarantine.jsonl",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
