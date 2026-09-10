from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


CACHE_FORMAT = "structural-screening-v1"
MODEL_DEFAULT = "gpt-5.4"
PROMPT_DEFAULT = "structural_screening.txt"
DIGEST_MAX_CHARS = 14_000
SNIPPET_CHARS = 220

DOC_ID_RE = re.compile(r"^(\d{3})_([a-z0-9]+)_[a-z0-9][a-z0-9._-]*$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DESC_HEADING_RE = re.compile(
    r"(overview|system description|describe the (system|information)|"
    r"information (the|collected|is collected)|what information|sources? of information|"
    r"purpose of the (information|system)|how is the information used|uses? of the information|"
    r"information sharing|data sharing|retention|records retention)",
    re.IGNORECASE,
)
RISK_RE = re.compile(
    r"(privacy risk|there is a risk|risk that|privacy impact analysis|"
    r"risk analysis|privacy analysis and risk)",
    re.IGNORECASE,
)
MITIGATION_RE = re.compile(
    r"(mitigation\b|mitigation:|to mitigate|this risk is mitigated|risk is mitigated|"
    r"mitigated by|to address this risk)",
    re.IGNORECASE,
)

RESPONSE_KEYS = {
    "class",
    "rationale",
    "description_sections",
    "risk_sections",
    "mitigation_sections",
    "removable_as_chunks",
    "confidence",
}
CLASS_ORDER = ("clean", "mixed", "messy")
CSV_FIELDS = (
    "doc_id",
    "agency",
    "model",
    "class",
    "confidence",
    "rationale",
    "description_sections",
    "risk_sections",
    "mitigation_sections",
    "removable_as_chunks",
    "seed",
    "cache_file",
    "prompt_sha256",
    "run_config_sha256",
)


class ScreeningError(ValueError):
    """An input, response, or cache failed validation."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScreeningError(f"required file is missing: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ScreeningError(f"invalid JSON in {path.name}: {exc}") from exc


def agency_of(doc_id: str) -> str:
    match = DOC_ID_RE.fullmatch(doc_id)
    if match is None:
        raise ScreeningError(f"invalid document directory name: {doc_id!r}")
    return match.group(2)


def parsed_documents(parsed_dir: Path) -> list[Path]:
    if not parsed_dir.is_dir():
        raise ScreeningError("--parsed-dir is not a directory")
    if parsed_dir.is_symlink():
        raise ScreeningError("--parsed-dir may not be a symbolic link")

    documents: list[Path] = []
    for candidate in sorted(parsed_dir.iterdir(), key=lambda item: item.name):
        if candidate.is_symlink():
            raise ScreeningError(f"symbolic links are not accepted: {candidate.name}")
        if not candidate.is_dir() or not (candidate / "sections.json").exists():
            continue
        agency_of(candidate.name)
        documents.append(candidate)
    if not documents:
        raise ScreeningError("--parsed-dir contains no document directories with sections.json")
    return documents


def load_sections(document: Path) -> list[dict[str, Any]]:
    path = document / "sections.json"
    if path.is_symlink():
        raise ScreeningError(f"symbolic links are not accepted for {document.name}/sections.json")
    sections = read_json(path)
    if not isinstance(sections, list):
        raise ScreeningError(f"{document.name}/sections.json must contain an array")
    validated: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ScreeningError(f"section {index} in {document.name} is not an object")
        row = dict(section)
        for field in ("name", "content"):
            value = row.get(field, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise ScreeningError(
                    f"section {index} field {field!r} in {document.name} must be text"
                )
            row[field] = value
        validated.append(row)
    return validated


def load_optional_spans(document: Path) -> list[Any]:
    path = document / "risk_mitigation_spans.json"
    if not path.exists():
        return []
    if path.is_symlink():
        raise ScreeningError(
            f"symbolic links are not accepted for {document.name}/risk_mitigation_spans.json"
        )
    spans = read_json(path)
    if not isinstance(spans, list):
        raise ScreeningError(
            f"{document.name}/risk_mitigation_spans.json must contain an array"
        )
    return spans


def load_optional_full_text(document: Path) -> str:
    path = document / "full_text.txt"
    if not path.exists():
        return ""
    if path.is_symlink():
        raise ScreeningError(f"symbolic links are not accepted for {document.name}/full_text.txt")
    return path.read_text(encoding="utf-8", errors="replace")


def section_flags(name: str, content: str) -> dict[str, bool]:
    return {
        "desc_head": bool(DESC_HEADING_RE.search(name)),
        "risk_head": bool(RISK_RE.search(name)),
        "mit_head": bool(MITIGATION_RE.search(name)),
        "risk_body": bool(RISK_RE.search(content)),
        "mit_body": bool(MITIGATION_RE.search(content)),
    }


def compute_signals(document: Path, sections: list[dict[str, Any]]) -> dict[str, Any]:
    n_desc = 0
    n_risk_head = 0
    n_mit_head = 0
    risk_body_hits = 0
    mitigation_body_hits = 0
    entangled = 0
    parse_risk_typed = 0

    for section in sections:
        flags = section_flags(section["name"], section["content"])
        if section.get("type") == "risk":
            parse_risk_typed += 1
        if flags["desc_head"] and not (flags["risk_head"] or flags["mit_head"]):
            n_desc += 1
            if flags["risk_body"] or flags["mit_body"]:
                entangled += 1
        n_risk_head += int(flags["risk_head"])
        n_mit_head += int(flags["mit_head"])
        risk_body_hits += int(flags["risk_body"])
        mitigation_body_hits += int(flags["mit_body"])

    return {
        "doc_id": document.name,
        "agency": agency_of(document.name),
        "n_sections": len(sections),
        "n_chars": len(load_optional_full_text(document)),
        "n_desc_sections": n_desc,
        "n_risk_head_sections": n_risk_head,
        "n_mit_head_sections": n_mit_head,
        "risk_body_hit_sections": risk_body_hits,
        "mit_body_hit_sections": mitigation_body_hits,
        "n_entangled_desc_sections": entangled,
        "parse_risk_typed_sections": parse_risk_typed,
        "parse_paired_spans": len(load_optional_spans(document)),
    }


def build_digest(sections: list[dict[str, Any]]) -> str:
    """Build the compact ordered section digest used by the historical run."""

    def render(index: int, section: dict[str, Any], require_signal: bool = False) -> str | None:
        name = re.sub(r"\s+", " ", section["name"]).strip()[:120]
        content = re.sub(r"\s+", " ", section["content"]).strip()
        flags = section_flags(section["name"], section["content"])
        signal = any(flags.values())
        if require_signal and not signal:
            return None
        tags = "".join(
            tag
            for tag, enabled in (
                ("[DESC]", flags["desc_head"]),
                ("[RISK]", flags["risk_head"] or flags["risk_body"]),
                ("[MIT]", flags["mit_head"] or flags["mit_body"]),
            )
            if enabled
        ) or "[--]"
        return f"#{index} {tags} {name}\n    {content[:SNIPPET_CHARS]}"

    digest = "\n".join(
        line for index, section in enumerate(sections) if (line := render(index, section))
    )
    if len(digest) <= DIGEST_MAX_CHARS:
        return digest

    retained = [
        line
        for index, section in enumerate(sections)
        if (line := render(index, section, require_signal=True)) is not None
    ]
    return ("[trimmed to structure-bearing sections]\n" + "\n".join(retained))[
        :DIGEST_MAX_CHARS
    ]


def validate_index_list(value: Any, field: str, section_count: int) -> list[int]:
    if not isinstance(value, list):
        raise ScreeningError(f"response field {field!r} must be an array")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ScreeningError(f"response field {field!r} must contain only integers")
        if not 0 <= item < section_count:
            raise ScreeningError(
                f"response field {field!r} contains out-of-range section index {item}"
            )
        output.append(item)
    if len(output) != len(set(output)):
        raise ScreeningError(f"response field {field!r} contains duplicate section indices")
    return output


def validate_response(value: Any, section_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScreeningError("response JSON must be an object")
    missing = RESPONSE_KEYS - set(value)
    extra = set(value) - RESPONSE_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ScreeningError("response schema mismatch: " + "; ".join(details))

    label = value["class"]
    if label not in CLASS_ORDER:
        raise ScreeningError("response class must be clean, mixed, or messy")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ScreeningError("response rationale must be non-empty text")
    removable = value["removable_as_chunks"]
    if not isinstance(removable, bool):
        raise ScreeningError("response removable_as_chunks must be a boolean")
    if removable != (label == "clean"):
        raise ScreeningError("removable_as_chunks must be true exactly when class is clean")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ScreeningError("response confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ScreeningError("response confidence must be between 0 and 1")

    return {
        "class": label,
        "rationale": rationale,
        "description_sections": validate_index_list(
            value["description_sections"], "description_sections", section_count
        ),
        "risk_sections": validate_index_list(
            value["risk_sections"], "risk_sections", section_count
        ),
        "mitigation_sections": validate_index_list(
            value["mitigation_sections"], "mitigation_sections", section_count
        ),
        "removable_as_chunks": removable,
        "confidence": float(confidence),
    }


def parse_response_text(raw_response_text: str, section_count: int) -> dict[str, Any]:
    if not isinstance(raw_response_text, str) or not raw_response_text.strip():
        raise ScreeningError("API returned empty response text")
    try:
        value = json.loads(raw_response_text)
    except json.JSONDecodeError as exc:
        raise ScreeningError(f"API response is not one JSON object: {exc}") from exc
    return validate_response(value, section_count)


def deterministic_seed(doc_id: str) -> int:
    return int(hashlib.sha256(f"{doc_id}|struct".encode("utf-8")).hexdigest(), 16) % 100_000


def run_config(
    *,
    doc_id: str,
    digest_sha256: str,
    prompt_sha256: str,
    model: str,
    temperature: float,
    base_url: str | None,
) -> tuple[dict[str, Any], str]:
    config = {
        "cache_format": CACHE_FORMAT,
        "doc_id": doc_id,
        "digest_sha256": digest_sha256,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "temperature": temperature,
        "seed": deterministic_seed(doc_id),
        "endpoint_sha256": sha256_bytes((base_url or "official-default").encode("utf-8")),
    }
    return config, canonical_sha256(config)


def response_metadata(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    returned_model = getattr(response, "model", None)
    finish_reason = getattr(choice, "finish_reason", None)
    metadata: dict[str, Any] = {
        "model": returned_model if isinstance(returned_model, str) else "",
        "finish_reason": finish_reason if isinstance(finish_reason, str) else "",
    }
    if usage is not None:
        for source, destination in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            count = getattr(usage, source, None)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                metadata[destination] = count
    return metadata


def validate_cache(
    value: Any,
    *,
    document: Path,
    section_count: int,
    expected_config: dict[str, Any],
    expected_config_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScreeningError("cache must contain a JSON object")
    required = {
        "cache_format",
        "doc_id",
        "agency",
        "run_config",
        "run_config_sha256",
        "raw_response_text",
        "parsed",
        "response_metadata",
    }
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ScreeningError("cache schema mismatch: " + "; ".join(details))
    if value["cache_format"] != CACHE_FORMAT:
        raise ScreeningError("cache format does not match this script")
    if value["doc_id"] != document.name or value["agency"] != agency_of(document.name):
        raise ScreeningError("cache document identity does not match its filename")
    if value["run_config"] != expected_config:
        raise ScreeningError("cache configuration does not match the current run")
    if value["run_config_sha256"] != expected_config_sha256:
        raise ScreeningError("cache configuration hash does not match the current run")
    if canonical_sha256(value["run_config"]) != value["run_config_sha256"]:
        raise ScreeningError("cache configuration hash is internally inconsistent")
    parsed = validate_response(value["parsed"], section_count)
    parsed_from_raw = parse_response_text(value["raw_response_text"], section_count)
    if parsed != parsed_from_raw:
        raise ScreeningError("cached parsed response does not match raw response text")
    metadata = value["response_metadata"]
    if not isinstance(metadata, dict):
        raise ScreeningError("cache response_metadata must be an object")
    metadata_keys = {"model", "finish_reason", "prompt_tokens", "completion_tokens", "total_tokens"}
    if not {"model", "finish_reason"}.issubset(metadata):
        raise ScreeningError("cache response_metadata lacks model or finish_reason")
    if set(metadata) - metadata_keys:
        raise ScreeningError("cache response_metadata contains unexpected fields")
    if not isinstance(metadata["model"], str) or not isinstance(metadata["finish_reason"], str):
        raise ScreeningError("cache response metadata model and finish_reason must be text")
    if not metadata["model"].strip() or not metadata["finish_reason"].strip():
        raise ScreeningError("cache response metadata model and finish_reason may not be empty")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if field in metadata and (
            isinstance(metadata[field], bool)
            or not isinstance(metadata[field], int)
            or metadata[field] < 0
        ):
            raise ScreeningError(f"cache response_metadata {field} must be non-negative integer")
    return value


def write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stage_signals(documents: list[Path], output_dir: Path) -> None:
    rows = [compute_signals(document, load_sections(document)) for document in documents]
    fields = list(rows[0])
    write_csv(output_dir / "per_doc_signals.csv", fields, rows)
    print(f"[signals] wrote per_doc_signals.csv ({len(rows)} documents)")


def create_client(api_key_env: str, base_url: str | None):
    from openai import OpenAI

    key = os.environ.get(api_key_env)
    if not key:
        raise ScreeningError(f"environment variable {api_key_env} is not set")
    options: dict[str, str] = {"api_key": key}
    if base_url:
        options["base_url"] = base_url
    return OpenAI(**options)


def call_model(
    client: Any,
    *,
    model: str,
    prompt: str,
    doc_id: str,
    digest: str,
    temperature: float,
    seed: int,
    max_retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"DOCUMENT: {doc_id}\n\nSECTION DIGEST:\n{digest}",
                    },
                ],
                temperature=temperature,
                seed=seed,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # The SDK exposes multiple transient error subclasses.
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(min(2**attempt, 30))
    raise ScreeningError(
        f"API call failed after {max_retries} attempt(s): {type(last_error).__name__}"
    ) from last_error


def stage_classify(
    documents: list[Path],
    output_dir: Path,
    *,
    prompts_dir: Path,
    prompt_name: str,
    model: str,
    temperature: float,
    api_key_env: str,
    base_url: str | None,
    max_retries: int,
    overwrite_stale_cache: bool,
    dry_run: bool,
) -> None:
    if prompts_dir.is_symlink():
        raise ScreeningError("--prompts-dir may not be a symbolic link")
    if not prompts_dir.is_dir():
        raise ScreeningError("--prompts-dir is not a directory")
    prompt_path = prompts_dir / prompt_name
    if prompt_path.is_symlink():
        raise ScreeningError("the structural-screening prompt may not be a symbolic link")
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScreeningError(f"prompt file is missing: {prompt_name}") from exc
    if not prompt.strip():
        raise ScreeningError(f"prompt file is empty: {prompt_name}")
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))

    prepared: list[tuple[Path, list[dict[str, Any]], str, dict[str, Any], str]] = []
    for document in documents:
        sections = load_sections(document)
        digest = build_digest(sections)
        config, config_sha256 = run_config(
            doc_id=document.name,
            digest_sha256=sha256_bytes(digest.encode("utf-8")),
            prompt_sha256=prompt_sha256,
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
        prepared.append((document, sections, digest, config, config_sha256))

    if dry_run:
        print(
            f"[classify] dry run validated {len(prepared)} documents and {prompt_name}; "
            "made no API calls and wrote no files"
        )
        return

    client = None
    cache_dir = output_dir / "cache"
    if cache_dir.is_symlink():
        raise ScreeningError("the cache directory may not be a symbolic link")
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ScreeningError("the cache path exists but is not a directory")
    expected_cache_names = {f"{document.name}.json" for document in documents}
    existing_cache_names = (
        {path.name for path in cache_dir.iterdir()} if cache_dir.exists() else set()
    )
    orphaned = sorted(existing_cache_names - expected_cache_names)
    if orphaned:
        preview = ", ".join(orphaned[:3])
        suffix = " ..." if len(orphaned) > 3 else ""
        raise ScreeningError(
            f"cache contains {len(orphaned)} document(s) absent from --parsed-dir: "
            f"{preview}{suffix}"
        )
    rows: list[dict[str, Any]] = []
    cached_count = 0
    called_count = 0

    for document, sections, digest, config, config_sha256 in prepared:
        cache_path = cache_dir / f"{document.name}.json"
        cache: dict[str, Any] | None = None
        if cache_path.exists():
            try:
                cache = validate_cache(
                    read_json(cache_path),
                    document=document,
                    section_count=len(sections),
                    expected_config=config,
                    expected_config_sha256=config_sha256,
                )
            except ScreeningError as exc:
                if not overwrite_stale_cache:
                    raise ScreeningError(
                        f"cache {cache_path.name} is stale or malformed ({exc}); "
                        "rerun with --overwrite-stale-cache to replace it"
                    ) from exc
            if cache is not None:
                cached_count += 1

        if cache is None:
            if client is None:
                client = create_client(api_key_env, base_url)
            response = call_model(
                client,
                model=model,
                prompt=prompt,
                doc_id=document.name,
                digest=digest,
                temperature=temperature,
                seed=config["seed"],
                max_retries=max_retries,
            )
            try:
                raw_response_text = response.choices[0].message.content
                metadata = response_metadata(response)
            except (AttributeError, IndexError, TypeError) as exc:
                raise ScreeningError("API response lacks a complete first choice") from exc
            parsed = parse_response_text(raw_response_text, len(sections))
            cache = {
                "cache_format": CACHE_FORMAT,
                "doc_id": document.name,
                "agency": agency_of(document.name),
                "run_config": config,
                "run_config_sha256": config_sha256,
                "raw_response_text": raw_response_text,
                "parsed": parsed,
                "response_metadata": metadata,
            }
            validate_cache(
                cache,
                document=document,
                section_count=len(sections),
                expected_config=config,
                expected_config_sha256=config_sha256,
            )
            atomic_write_text(
                cache_path,
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            called_count += 1

        parsed = cache["parsed"]
        rows.append(
            {
                "doc_id": document.name,
                "agency": agency_of(document.name),
                "model": model,
                "class": parsed["class"],
                "confidence": parsed["confidence"],
                "rationale": re.sub(r"\s+", " ", parsed["rationale"]).strip(),
                "description_sections": json.dumps(parsed["description_sections"]),
                "risk_sections": json.dumps(parsed["risk_sections"]),
                "mitigation_sections": json.dumps(parsed["mitigation_sections"]),
                "removable_as_chunks": json.dumps(parsed["removable_as_chunks"]),
                "seed": config["seed"],
                "cache_file": f"cache/{cache_path.name}",
                "prompt_sha256": prompt_sha256,
                "run_config_sha256": config_sha256,
            }
        )

    write_csv(output_dir / "per_doc_classification.csv", list(CSV_FIELDS), rows)
    print(
        f"[classify] wrote per_doc_classification.csv ({len(rows)} documents; "
        f"{cached_count} cached, {called_count} API calls)"
    )


def load_classification_csv(path: Path) -> list[dict[str, str]]:
    try:
        handle = path.open("r", newline="", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScreeningError("per_doc_classification.csv does not exist") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ScreeningError("per_doc_classification.csv has an unexpected schema")
        rows = list(reader)
    if not rows:
        raise ScreeningError("per_doc_classification.csv contains no rows")
    doc_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            agency = agency_of(row["doc_id"])
            if row["agency"] != agency:
                raise ScreeningError("agency does not match doc_id")
            if not row["model"].strip():
                raise ScreeningError("model is empty")
            if row["class"] not in CLASS_ORDER:
                raise ScreeningError("unknown class")
            confidence = float(row["confidence"])
            if not 0 <= confidence <= 1:
                raise ScreeningError("confidence outside [0, 1]")
            if not row["rationale"].strip():
                raise ScreeningError("rationale is empty")
            removable = json.loads(row["removable_as_chunks"])
            if not isinstance(removable, bool) or removable != (row["class"] == "clean"):
                raise ScreeningError("removable_as_chunks conflicts with class")
            for field in (
                "description_sections",
                "risk_sections",
                "mitigation_sections",
            ):
                indices = json.loads(row[field])
                if not isinstance(indices, list) or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in indices
                ):
                    raise ScreeningError(f"{field} is not an array of non-negative integers")
                if len(indices) != len(set(indices)):
                    raise ScreeningError(f"{field} contains duplicate indices")
            if int(row["seed"]) != deterministic_seed(row["doc_id"]):
                raise ScreeningError("seed does not match the deterministic document seed")
            if row["cache_file"] != f"cache/{row['doc_id']}.json":
                raise ScreeningError("cache_file does not match doc_id")
            if row["doc_id"] in doc_ids:
                raise ScreeningError("duplicate doc_id")
            for field in ("prompt_sha256", "run_config_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                    raise ScreeningError(f"{field} is not a SHA-256 digest")
            doc_ids.add(row["doc_id"])
        except (KeyError, ValueError, TypeError, AttributeError, ScreeningError) as exc:
            raise ScreeningError(f"invalid classification CSV row {line_number}: {exc}") from exc
    if len({row["model"] for row in rows}) != 1:
        raise ScreeningError("per_doc_classification.csv mixes multiple model configurations")
    if len({row["prompt_sha256"] for row in rows}) != 1:
        raise ScreeningError("per_doc_classification.csv mixes multiple prompt configurations")
    return rows


def stage_summary(output_dir: Path) -> None:
    rows = load_classification_csv(output_dir / "per_doc_classification.csv")
    overall = Counter(row["class"] for row in rows)
    per_agency: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        per_agency[row["agency"]][row["class"]] += 1

    models = sorted({row["model"] for row in rows})
    lines = [
        "# Structural-screening summary",
        "",
        f"Model(s): {', '.join(f'`{model}`' for model in models)}. Documents: {len(rows)}.",
        "",
        "## Class distribution",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {label} | {overall.get(label, 0)} |" for label in CLASS_ORDER)
    lines.extend(
        [
            "",
            "## Per agency",
            "",
            "| Agency | clean | mixed | messy | Total |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for agency in sorted(per_agency):
        counts = per_agency[agency]
        total = sum(counts.values())
        lines.append(
            f"| {agency} | {counts.get('clean', 0)} | {counts.get('mixed', 0)} | "
            f"{counts.get('messy', 0)} | {total} |"
        )
    atomic_write_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    print("[summary] wrote summary.md")


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.stage in {"signals", "classify", "all"} and args.parsed_dir is None:
        parser.error("--parsed-dir is required for signals, classify, and all")
    if args.stage in {"classify", "all"} and args.prompts_dir is None:
        parser.error("--prompts-dir is required for classify and all")
    if args.dry_run and args.stage not in {"classify"}:
        parser.error("--dry-run is supported only with --stage classify")
    if not ENV_NAME_RE.fullmatch(args.api_key_env):
        parser.error("--api-key-env must be a valid environment-variable name")
    if Path(args.prompt_name).name != args.prompt_name or args.prompt_name in {".", ".."}:
        parser.error("--prompt-name must be a filename, not a path")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    if not args.model.strip():
        parser.error("--model may not be empty")
    if any(character in args.model for character in "\r\n\0"):
        parser.error("--model may not contain control characters")
    if args.base_url:
        parsed = urlsplit(args.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            parser.error("--base-url must be an HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            parser.error("--base-url may not contain credentials")

    if args.parsed_dir is not None:
        parsed_root = args.parsed_dir.resolve()
        output_root = args.output_dir.resolve()
        if parsed_root == output_root or parsed_root in output_root.parents:
            parser.error("--output-dir may not be --parsed-dir or one of its parents")
        if output_root in parsed_root.parents:
            parser.error("--output-dir may not contain --parsed-dir")
    if args.output_dir.is_symlink():
        parser.error("--output-dir may not be a symbolic link")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("signals", "classify", "summary", "all"), required=True
    )
    parser.add_argument("--parsed-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path)
    parser.add_argument("--prompt-name", default=PROMPT_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-retries", type=positive_int, default=3)
    parser.add_argument("--overwrite-stale-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_arguments(args, parser)

    try:
        documents = (
            parsed_documents(args.parsed_dir)
            if args.stage in {"signals", "classify", "all"}
            else []
        )
        if args.stage in {"signals", "all"}:
            stage_signals(documents, args.output_dir)
        if args.stage in {"classify", "all"}:
            stage_classify(
                documents,
                args.output_dir,
                prompts_dir=args.prompts_dir,
                prompt_name=args.prompt_name,
                model=args.model,
                temperature=args.temperature,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                max_retries=args.max_retries,
                overwrite_stale_cache=args.overwrite_stale_cache,
                dry_run=args.dry_run,
            )
        if args.stage in {"summary", "all"}:
            stage_summary(args.output_dir)
    except ScreeningError as exc:
        raise SystemExit(f"[structural-screening] {exc}") from exc


if __name__ == "__main__":
    main()
