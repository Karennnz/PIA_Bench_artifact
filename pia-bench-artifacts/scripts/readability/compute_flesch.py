#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import nltk
    import textstat
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Install the exact packages in requirements.txt and download NLTK cmudict."
    ) from exc


EXPECTED_TEXTSTAT = "0.7.13"
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "results/track_b_ranked_deepseek_20260906/all_predictions_five_models.jsonl"
GOLD_FILES = {
    "risk": ROOT / "2026-08-31_infpriv_submission/content_analysis_73/inputs/risks.jsonl",
    "mitigation": ROOT
    / "2026-08-31_infpriv_submission/content_analysis_73/inputs/mitigations.jsonl",
}

MODEL_ORDER = [
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
    "llama-3.1-8b-instruct",
    "deepseek-r1-distill-qwen-14b",
    "gpt-5.4-mini-2026-03-17",
]
CONDITION_ORDER = ["zero_shot", "cot"]
TASK_ORDER = ["risk", "mitigation"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_item(text: str) -> str:
    """Normalize layout only and make each ranked item sentence-delimited."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def score_list(texts: Iterable[str]) -> dict[str, float | int] | None:
    items = [normalize_item(text) for text in texts]
    items = [text for text in items if text]
    if not items:
        return None

    # Items are distinct ranked statements. Terminal punctuation above keeps
    # them distinct when the original generator omitted a final full stop.
    passage = "\n".join(items)
    words = int(textstat.lexicon_count(passage))
    sentences = int(textstat.sentence_count(passage))
    syllables = int(textstat.syllable_count(passage))
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    library_score = float(textstat.flesch_reading_ease(passage))
    if abs(score - library_score) > 1e-9:
        raise RuntimeError("textstat result does not equal the documented English FRE formula")
    return {
        "n_items": len(items),
        "n_words": words,
        "n_sentences": sentences,
        "n_syllables": syllables,
        "words_per_item": words / len(items),
        "flesch_reading_ease": library_score,
    }


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def output_row(
    *,
    source: str,
    model: str,
    condition: str,
    seed: int | str,
    doc_id: str,
    task: str,
    status: str,
    result: dict[str, float | int] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "model": model,
        "condition": condition,
        "seed": seed,
        "doc_id": doc_id,
        "task": task,
        "status": status,
        "n_items": "",
        "n_words": "",
        "n_sentences": "",
        "n_syllables": "",
        "words_per_item": "",
        "flesch_reading_ease": "",
    }
    if result is not None:
        row.update(result)
    return row


def validate_environment() -> tuple[str, str]:
    version = importlib.metadata.version("textstat")
    if version != EXPECTED_TEXTSTAT:
        raise SystemExit(
            f"Expected textstat {EXPECTED_TEXTSTAT}, found {version}; refusing to change algorithms."
        )
    textstat.set_lang("en_US")
    try:
        cmudict_path = Path(str(nltk.data.find("corpora/cmudict/cmudict")))
    except LookupError as exc:
        raise SystemExit(
            "NLTK cmudict is missing. Run: python -m nltk.downloader cmudict"
        ) from exc
    # textstat otherwise calls nltk.download() on every fresh process even when
    # this verified local corpus is present. Disable that network check; the
    # pronunciation data used below are unchanged and their hash is recorded.
    nltk.download = lambda *args, **kwargs: True  # type: ignore[assignment]
    return version, sha256(cmudict_path)


def build_detail_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_doc: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"risk": [], "mitigation": []}
    )
    agencies: dict[str, str] = {}
    gold_counts: dict[str, int] = {}
    for task, path in GOLD_FILES.items():
        records = read_jsonl(path)
        gold_counts[task] = len(records)
        for record in records:
            gold_by_doc[record["doc_id"]][task].append(record["topic_input_text"])
            agencies[record["doc_id"]] = record["agency"]

    if gold_counts != {"risk": 444, "mitigation": 821}:
        raise SystemExit(f"Unexpected canonical gold counts: {gold_counts}")
    if len(gold_by_doc) != 73:
        raise SystemExit(f"Expected 73 gold documents, found {len(gold_by_doc)}")

    rows: list[dict[str, Any]] = []
    for doc_id in sorted(gold_by_doc):
        for task in TASK_ORDER:
            result = score_list(gold_by_doc[doc_id][task])
            rows.append(
                output_row(
                    source="agency_reference",
                    model="agency_reference",
                    condition="agency",
                    seed="",
                    doc_id=doc_id,
                    task=task,
                    status="scored" if result is not None else "empty_reference",
                    result=result,
                )
            )

    predictions = read_jsonl(PREDICTIONS)
    keys = [
        (r["model"], r["condition"], int(r["seed"]), r["doc_id"])
        for r in predictions
    ]
    if len(predictions) != 2190 or len(set(keys)) != 2190:
        raise SystemExit(
            f"Expected 2,190 unique prediction cells, found {len(predictions)} rows "
            f"and {len(set(keys))} keys"
        )
    if set(r["doc_id"] for r in predictions) != set(gold_by_doc):
        raise SystemExit("Prediction and canonical-gold document sets differ")

    status_counts = Counter(r.get("parse_status", "missing") for r in predictions)
    for record in predictions:
        base_status = record.get("parse_status", "missing")
        for task, plural in (("risk", "risks"), ("mitigation", "mitigations")):
            result = None
            status = base_status
            # Keep the model/reference comparison paired. Four documents have
            # no canonical risk span, so neither side receives a risk FRE there.
            if not gold_by_doc[record["doc_id"]][task]:
                status = "out_of_scope_empty_reference"
            elif base_status == "valid":
                result = score_list(
                    item.get("description", "")
                    for item in record.get("predicted", {}).get(plural, [])
                )
                status = "scored" if result is not None else "empty_prediction"
            rows.append(
                output_row(
                    source="model_prediction",
                    model=record["model"],
                    condition=record["condition"],
                    seed=int(record["seed"]),
                    doc_id=record["doc_id"],
                    task=task,
                    status=status,
                    result=result,
                )
            )

    metadata = {
        "gold_counts": gold_counts,
        "documents": len(gold_by_doc),
        "prediction_rows": len(predictions),
        "prediction_parse_status_counts": dict(sorted(status_counts.items())),
        "detail_status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "agencies": dict(sorted(Counter(agencies.values()).items())),
    }
    return rows, metadata


def build_summary(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [row for row in detail if row["status"] == "scored"]
    summary: list[dict[str, Any]] = []

    for task in TASK_ORDER:
        rows = [
            row
            for row in scored
            if row["source"] == "agency_reference" and row["task"] == task
        ]
        summary.append(
            {
                "model": "agency_reference",
                "condition": "agency",
                "task": task,
                "n_seeds": 1,
                "n_document_cells_scored": len(rows),
                "documents_scored_per_seed": str(len(rows)),
                "mean_flesch_reading_ease": mean(
                    float(row["flesch_reading_ease"]) for row in rows
                ),
                "median_document_flesch": statistics.median(
                    float(row["flesch_reading_ease"]) for row in rows
                ),
                "min_seed_mean_flesch": "",
                "max_seed_mean_flesch": "",
                "seed_sd_flesch": "",
                "mean_words_per_item": mean(float(row["words_per_item"]) for row in rows),
                "seed_sd_words_per_item": "",
            }
        )

    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            for task in TASK_ORDER:
                rows = [
                    row
                    for row in scored
                    if row["source"] == "model_prediction"
                    and row["model"] == model
                    and row["condition"] == condition
                    and row["task"] == task
                ]
                by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    by_seed[int(row["seed"])].append(row)
                if sorted(by_seed) != [1, 2, 3]:
                    raise SystemExit(f"Missing seed in {(model, condition, task)}")
                seed_fre = [
                    mean(float(row["flesch_reading_ease"]) for row in by_seed[seed])
                    for seed in sorted(by_seed)
                ]
                seed_words = [
                    mean(float(row["words_per_item"]) for row in by_seed[seed])
                    for seed in sorted(by_seed)
                ]
                summary.append(
                    {
                        "model": model,
                        "condition": condition,
                        "task": task,
                        "n_seeds": len(by_seed),
                        "n_document_cells_scored": len(rows),
                        "documents_scored_per_seed": "/".join(
                            str(len(by_seed[seed])) for seed in sorted(by_seed)
                        ),
                        "mean_flesch_reading_ease": mean(seed_fre),
                        "median_document_flesch": statistics.median(
                            float(row["flesch_reading_ease"]) for row in rows
                        ),
                        "min_seed_mean_flesch": min(seed_fre),
                        "max_seed_mean_flesch": max(seed_fre),
                        "seed_sd_flesch": statistics.stdev(seed_fre),
                        "mean_words_per_item": mean(seed_words),
                        "seed_sd_words_per_item": statistics.stdev(seed_words),
                    }
                )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    textstat_version, cmudict_sha = validate_environment()
    detail, inventory = build_detail_rows()
    summary = build_summary(detail)

    detail_path = OUT_DIR / "flesch_per_document.csv"
    summary_path = OUT_DIR / "flesch_summary.csv"
    metadata_path = OUT_DIR / "metadata.json"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)

    metadata = {
        "purpose": "Descriptive readability diagnostic; not a model-quality metric.",
        "formula": "206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)",
        "implementation": {
            "textstat": textstat_version,
            "language": "en_US",
            "nltk": importlib.metadata.version("nltk"),
            "cmudict_sha256": cmudict_sha,
            "silent_fallback": False,
        },
        "text_fields": {
            "agency_reference": "canonical risk/mitigation .topic_input_text (identical to .text)",
            "model_prediction": ".predicted.risks/mitigations[].description",
        },
        "preprocessing": (
            "Collapse whitespace; append terminal punctuation only when absent; "
            "join all same-task items in a document with a newline."
        ),
        "aggregation": (
            "Compute FRE on each document-task list; macro-average documents within "
            "each seed; average the three seed means. Invalid API/parse cells are missing, "
            "not assigned a readability score."
        ),
        "inputs": {
            "predictions": {"path": str(PREDICTIONS.relative_to(ROOT)), "sha256": sha256(PREDICTIONS)},
            **{
                task: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for task, path in GOLD_FILES.items()
            },
        },
        "inventory": inventory,
        "outputs": {
            "detail": detail_path.name,
            "summary": summary_path.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    for row in summary:
        print(
            f"{row['model']:34s} {row['condition']:10s} {row['task']:10s} "
            f"FRE={float(row['mean_flesch_reading_ease']):5.1f} "
            f"words/item={float(row['mean_words_per_item']):5.1f} "
            f"docs={row['documents_scored_per_seed']}"
        )


if __name__ == "__main__":
    main()
