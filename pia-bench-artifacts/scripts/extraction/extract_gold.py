from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

import prompt as P

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs"
CACHE = HERE / "cache"
DEFAULT_MODEL = "gpt-5.4"


class SpanType(str, Enum):
    risk = "risk"
    mitigation = "mitigation"
    null_finding = "null_finding"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    span_id: int
    type: SpanType
    sentence_ids: list[int] = Field(min_length=1)
    explanation: str
    confidence: Confidence


class Link(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mitigation_span_id: int
    risk_span_id: int
    confidence: Confidence


class ChunkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spans: list[Span]
    links: list[Link]


def validate(resp: ChunkResponse, chunk: dict) -> list[str]:
    """Return a list of problems. Empty means the response is structurally sound."""
    problems: list[str] = []
    valid_sids = {s["sid"] for s in chunk["sentences"]}
    seen_sids: dict[int, int] = {}
    span_ids: dict[int, SpanType] = {}

    for sp in resp.spans:
        if sp.span_id in span_ids:
            problems.append(f"duplicate span_id {sp.span_id}")
        span_ids[sp.span_id] = sp.type
        for sid in sp.sentence_ids:
            if sid not in valid_sids:
                problems.append(f"span {sp.span_id} cites sentence {sid}, which does not exist")
            elif sid in seen_sids:
                problems.append(
                    f"sentence {sid} claimed by spans {seen_sids[sid]} and {sp.span_id}")
            else:
                seen_sids[sid] = sp.span_id
        ordered = sorted(sp.sentence_ids)
        if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
            problems.append(f"span {sp.span_id} sentences are not consecutive: {sp.sentence_ids}")

    for ln in resp.links:
        if ln.mitigation_span_id not in span_ids:
            problems.append(f"link cites unknown mitigation span {ln.mitigation_span_id}")
        elif span_ids[ln.mitigation_span_id] != SpanType.mitigation:
            problems.append(f"link source span {ln.mitigation_span_id} is not a mitigation")
        if ln.risk_span_id not in span_ids:
            problems.append(f"link cites unknown risk span {ln.risk_span_id}")
        elif span_ids[ln.risk_span_id] != SpanType.risk:
            problems.append(f"link target span {ln.risk_span_id} is not a risk")
    return problems


def call_model(client, model, effort, max_tokens, chunk, force=False):
    user = P.build_user_prompt(chunk)
    schema = json.dumps(ChunkResponse.model_json_schema(), sort_keys=True)
    key = hashlib.sha256(
        f"{model}\n{effort}\n{P.PROMPT_VERSION}\n{P.SYSTEM_PROMPT}\n{schema}\n{user}"
        .encode()).hexdigest()
    cache_file = CACHE / f"{chunk['chunk_id'].replace('#', '_')}_{key[:16]}.json"

    if cache_file.exists() and not force:
        c = json.loads(cache_file.read_text())
        return ChunkResponse.model_validate(c["response"]), c.get("usage", {}), True, c["returned_model"]

    response = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": P.SYSTEM_PROMPT},
               {"role": "user", "content": user}],
        text_format=ChunkResponse,
        reasoning={"effort": effort, "context": "current_turn"},
        max_output_tokens=max_tokens,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"{chunk['chunk_id']}: model returned no parsed output "
                           f"(status={response.status})")
    u = response.usage
    usage = {"input_tokens": getattr(u, "input_tokens", 0),
             "output_tokens": getattr(u, "output_tokens", 0),
             "reasoning_tokens": int(getattr(getattr(u, "output_tokens_details", None),
                                             "reasoning_tokens", 0) or 0)}
    cache_file.write_text(json.dumps({
        "chunk_id": chunk["chunk_id"], "requested_model": model,
        "returned_model": response.model, "reasoning_effort": effort,
        "prompt_version": P.PROMPT_VERSION, "store": False,
        "usage": usage, "response": parsed.model_dump(mode="json")}, indent=2))
    return parsed, usage, False, response.model


def materialise(resp: ChunkResponse, chunk: dict) -> dict:
    """Attach verbatim text and unit provenance to the model's sentence ids."""
    by_sid = {s["sid"]: s for s in chunk["sentences"]}
    spans = []
    for sp in resp.spans:
        sents = [by_sid[i] for i in sorted(sp.sentence_ids) if i in by_sid]
        spans.append({
            "span_id": sp.span_id,
            "type": sp.type.value,
            "sentence_ids": sorted(sp.sentence_ids),
            "text": " ".join(s["text"] for s in sents),
            "unit_ids": sorted({s["unit_id"] for s in sents}),
            "section": sents[0].get("question") or sents[0].get("heading") or "" if sents else "",
            "explanation": sp.explanation,
            "confidence": sp.confidence.value,
        })
    return {
        "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "agency": chunk["agency"],
        "n_sentences": chunk["n_sentences"],
        "spans": spans,
        "links": [l.model_dump(mode="json") for l in resp.links],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--pilot", action="store_true",
                    help="only chunks containing units from the 50-unit pilot sample")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk-id", action="append", default=[])
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    OUTPUTS.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)

    chunks = [json.loads(l) for l in (INPUTS / "documents.jsonl").open()]
    if args.pilot:
        pilot_units = {json.loads(l)["unit_id"] for l in (INPUTS / "pilot_50.jsonl").open()}
        chunks = [c for c in chunks
                  if any(s["unit_id"] in pilot_units for s in c["sentences"])]
    if args.chunk_id:
        chunks = [c for c in chunks if c["chunk_id"] in set(args.chunk_id)]
    if args.limit:
        chunks = chunks[:args.limit]

    print(f"model {args.model}  effort {args.reasoning_effort}  prompt {P.PROMPT_VERSION}")
    print(f"chunks to process: {len(chunks)}")

    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    cached_n = 0
    all_problems: dict[str, list[str]] = {}
    returned_models: set[str] = set()
    counts = {"risk": 0, "mitigation": 0, "null_finding": 0, "links": 0}

    for i, chunk in enumerate(chunks, 1):
        resp, usage, was_cached, returned = call_model(
            client_, args.model, args.reasoning_effort, args.max_output_tokens,
            chunk, force=args.force)
        returned_models.add(returned)
        cached_n += int(was_cached)
        for k in totals:
            totals[k] += usage.get(k, 0)

        problems = validate(resp, chunk)
        if problems:
            all_problems[chunk["chunk_id"]] = problems

        out = materialise(resp, chunk)
        out["validation_problems"] = problems
        for sp in out["spans"]:
            counts[sp["type"]] += 1
        counts["links"] += len(out["links"])

        (OUTPUTS / f"{chunk['chunk_id'].replace('#', '_')}.json").write_text(
            json.dumps(out, indent=2))
        flag = "cache" if was_cached else "api  "
        bad = f"  !! {len(problems)} problems" if problems else ""
        print(f"[{i:>3}/{len(chunks)}] {flag} {chunk['chunk_id']:<48} "
              f"spans={len(out['spans']):>3} links={len(out['links']):>2}{bad}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_model": args.model,
        "returned_models": sorted(returned_models),
        "reasoning_effort": args.reasoning_effort,
        "prompt_version": P.PROMPT_VERSION,
        "store": False,
        "chunks": len(chunks), "from_cache": cached_n,
        "usage": totals,
        "span_counts": counts,
        "chunks_with_validation_problems": len(all_problems),
        "validation_problems": all_problems,
    }
    name = "run_summary_pilot.json" if args.pilot else "run_summary.json"
    (HERE / name).write_text(json.dumps(summary, indent=2))

    print(f"\nspans: risk {counts['risk']}  mitigation {counts['mitigation']}  "
          f"null_finding {counts['null_finding']}  links {counts['links']}")
    print(f"tokens: in {totals['input_tokens']:,}  out {totals['output_tokens']:,}  "
          f"(reasoning {totals['reasoning_tokens']:,})   cached chunks: {cached_n}")
    print(f"chunks with validation problems: {len(all_problems)}")
    print(f"summary -> {name}")
    return 0


if __name__ == "__main__":
    client_ = OpenAI()
    raise SystemExit(main())
