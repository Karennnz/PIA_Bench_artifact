from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_attributes"
CACHE = HERE / "cache_attributes"
PROMPT_VERSION = "attr-v1"

SYSTEM_PROMPT = """\
You are annotating US federal Privacy Impact Assessments (PIAs) for a research benchmark.

You will be given the units of one PIA document. A unit is one question from the form plus the
agency's answer, or a passage of narrative. For EVERY unit, say which of the eight input attributes
it explicitly provides.

THE EIGHT ATTRIBUTES

context             what the system/project is, its nature, scope and organisational setting
purpose             why the information is collected; the need, goal or intended benefit
legal_basis         the law, regulation, executive order or agreement permitting the processing
information_types   the categories and characteristics of personal information involved
operations          what is done with the information: obtained, checked, used, transformed,
                    analysed, transmitted, stored or deleted
recipients          people, roles, organisations or systems that receive it or have normal
                    operational access
retention           how long it is kept, or the event/schedule/rule determining disposal
assets              the technology and infrastructure supporting, hosting, storing or processing it

Use "none" when a unit provides none of the eight.

RULES

1. MULTI-LABEL. A unit often provides two or three attributes. Return all that apply. Do not
   force a single label.
2. The test is "does this unit EXPLICITLY provide this information", not "is it related to" and
   not "could a reader infer it".
3. Never combine "none" with a positive label.
4. Boundary cases that matter:
   - a goal without a system description is purpose, not context
   - a data source or collection method that does not identify the information is operations,
     not information_types
   - a recipient organisation is recipients, not information_types
   - technology named without an action is assets, not operations
   - storage with no duration or disposal rule is NOT retention
   - a data subject's right to access their own record is NOT recipients; operational access is
   - safeguards (encryption, authentication, monitoring, training) mentioned only as controls are
     none, unless the unit also states a recipient, data type or other positive attribute
5. The question text alone is not evidence. The agency's answer must provide the information.

Return one record per unit, using the unit_id exactly as given, with a confidence of high, medium
or low.
"""


class Attribute(str, Enum):
    context = "context"
    purpose = "purpose"
    legal_basis = "legal_basis"
    information_types = "information_types"
    operations = "operations"
    recipients = "recipients"
    retention = "retention"
    assets = "assets"
    none = "none"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class UnitLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    attributes: list[Attribute] = Field(min_length=1)
    confidence: Confidence


class DocResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: list[UnitLabel]


def build_user_prompt(doc_id, agency, units):
    lines = [f"Document: {doc_id}", f"Agency: {agency}",
             f"Units: {len(units)}", "", "UNITS"]
    for u in units:
        q = u.get("question") or u.get("heading") or "(no heading)"
        text = " ".join(s["text"] for s in u["sentences"])
        lines.append(f"\n[{u['unit_id']}]")
        lines.append(f"  section: {q}")
        lines.append(f"  text   : {text}")
    return "\n".join(lines)


def call(client, model, effort, max_tokens, doc_id, agency, units, force=False):
    user = build_user_prompt(doc_id, agency, units)
    schema = json.dumps(DocResponse.model_json_schema(), sort_keys=True)
    key = hashlib.sha256(
        f"{model}\n{effort}\n{PROMPT_VERSION}\n{SYSTEM_PROMPT}\n{schema}\n{user}".encode()
    ).hexdigest()
    cf = CACHE / f"{doc_id}_{key[:16]}.json"
    if cf.exists() and not force:
        c = json.loads(cf.read_text())
        return DocResponse.model_validate(c["response"]), c.get("usage", {}), True

    resp = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": user}],
        text_format=DocResponse,
        reasoning={"effort": effort, "context": "current_turn"},
        max_output_tokens=max_tokens,
        store=False,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError(f"{doc_id}: no parsed output (status={resp.status})")
    u = resp.usage
    usage = {"input_tokens": getattr(u, "input_tokens", 0),
             "output_tokens": getattr(u, "output_tokens", 0)}
    cf.write_text(json.dumps({"document_id": doc_id, "returned_model": resp.model,
                              "prompt_version": PROMPT_VERSION, "usage": usage,
                              "response": parsed.model_dump(mode="json")}, indent=2))
    return parsed, usage, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)

    units = [json.loads(l) for l in (HERE / "inputs/units.jsonl").open()]
    by_doc = collections.OrderedDict()
    for u in units:
        by_doc.setdefault(u["doc_id"], []).append(u)
    docs = list(by_doc)[:args.limit] if args.limit else list(by_doc)

    print(f"model {args.model}  prompt {PROMPT_VERSION}  documents {len(docs)}")
    totals = collections.Counter(); cached = 0; labelled = 0; missing_total = 0
    dist = collections.Counter(); per_n = collections.Counter()

    for i, doc_id in enumerate(docs, 1):
        us = by_doc[doc_id]
        resp, usage, was_cached = call(client_, args.model, args.reasoning_effort,
                                       args.max_output_tokens, doc_id,
                                       us[0]["agency"], us, force=args.force)
        cached += int(was_cached)
        for k, v in usage.items():
            totals[k] += v
        valid = {u["unit_id"] for u in us}
        got = {r.unit_id: [a.value for a in r.attributes] for r in resp.units
               if r.unit_id in valid}
        missing = sorted(valid - set(got))
        missing_total += len(missing)
        for uid, attrs in got.items():
            labelled += 1
            pos = [a for a in attrs if a != "none"]
            per_n[len(pos)] += 1
            for a in attrs:
                dist[a] += 1
        (OUT / f"{doc_id}.json").write_text(json.dumps(
            {"document_id": doc_id, "prompt_version": PROMPT_VERSION,
             "units": [{"unit_id": r.unit_id,
                        "attributes": [a.value for a in r.attributes],
                        "confidence": r.confidence.value}
                       for r in resp.units if r.unit_id in valid],
             "units_not_returned": missing}, indent=2))
        flag = "cache" if was_cached else "api  "
        print(f"[{i:>3}/{len(docs)}] {flag} {doc_id[:46]:<46} "
              f"units={len(got)}/{len(us)}" + (f"  !! {len(missing)} missing" if missing else ""))

    summary = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "model": args.model, "prompt_version": PROMPT_VERSION,
               "documents": len(docs), "units_labelled": labelled,
               "units_not_returned": missing_total, "from_cache": cached,
               "usage": dict(totals), "attribute_counts": dict(dist),
               "labels_per_unit": dict(per_n)}
    (HERE / "run_summary_attributes.json").write_text(json.dumps(summary, indent=2))
    mean = sum(k * v for k, v in per_n.items()) / max(1, sum(per_n.values()))
    print(f"\nunits labelled {labelled}   not returned {missing_total}")
    print(f"mean labels per unit {mean:.2f}   distribution {dict(sorted(per_n.items()))}")
    print(f"tokens in {totals['input_tokens']:,} out {totals['output_tokens']:,}")
    return 0


if __name__ == "__main__":
    client_ = OpenAI()
    raise SystemExit(main())
