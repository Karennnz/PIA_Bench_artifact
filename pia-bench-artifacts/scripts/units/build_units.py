#!/usr/bin/env python3
"""Build the model-ready unit dataset for gold extraction (Plan B).

Reassembles ALL corrected blocks — both sides of the old input/gold split — into
question-answer units, sentence-splits each unit, and assigns stable IDs.

The old split is deliberately ignored: Plan B re-extracts the risk/mitigation content from the
whole cleaned document. The old `source_status` is carried through as metadata only, so it can be
compared against later, never used as input.

Output: units.jsonl, one record per unit.

The reported run retained question-only units as metadata but did not insert
their question text into the chunk sentence stream. That historical behavior is
the default because later span IDs depend on it. ``--include-question-only-text``
enables a corrected representation, but its sentence IDs are not compatible
with the released extraction outputs.
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

# Documents longer than this are chunked at unit boundaries, with one unit of overlap.
MAX_SENTS_PER_CHUNK = 250

# Sentence splitter: split on . ! ? followed by whitespace + capital/digit, while protecting
# common abbreviations that appear in these documents.
ABBREV = (r"U\.S|U\.S\.C|e\.g|i\.e|etc|No|Nos|Inc|Corp|Dept|Div|Sec|Fig|Mr|Mrs|Ms|Dr|Jr|Sr|"
          r"vs|approx|St|Ave|Rd|Gov|Admin|Op|Cit|Pub|L|Stat|C\.F\.R|F\.R|Ct|Cir")
BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
ABBREV_END = re.compile(rf"\b(?:{ABBREV})\.$", re.I)
# A trailing section number ("4.1.", "8.", "1.6.") is a question label, not a sentence end.
NUMBER_END = re.compile(r"(?:^|\s)\d+(?:\.\d+)*\.$")


def sentences(text: str) -> list[str]:
    """Split on sentence boundaries, refusing to split directly after a known abbreviation."""
    text = " ".join(text.split())
    if not text:
        return []
    parts, start = [], 0
    for m in BOUNDARY.finditer(text):
        candidate = text[start:m.start()]
        if ABBREV_END.search(candidate) or NUMBER_END.search(candidate):
            continue  # "U.S. Department" and "4.1. How does..." are not sentence breaks
        parts.append(candidate.strip())
        start = m.end()
    parts.append(text[start:].strip())
    parts = [p for p in parts if p]
    # A unit with no terminal punctuation (headings, table rows, list items) stays whole.
    return parts or [text]


def is_question(t: str) -> bool:
    t = " ".join(t.split())
    return t.endswith("?") and len(t) > 15


def reassemble(blocks):
    """Same rule as prototype_reassembly.py, applied across the whole document."""
    units, cur, prev_head = [], None, object()
    for b in blocks:
        txt = " ".join((b.get("text") or "").split())
        head = (b.get("heading") or "").strip()
        ut = b.get("unit_type")
        boundary = (cur is None or head != prev_head or is_question(txt)
                    or ut == "table" or cur["unit_type"] == "table")
        if boundary:
            if cur:
                units.append(cur)
            cur = {"heading": head, "unit_type": ut,
                   "question": txt if is_question(txt) else None,
                   "body": [] if is_question(txt) else [txt],
                   "blocks": [b]}
        else:
            cur["blocks"].append(b)
            cur["body"].append(txt)
        prev_head = head
    if cur:
        units.append(cur)
    return units


def chunk_document(doc_id, agency, unit_recs):
    """Group a document's units into model-sized chunks with globally numbered sentences.

    The model reads a document (or chunk) rather than a unit: 54% of units are a single
    sentence, and both the judgement and the risk->mitigation linking need surrounding context.
    """
    chunks, cur, cur_sents = [], [], 0
    for u in unit_recs:
        if cur and cur_sents + u["n_sentences"] > MAX_SENTS_PER_CHUNK:
            chunks.append(cur)
            cur, cur_sents = [cur[-1]], cur[-1]["n_sentences"]  # one unit of overlap
        cur.append(u)
        cur_sents += u["n_sentences"]
    if cur:
        chunks.append(cur)

    out = []
    for ci, units in enumerate(chunks):
        sents, sid = [], 0
        for u in units:
            for s in u["sentences"]:
                sents.append({"sid": sid, "text": s["text"], "unit_id": u["unit_id"],
                              "question": u["question"], "heading": u["heading"]})
                sid += 1
        out.append({"chunk_id": f"{doc_id}#chunk#{ci:02d}", "doc_id": doc_id,
                    "agency": agency, "n_units": len(units), "n_sentences": len(sents),
                    "n_chars": sum(len(s["text"]) for s in sents), "sentences": sents})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of corrected per-document JSON records containing a blocks array.",
    )
    parser.add_argument("--units-output", type=Path, required=True)
    parser.add_argument("--documents-output", type=Path, required=True)
    parser.add_argument(
        "--include-question-only-text",
        action="store_true",
        help=(
            "include an otherwise-empty question as model-visible text; this changes "
            "historical sentence IDs and is incompatible with the released extractions"
        ),
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {args.input_dir}")
    input_paths = {path.resolve() for path in files}
    units_output = args.units_output.resolve()
    documents_output = args.documents_output.resolve()
    if units_output == documents_output:
        raise SystemExit("--units-output and --documents-output must differ")
    if units_output in input_paths or documents_output in input_paths:
        raise SystemExit("An output path aliases an input JSON file")
    args.units_output.parent.mkdir(parents=True, exist_ok=True)
    args.documents_output.parent.mkdir(parents=True, exist_ok=True)
    n_units = 0
    all_docs = []
    with args.units_output.open("w", encoding="utf-8") as fh:
        for f in files:
            doc = json.loads(f.read_text())
            doc_id = doc["document_id"]
            # source_audit records carry no agency field; it is the second token of the doc id
            # (e.g. "004_cfpb_hud-counselor-tool-pia" -> "cfpb").
            parts = doc_id.split("_")
            agency = parts[1] if len(parts) > 1 else ""
            doc_units = []
            for i, u in enumerate(reassemble(doc["blocks"])):
                body = " ".join(x for x in u["body"] if x).strip()
                if not body and u["question"] and args.include_question_only_text:
                    body = u["question"]
                sents = sentences(body)
                rec = {
                    "unit_id": f"{doc_id}#unit#{i:04d}",
                    "doc_id": doc_id,
                    "agency": agency,
                    "heading": u["heading"],
                    "question": u["question"],
                    "unit_type": u["unit_type"],
                    "sentences": [{"sid": j, "text": s} for j, s in enumerate(sents)],
                    "n_sentences": len(sents),
                    "n_chars": len(body),
                    "block_ids": [b.get("docling_item_refs") for b in u["blocks"]],
                    # metadata only — never shown to the model
                    "_legacy_status": sorted({b.get("source_status") for b in u["blocks"]}),
                    "_legacy_gold_blocks": sum(
                        1 for b in u["blocks"] if b.get("source_status") == "gold_removed"),
                }
                fh.write(json.dumps(rec) + "\n")
                doc_units.append(rec)
                n_units += 1
            all_docs.extend(chunk_document(doc_id, agency, doc_units))

    with args.documents_output.open("w", encoding="utf-8") as fh:
        for c in all_docs:
            fh.write(json.dumps(c) + "\n")

    cs = sorted(c["n_sentences"] for c in all_docs)
    print(f"documents {len(files)}   units {n_units}   ->  {args.units_output.name}")
    print(f"model chunks {len(all_docs)}  ->  {args.documents_output.name}")
    print(f"  sentences/chunk: median {cs[len(cs)//2]}  p90 {cs[int(.9*len(cs))]}  max {max(cs)}")
    print(f"  est. tokens/chunk: median ~{sorted(c['n_chars'] for c in all_docs)[len(cs)//2]//4:,}"
          f"  max ~{max(c['n_chars'] for c in all_docs)//4:,}")

    # quick shape report
    recs = [json.loads(l) for l in args.units_output.open(encoding="utf-8")]
    ns = sorted(r["n_sentences"] for r in recs)
    print(f"sentences per unit: median {ns[len(ns)//2]}  p90 {ns[int(.9*len(ns))]}  max {max(ns)}")
    print(f"units with a question: {sum(1 for r in recs if r['question']):,}")
    print(f"units containing legacy gold blocks: "
          f"{sum(1 for r in recs if r['_legacy_gold_blocks']):,}")
    print(f"total sentences: {sum(r['n_sentences'] for r in recs):,}")


if __name__ == "__main__":
    main()
