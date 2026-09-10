from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTRS = ["context", "purpose", "legal_basis", "information_types", "operations", "assets", "recipients", "retention"]
LABEL = {a: a.replace("_", " ").upper() for a in ATTRS}


def render(rec) -> tuple[str, int]:
    blocks = ["SYSTEM CONTEXT, organised into labelled fields.\n"]
    n = 0
    for a in ATTRS:
        items = rec["input_attributes"].get(a, [])
        if not items:
            continue
        lines = [f"[{LABEL[a]}]"]
        for g in items:
            lines.append(g["text"])
            n += 1
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_files = sorted(args.benchmark_dir.glob("*.json"))
    if not source_files:
        raise SystemExit(f"No benchmark JSON files found in {args.benchmark_dir}")
    if args.output.resolve() in {path.resolve() for path in source_files}:
        raise SystemExit("--output aliases an input benchmark JSON file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in source_files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not {"doc_id", "agency", "input_attributes"}.issubset(rec):
            raise SystemExit(f"Invalid benchmark record: {f.name}")
        served, n = render(rec)
        rows.append({"doc_id": rec["doc_id"], "agency": rec["agency"],
                     "served_input": served, "n_input_paras": n})
    doc_ids = [row["doc_id"] for row in rows]
    if len(doc_ids) != len(set(doc_ids)):
        raise SystemExit("Duplicate doc_id values in benchmark records")
    with args.output.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    empty = [r["doc_id"] for r in rows if r["n_input_paras"] == 0]
    print(f"[inputs] wrote {len(rows)} served inputs to {args.output.name}")
    print(f"[inputs] paragraphs per doc: min {min(r['n_input_paras'] for r in rows)}, "
          f"max {max(r['n_input_paras'] for r in rows)}, "
          f"total {sum(r['n_input_paras'] for r in rows)}")
    if empty:
        print(f"[inputs] WARNING docs with empty input ({len(empty)}): {empty}")


if __name__ == "__main__":
    main()
