#!/usr/bin/env python3
"""Build the document-level retrieval annotation sheet.

Measures the question the sentence-level pilot cannot answer: over a WHOLE document, does the
model find all the risk and mitigation content?

One row per unit, every unit of three complete documents, in document order. The annotator marks
which units CONTAIN risk content and which CONTAIN mitigation content — a coarser, faster
judgement than sentence-level typing.

The model's output is deliberately NOT shown: the measurement is only valid if the labelling is
blind.
"""

from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
OUT = HERE / "retrieval_annotation.xlsx"

# Chosen on structure alone — agency spread and question density — not on model performance.
DOCS = [
    ("007_cfpb_consumer-experience-research-pia",
     "CFPB · questionnaire with explicit risk questions"),
    ("146_opm_research-and-oversight-repository-rovr",
     "OPM · mixed structure"),
    ("473_usda_damwatch-damwatch",
     "USDA · narrative prose, NO risk questions at all — the hard case"),
]

TEAL, BAND, INK = "1F9D82", "EAF4F1", "0B0B0B"


def main():
    units = [json.loads(l) for l in (HERE / "inputs/units.jsonl").open()]
    by_doc = {}
    for u in units:
        by_doc.setdefault(u["doc_id"], []).append(u)

    wb = Workbook()
    ws = wb.active
    ws.title = "Annotate"
    head = ["#", "section / governing question", "UNIT TEXT",
            "RISK?", "MITIGATION?", "NOTE", "unit_id (do not edit)"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=TEAL)
    ws.freeze_panes = "C2"

    dv = DataValidation(type="list", formula1='"yes"', allow_blank=True, showDropDown=False)
    ws.add_data_validation(dv)

    total = 0
    for doc_id, note in DOCS:
        us = sorted(by_doc[doc_id], key=lambda x: x["unit_id"])
        ws.append([f"▼ {doc_id}", note, f"{len(us)} units — read straight through", "", "", "", ""])
        r = ws.max_row
        for c in ws[r]:
            c.fill = PatternFill("solid", fgColor=BAND)
            c.font = Font(bold=True, color=INK)
        for i, u in enumerate(us, 1):
            text = " ".join(s["text"] for s in u["sentences"]).strip()
            ws.append([i, u.get("question") or u.get("heading") or "",
                       text, "", "", "", u["unit_id"]])
            row = ws.max_row
            dv.add(ws.cell(row=row, column=4))
            dv.add(ws.cell(row=row, column=5))
            for col in (2, 3, 6):
                ws.cell(row=row, column=col).alignment = Alignment(wrap_text=True, vertical="top")
            ws.cell(row=row, column=7).font = Font(size=8, color="999999")
            total += 1
        ws.append([])

    for col, w in {"A": 6, "B": 34, "C": 96, "D": 10, "E": 14, "F": 30, "G": 30}.items():
        ws.column_dimensions[col].width = w

    rm = wb.create_sheet("Read me", 0)
    for a, b in [
        ("RETRIEVAL CHECK — does the model find everything?", ""),
        ("", ""),
        ("What this measures", "The earlier sheet asked: given a sentence, is it typed correctly?"),
        ("", "This one asks: over a WHOLE document, is any risk or mitigation content MISSED?"),
        ("", "That is the more important question — a missed statement is absent from the answer"),
        ("", "key AND left sitting in the benchmark input."),
        ("", ""),
        ("What to do", "Read each document straight through, in order. For every unit, ask two questions:"),
        ("  RISK? (column D)", "Does this unit contain ANY statement of a privacy risk?  ->  type 'yes'"),
        ("  MITIGATION? (column E)", "Does this unit contain ANY mitigation?  ->  type 'yes'"),
        ("", "Both can be 'yes'. Leave blank for no. Blank is the normal answer."),
        ("", ""),
        ("This is a SCAN, not a close read", "You are not typing every sentence — only asking whether the unit contains"),
        ("", "any such statement. Go at pace. Roughly 150 units across three documents."),
        ("", ""),
        ("Do not skip units", "Even boilerplate and headers. The whole point is completeness — a unit you"),
        ("", "skip looks identical to a unit with nothing in it."),
        ("", ""),
        ("Definitions", "Exactly as in the gold codebook — no change:"),
        ("  risk", "a statement that privacy could be harmed, or information handled undesirably"),
        ("  mitigation", "a measure presented as answering a risk — by its own wording, by the question"),
        ("", "it sits under, or by the risk stated immediately before it"),
        ("  not a mitigation", "a control stated as plain fact with none of those three connections"),
        ("  'risks are minimal'", "-> RISK (minimal is not none)"),
        ("  'no risks exist'", "-> mark neither; note it as a null finding if you like"),
        ("", ""),
        ("The model's output is not shown", "On purpose. If you could see it, the measurement would be worthless."),
        ("", ""),
        ("Documents", "Chosen for structural spread, not for how the model performed on them:"),
        ("  007_cfpb", "questionnaire with explicit risk questions"),
        ("  146_opm", "mixed structure"),
        ("  473_usda_damwatch", "narrative prose, no risk questions at all — the hard case"),
        ("", ""),
        ("Please time yourself", "A rough figure per document is enough."),
    ]:
        rm.append([a, b])
        rm.cell(row=rm.max_row, column=1).font = Font(bold=True)
        for col in (1, 2):
            rm.cell(row=rm.max_row, column=col).alignment = Alignment(wrap_text=True, vertical="top")
    rm["A1"].font = Font(bold=True, size=13)
    rm.column_dimensions["A"].width = 34
    rm.column_dimensions["B"].width = 96

    wb.save(OUT)
    print(f"wrote {OUT.name}: {total} units across {len(DOCS)} documents")
    for d, _ in DOCS:
        print(f"   {d}: {len(by_doc[d])} units")


if __name__ == "__main__":
    main()
