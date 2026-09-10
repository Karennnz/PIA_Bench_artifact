#!/usr/bin/env python3
"""Score document-level retrieval: over a whole document, does the model find the risk and
mitigation content?

Complements score_against_annotation.py, which measures classification on sampled sentences.
This measures completeness over every unit of three complete documents.

Also reports the contamination rate, which falls out for free: a unit the human marks as
containing risk content that sits on the INPUT side of the corpus is a confirmed leak.
"""

from __future__ import annotations

import collections
import glob
import json
import random
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
SHEET = HERE / "retrieval_annotation_final_150.xlsx"


def load_human():
    ws = openpyxl.load_workbook(SHEET)["Annotate"]
    human, order, cur = {}, [], None
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] and str(r[0]).startswith("▼"):
            cur = str(r[0]).replace("▼ ", "").strip()
            continue
        uid = r[6]
        if not uid:
            continue
        tags = set()
        if r[3] and str(r[3]).strip().lower().startswith("y"):
            tags.add("risk")
        if r[4] and str(r[4]).strip().lower().startswith("y"):
            tags.add("mitigation")
        human[uid] = tags
        order.append((cur, uid))
    return human, order


def load_model(docs):
    model = collections.defaultdict(set)
    for f in sorted(glob.glob(str(HERE / "outputs/*.json"))):
        o = json.load(open(f))
        if o["doc_id"] not in docs:
            continue
        for sp in o["spans"]:
            if sp["type"] == "null_finding":
                continue
            for uid in sp["unit_ids"]:
                model[uid].add(sp["type"])
    return model


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    human, order = load_human()
    units = {u["unit_id"]: u for u in
             (json.loads(l) for l in (HERE / "inputs/units.jsonl").open())}
    docs = {units[u]["doc_id"] for u in human if u in units}
    model = load_model(docs)

    L: list[str] = []
    def out(s=""):
        print(s); L.append(s)

    out("# Document-level retrieval — GPT-5.6 vs human\n")
    out(f"Documents: {len(docs)}   units annotated: {len(human)}   "
        f"units the human marked as carrying content: "
        f"{sum(1 for v in human.values() if v)}\n")

    # ---- headline: did the model find the content at all? ------------------
    out("## Retrieval — any risk/mitigation content in the unit\n")
    out("| document | units | human marked | model found | recall | precision |")
    out("|---|---:|---:|---:|---:|---:|")
    tot = [0, 0, 0]
    per_doc = collections.defaultdict(lambda: [0, 0, 0])
    for doc, uid in order:
        H = bool(human.get(uid))
        M = bool(model.get(uid))
        d = units[uid]["doc_id"]
        per_doc[d][0] += int(H and M)
        per_doc[d][1] += int(M and not H)
        per_doc[d][2] += int(H and not M)
    for d in sorted(per_doc):
        tp, fp, fn = per_doc[d]
        p, r, _ = prf(tp, fp, fn)
        n = sum(1 for u in human if units[u]["doc_id"] == d)
        out(f"| {d[:40]} | {n} | {tp+fn} | {tp+fp} | **{r:.3f}** | {p:.3f} |")
        for i in range(3):
            tot[i] += per_doc[d][i]
    P, R, F = prf(*tot)
    out(f"| **all three** | {len(human)} | {tot[0]+tot[2]} | {tot[0]+tot[1]} | "
        f"**{R:.3f}** | **{P:.3f}** |")
    out(f"\n**Overall retrieval recall {R:.3f} · precision {P:.3f} · F1 {F:.3f}**\n")

    # ---- per type ----------------------------------------------------------
    out("## By type\n")
    out("| type | human units | model units | recall | precision |")
    out("|---|---:|---:|---:|---:|")
    for t in ("risk", "mitigation"):
        tp = sum(1 for u in human if t in human[u] and t in model.get(u, set()))
        fp = sum(1 for u in human if t in model.get(u, set()) and t not in human[u])
        fn = sum(1 for u in human if t in human[u] and t not in model.get(u, set()))
        p, r, _ = prf(tp, fp, fn)
        out(f"| {t} | {tp+fn} | {tp+fp} | **{r:.3f}** | {p:.3f} |")

    # ---- contamination -----------------------------------------------------
    out("\n## Contamination — risk content still on the input side\n")
    leaks = [u for u in human if human[u] and
             "gold_removed" not in (units[u].get("_legacy_status") or [])]
    marked = [u for u in human if human[u]]
    out(f"units the human marked as carrying risk/mitigation content: **{len(marked)}**")
    out(f"of those, sitting on the INPUT side (a confirmed leak): **{len(leaks)}** "
        f"({100*len(leaks)/len(marked) if marked else 0:.0f}%)")
    if leaks:
        out("\nexamples:")
        for u in leaks[:6]:
            txt = " ".join(s["text"] for s in units[u]["sentences"])
            out(f"- [{'/'.join(sorted(human[u]))}] `{u.split('#')[0][:24]}` — {txt[:110]}")

    # ---- what was missed ---------------------------------------------------
    missed = [u for u in human if human[u] and not model.get(u)]
    out(f"\n## Missed by the model: {len(missed)} units\n")
    for u in missed[:12]:
        txt = " ".join(s["text"] for s in units[u]["sentences"])
        out(f"- [{'/'.join(sorted(human[u]))}] `{u.split('#')[0][:24]}` — {txt[:115]}")

    # ---- bootstrap ---------------------------------------------------------
    rng = random.Random(20260807)
    names = sorted(docs)
    boot = []
    for _ in range(1000):
        s = [0, 0, 0]
        for _ in names:
            d = rng.choice(names)
            for i in range(3):
                s[i] += per_doc[d][i]
        boot.append(prf(*s)[1])
    boot.sort()
    out(f"\n**recall 95% CI (document bootstrap): [{boot[25]:.3f}, {boot[975]:.3f}]** "
        f"— only 3 documents, so this is wide by construction")

    (HERE / "retrieval_report.md").write_text("\n".join(L) + "\n")
    print("\nwritten to retrieval_report.md")


if __name__ == "__main__":
    main()
