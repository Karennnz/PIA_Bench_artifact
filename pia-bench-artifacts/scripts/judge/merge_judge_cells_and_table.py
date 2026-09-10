"""Fold the per-cell judge outputs for gpt-5.4-mini back into the standard three-seed rows, then build
the one results table for the InfPriv revision (judge F1, BGE F1, Flesch) across all five models.

The judge was run as twelve processes, condition x seed x task, each writing
scoring_judge_gpt54mini/<condition>_s<seed>_<task>/{metrics_judge,per_document_judge}.csv with
seeds=1. judge_score_task1.py's three-seed rule is: macro over documents per seed, then the mean over
seeds, with the interval as the min and max of the per-seed macro-F1. That rule is applied here to the
per-seed rows, so the merged metrics_judge.csv matches what a single three-seed run would write.

The four reported open models keep their published numbers: zero-shot from
zero_shot_analysis/tables/t1_headline_metrics.csv (k=all), cot from t6_zs_vs_cot_paired.csv, and the
plain-prompt DeepSeek run from the pia_handoff copy of per_document_judge.csv (one seed, asterisked in
the paper). Nothing is retyped.

  python merge_judge_cells_and_table.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TB = ROOT / "2026-06-30_meeting_followup/task1_build"
CELLS = TB / "scoring_judge_gpt54mini"
Z = ROOT / "2026-06-30_meeting_followup/zero_shot_analysis/tables"
DJ_DEEPSEEK = ROOT / "pia_handoff_v2_preprocessing_analysis/10_benchmark_and_results/task1_build/scoring_judge/per_document_judge.csv"
TRACK_B = ROOT / "results/track_b_20260904"

NEW_MODEL = "gpt-5.4-mini-2026-03-17"
CONDS, SEEDS, TASKS = ("zero_shot", "cot"), (1, 2, 3), ("risk", "mitigation")

LABELS = {
    "qwen2.5-14b-instruct": "Qwen2.5-14B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "deepseek-r1-distill-qwen-14b": "R1-Distill-14B*",
    NEW_MODEL: "GPT-5.4 mini",
}


def read_csv(p: Path):
    with p.open() as fh:
        return list(csv.DictReader(fh))


def merge_cells():
    missing = [f"{c}_s{s}_{t}" for c in CONDS for s in SEEDS for t in TASKS
               if not (CELLS / f"{c}_s{s}_{t}" / "metrics_judge.csv").exists()]
    if missing:
        sys.exit(f"[merge] {len(missing)} judge cell(s) not finished: {missing}")
    by_key = defaultdict(list)
    per_doc = []
    for c in CONDS:
        for s in SEEDS:
            for t in TASKS:
                d = CELLS / f"{c}_s{s}_{t}"
                for r in read_csv(d / "metrics_judge.csv"):
                    by_key[(r["model"], r["condition"], r["task"], r["k"])].append(r)
                per_doc.extend(read_csv(d / "per_document_judge.csv"))
    rows = []
    for (m, c, t, k), rs in sorted(by_key.items()):
        f1s = [float(r["macro_f1"]) for r in rs]
        rows.append({"model": m, "condition": c, "task": t, "k": k, "seeds": len(rs),
                     "macro_precision": round(statistics.mean(float(r["macro_precision"]) for r in rs), 4),
                     "macro_recall": round(statistics.mean(float(r["macro_recall"]) for r in rs), 4),
                     "macro_f1": round(statistics.mean(f1s), 4),
                     "f1_lo": round(min(f1s), 4), "f1_hi": round(max(f1s), 4)})
    with (CELLS / "metrics_judge.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (CELLS / "per_document_judge.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_doc[0].keys())); w.writeheader(); w.writerows(per_doc)
    print(f"[merge] {len(rows)} metric rows, {len(per_doc)} per-document rows -> {CELLS}")
    return {(r["condition"], r["task"]): r for r in rows if r["k"] == "all"}


def published_judge():
    """(model, condition, task) -> (f1, lo, hi) for the four reported models and the floor."""
    out = {}
    for r in read_csv(Z / "t1_headline_metrics.csv"):
        if r["k"] == "all":
            out[(r["model"], "zero_shot", r["task"])] = (float(r["F1"]), float(r["F1_min"]), float(r["F1_max"]))
    for r in read_csv(Z / "t6_zs_vs_cot_paired.csv"):
        out[(r["model"], "cot", r["task"])] = (float(r["cot_f1"]), None, None)
    ds = defaultdict(list)
    for r in read_csv(DJ_DEEPSEEK):
        if r["model"] == "deepseek-r1-distill-qwen-14b":
            ds[(r["condition"], r["task"])].append(float(r["f1"]))
    for (c, t), v in ds.items():
        out[("deepseek-r1-distill-qwen-14b", c, t)] = (sum(v) / len(v), None, None)
    return out


def bge_and_fre():
    bge = {}
    for t in TASKS:
        for sub in ("bge", "bge_controls"):
            p = TRACK_B / sub / f"metrics_task1_{t}.csv"
            for r in read_csv(p):
                bge[(r["model"], r["condition"], t)] = float(r["macro_f1"])
    fre = {(r["model"], r["condition"], r["task"]): (float(r["mean_fre"]), float(r["mean_words"]))
           for r in read_csv(TRACK_B / "readability_fre.csv")}
    return bge, fre


def main():
    new = merge_cells()
    pub = published_judge()
    bge, fre = bge_and_fre()
    rows = []
    for m, label in LABELS.items():
        for c in CONDS:
            for t in TASKS:
                if m == NEW_MODEL:
                    r = new[(c, t)]
                    jf, lo, hi = float(r["macro_f1"]), float(r["f1_lo"]), float(r["f1_hi"])
                else:
                    jf, lo, hi = pub.get((m, c, t), (None, None, None))
                fv, wv = fre.get((m, c, t), (None, None))
                rows.append({"model": label, "model_id": m, "condition": c, "task": t,
                             "judge_f1": None if jf is None else round(jf, 3),
                             "judge_f1_min": None if lo is None else round(lo, 3),
                             "judge_f1_max": None if hi is None else round(hi, 3),
                             "bge_f1": round(bge.get((m, c, t), float("nan")), 3),
                             "flesch": fv, "mean_words": wv})
    for t in TASKS:
        jf = pub[("floor_crossdoc_gold", "zero_shot", t)]
        rows.append({"model": "cross-document floor", "model_id": "floor_crossdoc_gold", "condition": "-", "task": t,
                     "judge_f1": round(jf[0], 3), "judge_f1_min": round(jf[1], 3), "judge_f1_max": round(jf[2], 3),
                     "bge_f1": round(bge[("floor_crossdoc_gold", "floor_crossdoc_gold", t)], 3),
                     "flesch": None, "mean_words": None})
        fv, wv = fre[("gold", "agency", t)]
        rows.append({"model": "agency gold spans", "model_id": "gold", "condition": "-", "task": t,
                     "judge_f1": None, "judge_f1_min": None, "judge_f1_max": None, "bge_f1": None,
                     "flesch": fv, "mean_words": wv})
    out = TRACK_B / "summary_table.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[table] -> {out}")
    print(f"{'model':22s} {'cond':10s} {'task':10s} {'judgeF1':>8s} {'(min-max)':>14s} {'bgeF1':>6s} {'FRE':>6s} {'words':>6s}")
    for r in rows:
        rng = "" if r["judge_f1_min"] is None else f"({r['judge_f1_min']:.3f}-{r['judge_f1_max']:.3f})"
        jf = "" if r["judge_f1"] is None else f"{r['judge_f1']:.3f}"
        bf = "" if r["bge_f1"] is None or r["bge_f1"] != r["bge_f1"] else f"{r['bge_f1']:.3f}"
        fv = "" if r["flesch"] is None else f"{r['flesch']:.1f}"
        wv = "" if r["mean_words"] is None else f"{r['mean_words']:.1f}"
        print(f"{r['model']:22s} {r['condition']:10s} {r['task']:10s} {jf:>8s} {rng:>14s} {bf:>6s} {fv:>6s} {wv:>6s}")


if __name__ == "__main__":
    main()
