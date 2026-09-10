#!/usr/bin/env python3
"""Plan task 1, judge-based scoring with Precision@k and Recall@k. No wording match, meaning match.

For each document and task, the judge is given the document's own gold risks (or mitigations) and a
condition's predictions, and aligns them one to one. A predicted item matches a gold item only if it
names the same underlying privacy problem, not merely because both are privacy statements. The
predictions are ranked (the ranked up-to-K prompt orders them), so we report at cut-offs k, a match
counts at cut-off k only if the matched prediction is in the top k. Precision@k, Recall@k, F1@k, plus
the overall no-cutoff view.

Runs the model conditions and the controls through the same judge. gpt-5.5 via the laptop API, one
cached call per (condition, seed, document, task), seeded. The cache stores the raw alignment, so
changing the cut-offs or the aggregation does not re-spend the API.

Usage:
  python3 judge_score_task1.py --limit 3 --seeds 1 --conditions zero_shot ceiling_goldecho   # smoke
  python3 judge_score_task1.py --seeds 1 2 3 --ks 3 5 10 \
      --conditions zero_shot cot floor_crossdoc_gold floor_countmatched ceiling_goldecho
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
TB = ROOT / "2026-06-30_meeting_followup/task1_build"
CACHE = TB / "judge_cache"
OUT = TB / "scoring_judge"
MODEL = "gpt-5.5"

_spec = importlib.util.spec_from_file_location("st", TB / "score_task1.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

DEFAULT_PREDS = [
    str(ROOT / "results/task1_deepseek_gala1_20260702T122022Z/all_predictions.jsonl"),
    str(TB / "run_pack/audit_floors.jsonl"),
]

SYSTEM = (
    "You judge whether predicted privacy items refer to the SAME underlying issue as the reference "
    "items for one system. Two items match ONLY if they concern the same privacy problem about the "
    "same kind of data or data flow. Do NOT match two items just because both are privacy risks, "
    "both mention security, or both use similar words. Be strict about specificity. When in doubt, "
    "do not match."
)


def boot_ci(vals, n=1000, seed=20260702):
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(vals)
    means = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def prompt(task, gold, pred):
    kind = "RISKS" if task == "risk" else "MITIGATIONS"
    g = "\n".join(f"{i+1}. {t}" for i, t in enumerate(gold))
    p = "\n".join(f"{i+1}. {t}" for i, t in enumerate(pred))
    return (
        f"System task, {kind.lower()} for one privacy assessment.\n\n"
        f"REFERENCE {kind} (the correct answers for this system):\n{g}\n\n"
        f"PREDICTED {kind} (to judge):\n{p}\n\n"
        "Align predicted items to reference items. A predicted item matches a reference item only if "
        "it names the same specific issue. Use each reference and each predicted item at most once, "
        "choose the best one-to-one pairing.\n"
        'Return ONLY JSON: {"matches": [{"predicted": <int>, "reference": <int>}, ...]}. '
        "If nothing matches, return an empty list."
    )


def unique_pairs(raw, npred, ngold):
    """Return the one-to-one (predicted_index, reference_index) pairs, 1-indexed, validated."""
    used_p, used_g, pairs = set(), set(), []
    for m in raw:
        try:
            pi, gi = int(m["predicted"]), int(m["reference"])
        except (KeyError, ValueError, TypeError):
            continue
        if 1 <= pi <= npred and 1 <= gi <= ngold and pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); pairs.append((pi, gi))
    return pairs


def judge(client, cond, seed, doc, task, gold, pred):
    if not gold:
        return None
    if not pred:
        return {"pairs": [], "pred_n": 0, "gold_n": len(gold)}
    h = hashlib.sha256(("|".join([cond, str(seed), doc, task] + gold + pred)).encode()).hexdigest()[:16]
    cp = CACHE / f"{cond}_{seed}_{doc}_{task}_{h}.json"
    if cp.exists():
        raw = json.loads(cp.read_text()).get("matches", [])
    else:
        s = int(h, 16) % 100000
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt(task, gold, pred)}],
            seed=s, response_format={"type": "json_object"})
        obj = json.loads(resp.choices[0].message.content)
        cp.write_text(json.dumps(obj, indent=2))
        raw = obj.get("matches", [])
    return {"pairs": unique_pairs(raw, len(pred), len(gold)), "pred_n": len(pred), "gold_n": len(gold)}


def prf_at_k(pairs, pred_n, gold_n, k):
    """k is an int cut-off or None for no cut-off."""
    if k is None:
        m = len(pairs); denom_p = pred_n
    else:
        m = sum(1 for pi, _ in pairs if pi <= k); denom_p = min(k, pred_n)
    p = m / denom_p if denom_p else 0.0
    r = m / gold_n if gold_n else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", default=DEFAULT_PREDS)
    ap.add_argument("--conditions", nargs="+",
                    default=["zero_shot", "cot", "floor_crossdoc_gold", "floor_countmatched", "ceiling_goldecho"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir for the two CSVs, default scoring_judge/")
    ap.add_argument("--tasks", nargs="+", choices=["risk", "mitigation"], default=["risk", "mitigation"],
                    help="restrict to one task so (condition, seed, task) cells can run as parallel processes")
    args = ap.parse_args()
    ks = list(args.ks) + [None]        # None is the overall no-cutoff view

    CACHE.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key, max_retries=6)  # parallel cells share one rate limit

    gold = st.load_gold()
    docs = list(gold)[:args.limit] if args.limit else list(gold)
    preds = {}
    for p in args.preds:
        for k, dm in st.load_preds(p).items():
            preds.setdefault(k, {}).update(dm)
    keyed = collections.defaultdict(lambda: collections.defaultdict(dict))   # (model, cond) -> seed -> dm
    for (model, cond, seed), dm in preds.items():
        keyed[(model, cond)][seed] = dm
    wanted = set(args.conditions) if args.conditions else None

    grid_rows, per_doc = [], []
    for (model, cond) in sorted(keyed):
        if wanted and cond not in wanted:
            continue
        seeds_dm = keyed[(model, cond)]
        for task in args.tasks:
            seed_stats = {k: [] for k in ks}
            for seed in args.seeds:
                dm = seeds_dm.get(seed) or seeds_dm.get(1)
                if dm is None:
                    continue
                doc_vals = {k: [] for k in ks}
                for doc in docs:
                    # cache is keyed on the original condition, so existing calls are reused
                    res = judge(client, cond, seed, doc, task, gold[doc][task],
                                dm.get(doc, {}).get(task, []))
                    if res is None:
                        continue
                    for k in ks:
                        p, r, f = prf_at_k(res["pairs"], res["pred_n"], res["gold_n"], k)
                        doc_vals[k].append((p, r, f))
                        if k is None:
                            per_doc.append({"model": model, "condition": cond, "seed": seed, "doc_id": doc,
                                            "task": task, "matches": len(res["pairs"]), "pred_n": res["pred_n"],
                                            "gold_n": res["gold_n"], "f1": round(f, 4)})
                for k in ks:
                    if doc_vals[k]:
                        v = doc_vals[k]
                        seed_stats[k].append((sum(x[0] for x in v) / len(v),
                                              sum(x[1] for x in v) / len(v),
                                              sum(x[2] for x in v) / len(v),
                                              [x[2] for x in v]))
            for k in ks:
                s = seed_stats[k]
                if not s:
                    continue
                mp = sum(x[0] for x in s) / len(s)
                mr = sum(x[1] for x in s) / len(s)
                mf = sum(x[2] for x in s) / len(s)
                ci = (min(x[2] for x in s), max(x[2] for x in s)) if len(s) > 1 else boot_ci(s[0][3])
                grid_rows.append({"model": model, "condition": cond, "task": task,
                                  "k": ("all" if k is None else k), "seeds": len(s),
                                  "macro_precision": round(mp, 4), "macro_recall": round(mr, 4),
                                  "macro_f1": round(mf, 4), "f1_lo": round(ci[0], 4), "f1_hi": round(ci[1], 4)})

    with (out / "metrics_judge.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "condition", "task", "k", "seeds", "macro_precision",
                                           "macro_recall", "macro_f1", "f1_lo", "f1_hi"])
        w.writeheader(); w.writerows(grid_rows)
    with (out / "per_document_judge.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "condition", "seed", "doc_id", "task", "matches", "pred_n", "gold_n", "f1"])
        w.writeheader(); w.writerows(per_doc)

    labels = []
    for r in grid_rows:
        lab = (r["model"], r["condition"])
        if lab not in labels:
            labels.append(lab)
    print(f"[judge] {len(docs)} docs, seeds {args.seeds}, ks {args.ks} + all")
    for task in ("risk", "mitigation"):
        print(f"\n== {task.upper()} macro F1 by model/condition and k ==")
        print(f"  {'model/condition':30s}" + "".join(f"{('@'+str(k) if k!='all' else 'all'):>8s}" for k in [*args.ks, 'all']))
        for (model, cond) in labels:
            name = cond if model == cond else f"{model.split('/')[-1][:16]}/{cond}"
            cells = {r["k"]: r["macro_f1"] for r in grid_rows if r["model"] == model and r["condition"] == cond and r["task"] == task}
            if not cells:
                continue
            print(f"  {name:30s}" + "".join(f"{cells.get(k, float('nan')):8.3f}" for k in [*args.ks, 'all']))


if __name__ == "__main__":
    main()
