from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_PACK = ROOT / "2026-06-30_meeting_followup/task1_build/run_pack"
INPUTS = RUN_PACK / "benchmark_inputs.jsonl"
PROMPTS = RUN_PACK / "prompts"


def extract_json(text: str):
    """Return the last balanced JSON object in the text, after removing reasoning and fences."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    t = re.sub(r"^.*?</think>", " ", t, flags=re.S | re.I)   # stray unclosed reasoning
    t = re.sub(r"```(?:json)?", " ", t, flags=re.I)
    found = None
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        found = json.loads(t[start:i + 1])
                    except Exception:
                        found = None
                    break
        if found is not None:
            return found
        start = t.find("{", start + 1)
    return None


def coerce_items(items, keys):
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        row = {k: it.get(k, "" if k != "proposed" else False) for k in keys}
        if str(row.get("description", "")).strip():
            out.append(row)
    return out


def build_predicted(obj):
    if not isinstance(obj, dict):
        return None
    risks = coerce_items(obj.get("risks"), ["id", "description", "evidence_quote", "rationale"])
    mits = coerce_items(obj.get("mitigations"),
                        ["id", "addresses", "description", "evidence_quote", "proposed", "rationale"])
    if not isinstance(obj.get("risks"), list) and not isinstance(obj.get("mitigations"), list):
        return None
    return {"risks": risks, "mitigations": mits}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="deepseek-r1-distill-qwen-14b")
    ap.add_argument("--run-id", default="task1_deepseek_20260702")
    ap.add_argument("--conditions", nargs="+", default=["zero_shot", "cot"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 uses per-condition defaults")
    ap.add_argument("--prompt-set", choices=["plain", "ranked"], default="plain")
    ap.add_argument("--k", type=int, default=10, help="budget K, fills {K} in the ranked prompt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0, help="0-based, for splitting docs across parallel replicas")
    ap.add_argument("--num-shards", type=int, default=1, help="total replicas, docs sliced as docs[shard_index::num_shards]")
    ap.add_argument("--api-style", choices=["vllm", "openai", "gemini", "anthropic"], default="vllm",
                    help="vllm: local server, key EMPTY, max_tokens. openai: OPENAI_API_KEY, "
                         "max_completion_tokens, optional reasoning_effort. gemini/anthropic: their "
                         "OpenAI-compatible endpoints via --base-url and --api-key-env")
    ap.add_argument("--api-key-env", default=None, help="env var holding the API key for hosted styles")
    ap.add_argument("--reasoning-effort", default=None, help="openai style only, e.g. none or low")
    ap.add_argument("--max-retries", type=int, default=5, help="retries on rate limit / transient API errors")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing predictions file and skip completed model/condition/seed/doc keys")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "results" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"

    docs = [json.loads(l) for l in INPUTS.read_text().splitlines() if l.strip()]
    if args.limit:
        docs = docs[:args.limit]
    if args.num_shards > 1:
        docs = docs[args.shard_index::args.num_shards]
    system = (PROMPTS / "system.txt").read_text()
    prefix = "ranked_" if args.prompt_set == "ranked" else ""
    templates = {c: (PROMPTS / f"{prefix}{c}.txt").read_text() for c in args.conditions}
    max_tok = {"zero_shot": 6000, "cot": 8000}

    client = None
    if not args.dry_run:
        from openai import OpenAI
        key = "EMPTY"
        if args.api_style != "vllm":
            env = args.api_key_env or {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
                                       "anthropic": "ANTHROPIC_API_KEY"}[args.api_style]
            key = os.environ.get(env) or sys.exit(f"[infer] {env} is not set")
        client = OpenAI(base_url=args.base_url, api_key=key)

    existing_keys = set()
    existing_valid = 0
    if args.resume and preds_path.exists():
        for line_no, line in enumerate(preds_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("model") != args.model:
                    raise ValueError(f"model {rec.get('model')!r} != requested model {args.model!r}")
                key = (rec["condition"], int(rec["seed"]), rec["doc_id"])
            except Exception as exc:
                raise SystemExit(f"[infer] cannot resume {preds_path}, bad line {line_no}: {exc}") from exc
            if key in existing_keys:
                raise SystemExit(f"[infer] cannot resume {preds_path}, duplicate key {key}")
            existing_keys.add(key)
            existing_valid += rec.get("parse_status") == "valid"
        print(f"[infer] resume found {len(existing_keys)} existing records, {existing_valid} valid")

    written = 0
    valid = 0
    fh = preds_path.open("a" if args.resume else "w", encoding="utf-8")
    prev = out_dir / "prompt_preview.txt"
    t0 = time.time()
    for cond in args.conditions:
        for seed in args.seeds:
            for d in docs:
                record_key = (cond, seed, d["doc_id"])
                if record_key in existing_keys:
                    continue
                user = system + "\n\n" + templates[cond].replace("{SERVED_INPUT}", d["served_input"]).replace("{K}", str(args.k))
                if args.dry_run:
                    if written < 2:
                        prev.write_text(user[:4000])
                    rec = {"doc_id": d["doc_id"], "agency": d["agency"], "model": args.model,
                           "condition": cond, "seed": seed, "parse_status": "dry_run",
                           "predicted": {"risks": [], "mitigations": []}}
                    fh.write(json.dumps(rec) + "\n")
                    written += 1
                    continue
                mt = args.max_tokens or max_tok.get(cond, 6000)
                status, predicted, raw_len = "invalid", {"risks": [], "mitigations": []}, 0
                prompt_tokens = completion_tokens = 0
                reasoning_tokens = None
                kw = {"model": args.model, "messages": [{"role": "user", "content": user}],
                      "temperature": args.temperature, "seed": seed}
                if args.api_style != "anthropic":  # newer Claude models reject temperature with top_p
                    kw["top_p"] = args.top_p
                if args.api_style == "openai":
                    kw["max_completion_tokens"] = mt
                    if args.reasoning_effort:
                        kw["reasoning_effort"] = args.reasoning_effort
                else:
                    kw["max_tokens"] = mt
                t0_call = time.time()
                try:
                    for attempt in range(args.max_retries):
                        try:
                            resp = client.chat.completions.create(**kw)
                            break
                        except Exception as exc:
                            transient = type(exc).__name__ in ("RateLimitError", "APIConnectionError",
                                                               "APITimeoutError", "InternalServerError")
                            if not transient or attempt == args.max_retries - 1:
                                raise
                            time.sleep(min(60, 2 * 2 ** attempt))
                    content = resp.choices[0].message.content or ""
                    raw_len = len(content)
                    if resp.usage is not None:
                        prompt_tokens = resp.usage.prompt_tokens
                        completion_tokens = resp.usage.completion_tokens
                        details = getattr(resp.usage, "completion_tokens_details", None)
                        reasoning_tokens = getattr(details, "reasoning_tokens", None)
                    obj = extract_json(content)
                    p = build_predicted(obj)
                    if p is not None:
                        predicted, status = p, "valid"
                except Exception as exc:  # keep going, record the failure
                    status = f"error: {type(exc).__name__}"
                latency_s = round(time.time() - t0_call, 2)
                rec = {"doc_id": d["doc_id"], "agency": d["agency"], "model": args.model,
                       "condition": cond, "seed": seed, "parse_status": status,
                       "predicted": predicted, "raw_len": raw_len,
                       "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                       "reasoning_tokens": reasoning_tokens, "latency_s": latency_s}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                written += 1
                valid += (status == "valid")
            print(f"[infer] {cond} seed {seed} done, {len(existing_keys) + written} total records, "
                  f"{written} new, {time.time()-t0:.0f}s")
    fh.close()
    print(f"[infer] wrote {written} new records to {preds_path}, "
          f"total {len(existing_keys) + written}, valid {existing_valid + valid}"
          f"{' (dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
