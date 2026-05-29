"""
merge_sft_jsonl.py
==================

Merges multiple sft_*.jsonl files into one training-ready dataset.

Handles:
  - de-duplication by completion text (drops byte-identical clones)
  - optional shuffle (seeded, reproducible)
  - optional train/val split
  - basic stats: per-source counts, completion-length histogram

CLI:
    python scripts/merge_sft_jsonl.py \
        --inputs data/sft/sft_from_existing.jsonl \
                 data/sft/sft_from_generation.jsonl \
        --out-train data/sft/sft_train.jsonl \
        --out-val   data/sft/sft_val.jsonl \
        --val-frac  0.10 \
        --seed      42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        print(f"  WARN: {p} does not exist; skipping", file=sys.stderr)
        return []
    rows = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            print(f"  WARN: parse error in {p}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-val", type=Path, default=None,
                    help="If given, write a held-out val split here")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len-chars", type=int, default=10_000,
                    help="Drop any (prompt+completion) longer than this — "
                         "guards against pathological tokenizer blowups")
    args = ap.parse_args()

    # ── Load + dedup ──────────────────────────────────────────────────
    all_rows: list[dict] = []
    src_counts = Counter()
    for inp in args.inputs:
        rows = _read_jsonl(inp)
        print(f"[load] {inp.name}: {len(rows)} rows")
        for r in rows:
            r["_origin_file"] = inp.name
        src_counts[inp.name] = len(rows)
        all_rows.extend(rows)

    print(f"[load] total before dedup: {len(all_rows)}")

    seen_completions: set[str] = set()
    deduped = []
    for r in all_rows:
        key = (r.get("completion") or "").strip()
        if not key:
            continue
        if key in seen_completions:
            continue
        seen_completions.add(key)
        # Length filter
        plen = len(r.get("prompt", "")) + len(key)
        if plen > args.max_len_chars:
            print(f"  WARN: dropping over-long sample "
                  f"{r.get('source','?')} ({plen} chars)")
            continue
        deduped.append(r)
    print(f"[dedup] unique completions: {len(deduped)} "
          f"(dropped {len(all_rows) - len(deduped)})")

    # ── Stats ─────────────────────────────────────────────────────────
    comp_lens = [len(r["completion"]) for r in deduped]
    if comp_lens:
        comp_lens.sort()
        n = len(comp_lens)
        print(f"\n[stats] completion length (chars):")
        print(f"  min={comp_lens[0]}, p25={comp_lens[n//4]}, "
              f"p50={comp_lens[n//2]}, p75={comp_lens[3*n//4]}, "
              f"max={comp_lens[-1]}")
        with_fitness = [r for r in deduped if r.get("fitness") is not None]
        without_fitness = len(deduped) - len(with_fitness)
        print(f"[stats] {len(with_fitness)} samples have fitness, "
              f"{without_fitness} are validator-only (no fitness)")

    # ── Shuffle + split ───────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(deduped)

    if args.out_val and args.val_frac > 0:
        n_val = max(1, int(len(deduped) * args.val_frac))
        val_rows = deduped[:n_val]
        train_rows = deduped[n_val:]
    else:
        val_rows = []
        train_rows = deduped

    # ── Write ─────────────────────────────────────────────────────────
    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w", encoding="utf-8") as f:
        for r in train_rows:
            r.pop("_origin_file", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if val_rows:
        with args.out_val.open("w", encoding="utf-8") as f:
            for r in val_rows:
                r.pop("_origin_file", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(train_rows)} train samples → {args.out_train}")
    if val_rows:
        print(f"[done] {len(val_rows)} val samples   → {args.out_val}")


if __name__ == "__main__":
    main()
