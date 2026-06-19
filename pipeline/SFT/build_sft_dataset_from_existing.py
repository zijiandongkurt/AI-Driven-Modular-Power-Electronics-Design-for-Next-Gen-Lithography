"""
build_sft_dataset_from_existing.py
==================================

Mines SFT training pairs from any `pipeline/data/batch_*/` folders that have
both a `reward_results.json` and a corresponding `validation_results.json`.

No LLM call, no simulator call.  Pure file-mining.

Output: a JSONL file where each line is:
    {
      "prompt":      <full prompt string built via make_prompt(constraint)>,
      "completion":  <raw SPICE netlist text from topN.net>,
      "fitness":     <float>,
      "source":      "<batch_id>/<top_id>"
    }

Filters:
    - validation_results.json must mark this topology as `passed: True`
    - reward_results.json must give a fitness_score >= --min-fitness
    - top-K per batch by fitness (default K=2)

CLI:
    python scripts/build_sft_dataset_from_existing.py \
        --root pipeline/data \
        --out  data/sft/sft_from_existing.jsonl \
        --top-k 2 \
        --min-fitness -300

Run with --root pointing at another worktree (e.g. a checkout of `main`)
to mine batches that aren't on the current branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_topology_generation.prompt_input import make_prompt


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  WARN: failed to parse {p}: {e}", file=sys.stderr)
        return None


def _find_netlist(batch_dir: Path, top_id: str) -> Path | None:
    """The project uses both 'LLM_output' (main) and 'llm_output' (LLM_Syh)."""
    for case in ("LLM_output", "llm_output"):
        p = batch_dir / case / f"{top_id}.net"
        if p.exists():
            return p
    return None


def mine_batch(batch_dir: Path, top_k: int, min_fitness: float) -> list[dict]:
    """Mine a single batch directory for valid (prompt, completion) pairs.

    Args:
        batch_dir (Path): Path to a ``batch_*/`` directory containing
            ``reward_results.json`` and ``validation_results.json``.
        top_k (int): Maximum number of top-fitness pairs to return.
        min_fitness (float): Minimum fitness score; candidates below this
            threshold are discarded.

    Returns:
        list[dict]: At most ``top_k`` dicts with keys ``"prompt"``,
            ``"completion"``, ``"fitness"``, ``"source"``, and
            ``"constraint"``, sorted by descending fitness.
    """
    rew = _read_json(batch_dir / "reward_results.json")
    val = _read_json(batch_dir / "validation_results.json")

    if not rew or "circuits" not in rew:
        return []
    if not val:
        return []

    constraint = rew.get("active_constraints", {})
    if not constraint:
        print(f"  WARN: {batch_dir.name} has no active_constraints", file=sys.stderr)
        return []
    prompt = make_prompt(constraint)

    # Score each circuit and keep only those that passed validation + min_fitness
    candidates = []
    for top_id, info in rew["circuits"].items():
        fitness = info.get("fitness_score")
        if fitness is None:
            continue
        if fitness < min_fitness:
            continue

        # validation gate
        vinfo = val.get(top_id) or val.get(f"{top_id}.net") or {}
        if not vinfo.get("passed"):
            continue

        net_path = _find_netlist(batch_dir, top_id)
        if not net_path:
            continue

        completion = net_path.read_text(encoding="utf-8")
        candidates.append({
            "prompt":     prompt,
            "completion": completion,
            "fitness":    float(fitness),
            "source":     f"{batch_dir.name}/{top_id}",
            "constraint": {k: v for k, v in constraint.items() if not k.startswith("_")},
        })

    # Keep top-k by fitness
    candidates.sort(key=lambda r: r["fitness"], reverse=True)
    return candidates[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "pipeline" / "data",
                    help="Root of batch_*/ folders to scan")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "sft" / "sft_from_existing.jsonl",
                    help="Output JSONL path")
    ap.add_argument("--top-k", type=int, default=2,
                    help="How many top-fitness samples per batch to keep")
    ap.add_argument("--min-fitness", type=float, default=-1_000_000.0,
                    help="Reject samples below this fitness (default: accept all)")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Only scan these batch names (default: all batch_*)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"FATAL: --root {args.root} does not exist", file=sys.stderr)
        sys.exit(1)

    batches = sorted(args.root.glob("batch_*"))
    if args.include:
        batches = [b for b in batches if b.name in set(args.include)]
    print(f"[scan] {len(batches)} batch folder(s) under {args.root}")

    all_pairs: list[dict] = []
    per_batch_kept = {}
    for batch in batches:
        if not batch.is_dir():
            continue
        kept = mine_batch(batch, args.top_k, args.min_fitness)
        per_batch_kept[batch.name] = len(kept)
        all_pairs.extend(kept)
        print(f"  {batch.name:<25} kept {len(kept)} sample(s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"[done] {len(all_pairs)} SFT pair(s) written to {args.out}")
    if all_pairs:
        f_min = min(r["fitness"] for r in all_pairs)
        f_max = max(r["fitness"] for r in all_pairs)
        print(f"[done] fitness range: [{f_min:.2f}, {f_max:.2f}]")
        # Show a teaser
        teaser = max(all_pairs, key=lambda r: r["fitness"])
        print(f"[done] best sample: {teaser['source']} fitness={teaser['fitness']:.2f}")


if __name__ == "__main__":
    main()
