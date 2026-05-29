# `data/sft/` — Supervised Fine-Tuning datasets

Files here are training pairs for the **SFT stage** of our two-stage
recipe (`SFT → GRPO`).  Each line of every `*.jsonl` is one example:

```json
{
  "prompt":     "<full prompt the model sees (NAMING_RULES + Constraint + '### SPICE Netlist:')>",
  "completion": "<the gold netlist text we want the model to learn to produce>",
  "fitness":    -0.18,                             // null if not yet scored
  "source":     "batch_1/top4",
  "constraint": {"vin_min": 12, "vout_target": 5, ...}
}
```

## Files in this folder — which one do I use?

### ⭐ Files actually consumed by training

| File | Use it for | Why |
|---|---|---|
| **`sft_train.jsonl`** (81 samples) | **Training** — `sft_trainer.py` reads this | Deduped, shuffled, covers all 13 constraints. **This is the only file the SFT trainer touches by default.** |
| **`sft_val.jsonl`** (9 samples) | **Validation** — `sft_trainer.py` computes mean CE loss on this every epoch | Held-out 10% split, never trained on. Used to detect over-fitting. |

### Intermediate / raw / archival (not for training directly)

| File | What it is | Don't pass to training because… |
|---|---|---|
| `sft_from_existing.jsonl` (6 pairs) | Raw mined samples from main's batch_1, 2, 4 and LLM_Syh's batch_2 (have real `fitness`) | Tiny; un-shuffled; intentionally a *subset* of `sft_train.jsonl`. Use it only as a **fewshot pool** for `generate_sft_candidates.py`. |
| `sft_from_generation.jsonl` (33 pairs) | Coder-2.5 generated for constraints **0-4** | Un-shuffled, un-deduped against the existing file. Already folded into `sft_train.jsonl`. |
| `sft_from_generation_pt2.jsonl` (53 pairs) | Coder-2.5 generated for constraints **5-12** | Same as above for the rest of the constraints. |
| `generation.log` / `generation_pt2.log` | Full stdout of the two `generate_sft_candidates.py` runs | Diagnostic record. Useful for "why did constraint X have 5/8 pass rate?" |

**Rule of thumb**: if you want to *train*, point at `sft_train.jsonl`. Every
other file in this folder is either an input that produced it (`sft_from_*`),
a holdout it was split from (`sft_val.jsonl`), or a log from when it was built.

### Rebuild flow at a glance

```
                  sft_from_existing.jsonl (6)
                  sft_from_generation.jsonl (33)
                  sft_from_generation_pt2.jsonl (53)
                                │
                                ▼
                  scripts/merge_sft_jsonl.py
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
            sft_train.jsonl (81)       sft_val.jsonl (9)
                  │                           │
                  ▼                           ▼
        pipeline/.../sft_trainer.py  ─consumed every epoch─→
              trainer.train(train_path, val_path)
```

## How to train (on HPC)

```bash
# Recommended: submit the SLURM script
sbatch hpc_configs/slurm/train_sft.slurm

# Or run inline once you have an H100 80GB allocated:
module purge && module load 2023 Python/3.11.3-GCCcore-12.3.0 \
                                 CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
source .venv/bin/activate
export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_sft.py
```

Output: `checkpoints/sft-lora/{epoch-N/, final/, history.json}`.
After it finishes, GRPO can pick up from `checkpoints/sft-lora/final/`:
```python
llm = TopologyLLM()
llm.engine.load_adapter("sft", "./checkpoints/sft-lora/final")
llm.engine.set_active("sft")
GRPOTrainer(llm=llm, ...).train(...)
```

## How to rebuild the data (if you want different splits / more samples)

```bash
# 1. Mine the existing reward-scored batches (no model needed)
python scripts/build_sft_dataset_from_existing.py --top-k 2 --min-fitness -1000
# ↳ writes sft_from_existing.jsonl

# 2. Bootstrap more via Coder-2.5 + few-shot (Windows, 10–20 min on RTX laptop)
python scripts/generate_sft_candidates.py \
    --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 \
    --n-per-constraint 8 \
    --temperature 0.9
# ↳ uses sft_from_existing.jsonl as in-context fewshots
# ↳ writes sft_from_generation.jsonl  (validator-passing only, fitness=null)

# 3. Merge + split
python scripts/merge_sft_jsonl.py \
    --inputs data/sft/sft_from_existing.jsonl \
             data/sft/sft_from_generation.jsonl \
             data/sft/sft_from_generation_pt2.jsonl \
    --out-train data/sft/sft_train.jsonl \
    --out-val   data/sft/sft_val.jsonl \
    --val-frac  0.10 \
    --seed      42
# ↳ writes sft_train.jsonl + sft_val.jsonl (deduped, shuffled)
```

## How to score the generated batches on HPC (when ngspice/ltspice available)

Each generation creates a `pipeline/data/batch_sft_gen_idx<N>/` folder
in the project-standard layout (with `LLM_output/topN.net`,
`validation_results.json`).  On HPC where the simulator works:

```python
# pseudo-code, fits into the existing pipeline
from pipeline.simulation.ngspice_runner import NGSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm

for batch_id in [f"batch_sft_gen_idx{i}" for i in range(5)]:
    NGSpiceSimulator().simulate(batch_id)
    RewardFunctionNorm().process_batch(batch_id, constraint, weights={...})

# Then re-mine to fill in fitness scores:
python scripts/build_sft_dataset_from_existing.py \
    --root pipeline/data --include batch_sft_gen_idx0 batch_sft_gen_idx1 \
                                  batch_sft_gen_idx2 batch_sft_gen_idx3 \
                                  batch_sft_gen_idx4 \
    --out  data/sft/sft_from_generation_scored.jsonl
```

## Decisions made during data prep

1. **`top-k = 2` per batch**: keeps only the best 2 per constraint (the
   tail is often dominated by one outlier configuration). Bumps from 1
   to 2 give the model 2× the per-constraint variety without much
   diminishing returns.

2. **`min-fitness` left wide open (-1000)**: with 13 known constraints
   and only a few good batches, we'd rather take a `fitness=-228` buck
   that *almost* hits target than drop it. SFT loss is symmetric on
   token-level CE — it doesn't care if vout=4V vs 5V, just that the
   netlist is well-formed and matches the format. The fitness signal
   matters at GRPO stage.

3. **Few-shot during generation**: pure 0-shot prompting of Qwen2.5-Coder
   gave 0/4 validation pass-rate; 2-shot from the high-fitness batch_1
   samples jumped that to 3/4. The few-shot examples are sampled by
   **top fitness**, so they double as both "format teacher" and
   "quality bound" for the new generations.

4. **No simulator scoring on Windows**: PyLTSpice 1.5 has a parser
   incompatibility with ngspice ≥ 41 (we proved this in WSL — see
   `experiment_logs/ngspice_subprocess_test.py`), and LTspice isn't
   installed on this machine. So generated samples currently have
   `fitness: null` — score them on HPC if you want fitness-weighted
   SFT (Section above).

## Stats from the last build

Run `wc -l data/sft/*.jsonl` to see live counts.
The `generation.log` in this folder captures the per-constraint pass-rate.
