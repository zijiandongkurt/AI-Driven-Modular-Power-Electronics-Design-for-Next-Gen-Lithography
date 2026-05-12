# `hpc_configs/` — HPC deployment kit

Drop-in SLURM scripts + override files for running GRPO training on
Snellius (or any SLURM cluster with similar partitions).

```
hpc_configs/
├── README.md                              ← this file (general overview)
├── TUNING_ON_40GB.md                      ← detailed playbook for the 40GB case
├── qwen_io_demo.py                        ← standalone Qwen call (no GRPO/sim) ⭐
├── slurm/
│   ├── smoke_qwen.slurm                   ← runs qwen_io_demo on HPC ⭐
│   ├── train_single_80gb.slurm            ← Scenario A: 1× 80GB GPU
│   ├── train_quad_40gb.slurm              ← Scenario B: 4× 40GB GPU
│   └── train_single_40gb.slurm            ← Scenario C: 1× 40GB GPU (most common!)
└── overrides/
    ├── new_rl_updater.py                  ← replaces pipeline/reinforcement_algorithm/new_rl_updater.py
    ├── grpo_trainer.py                    ← replaces pipeline/reinforcement_algorithm/grpo_trainer.py
    └── rl_demo.py                         ← replaces pipeline/reinforcement_algorithm/rl_demo.py
                                              (reads SLURM-side env vars, see §7)
```

⭐ **Always run `smoke_qwen.slurm` FIRST on a new node.** If Qwen can't
load and emit text there, GRPO has zero chance — saves you debugging
the wrong layer.

---

## 1. Which scenario should I pick?

|                              | **A — single 80GB** | **B — 4× 40GB**             | **C — single 40GB**          |
|------------------------------|---------------------|-----------------------------|-------------------------------|
| GPUs                         | 1                   | 4                           | 1                             |
| VRAM per GPU                 | 80 GB               | 40 GB                       | 40 GB                         |
| Snellius partition           | `gpu_a100` + constraint=a100_80gb (or `gpu_h100`) | `gpu_a100` (default 40GB) | `gpu_a100` (default 40GB)     |
| Qwen3-14B placement          | one card, bf16      | sharded across 4 cards bf16 | one card, **4-bit NF4**       |
| Quantization required?       | no                  | no (optional)               | **yes**                       |
| Forward pass speed           | fast                | ~30-50% slower (NCCL hops)  | fast                          |
| GPU-hour cost per step       | 1×                  | 4×                          | 1×                            |
| Easier to schedule           | hard — 80GB rare    | medium                      | easy — 40GB is the default    |
| Setup complexity             | simplest            | needs sharding              | needs quantization knob       |

**Rule of thumb**: try A → B → C in order of preference. If you only got
one 40GB card (the most common case), go straight to **Scenario C** and
see [`TUNING_ON_40GB.md`](TUNING_ON_40GB.md) for a step-by-step playbook.

---

## 2. Pre-requisites (do these once)

```bash
ssh snellius              # or your HPC login
git clone <repo-url>
cd AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
bash env.sh               # builds .venv with all deps
```

Verify Qwen3-14B is in the shared HF cache:
```bash
ls /projects/2/managed_datasets/hf_cache_dir/hub/ | grep -i qwen
# expected: models--Qwen--Qwen3-14B
```
If it's not there, see "Custom HF cache" below.

---

## 3. Apply the overrides (1 minute)

The override files are **drop-in replacements** for the corresponding
files in `pipeline/reinforcement_algorithm/`. Two ways to install them:

### Option A — direct copy (recommended)
```bash
cp hpc_configs/overrides/new_rl_updater.py pipeline/reinforcement_algorithm/new_rl_updater.py
cp hpc_configs/overrides/grpo_trainer.py   pipeline/reinforcement_algorithm/grpo_trainer.py
cp hpc_configs/overrides/rl_demo.py        pipeline/reinforcement_algorithm/rl_demo.py
```

### Option B — symlink (Linux only, safer if you'll re-pull)
```bash
for f in new_rl_updater.py grpo_trainer.py rl_demo.py; do
  ln -sf "$PWD/hpc_configs/overrides/$f" "pipeline/reinforcement_algorithm/$f"
done
```

### Revert at any time
```bash
git checkout pipeline/reinforcement_algorithm/new_rl_updater.py \
              pipeline/reinforcement_algorithm/grpo_trainer.py \
              pipeline/reinforcement_algorithm/rl_demo.py
```

---

## 4. What the overrides actually change

### `new_rl_updater.py` — `RLConfig` defaults

| Field                   | Original | Override |
|-------------------------|----------|----------|
| `max_length`            | 512      | **1024** |
| `max_prompt_length`     | 256      | **768**  |
| `max_completion_length` | 256      | **512**  |

**Why**: The original 256-token prompt cap was silently truncating ~60% of
our `system_prompt.txt + constraint` (which runs ~600–650 tokens). The
truncated prompt is what `_completion_log_prob` sees during training,
so the policy gradient was being computed against a different prompt
than the one used at inference. Aligning them costs ~12 GB of activation
memory, which both scenarios can absorb.

Algorithm is unchanged — same forward / log-prob / policy-loss math.

### `grpo_trainer.py` — three fixes

1. **Removed the `[:2]` OOM workaround** in `train_from_existing_batch`.
   On HPC we have the VRAM for the full 4-sample batch, which is what
   GRPO's group-relative reward normalization needs to estimate variance
   reliably.
2. **Absolute path resolution** for `system_prompt.txt` (uses
   `_REPO_ROOT`), so the trainer works regardless of what cwd SLURM
   hands it.
3. **Loud per-stage logging** (`[stage] N/5  generate`, etc.). Useful
   because SLURM stdout is only readable post-mortem in `logs/`.

No changes to the algorithm itself.

---

## 4.5 Sanity check: Qwen-only smoke test (do this FIRST)

Before booking GPU hours on the real GRPO job, verify Qwen can actually
load and generate on the node you got:

```bash
sbatch hpc_configs/slurm/smoke_qwen.slurm
```

This takes ~3-5 minutes (15s load + 30s generate × 2 candidates) and
exercises:

- module loads + venv activation
- HuggingFace cache lookup (offline mode)
- Quantization auto-toggle (4-bit if SLURM gave you < 50 GB VRAM)
- Real `model.generate()` call via `TopologyLLM`

Look in `logs/qwen_smoke_<jobid>.out` for two candidate netlists. If you
see them, the GPU + model + prompt pipeline is healthy and you can
submit the real `train_*` SLURM with confidence.

You can also run it interactively on the GPU node if you grabbed one
via `srun --pty bash`:

```bash
export GRPO_QUANTIZATION="4bit"     # if on 40GB
python hpc_configs/qwen_io_demo.py
```

## 5. Submit a job

Both scripts have a couple of placeholders you must fill in before
submitting — search for `← replace` comments at the top:

```bash
#SBATCH --account=scur2545        # ← your SLURM account / budget name
#SBATCH --mail-user=<your email>  # ← optional, uncomment to get notified
```

Then:

```bash
# Scenario A — 1× 80GB
sbatch hpc_configs/slurm/train_single_80gb.slurm

# Scenario B — 4× 40GB
sbatch hpc_configs/slurm/train_quad_40gb.slurm

# Scenario C — 1× 40GB  (most common; see TUNING_ON_40GB.md for the playbook)
sbatch hpc_configs/slurm/train_single_40gb.slurm
```

Track it:

```bash
squeue -u $USER
# once it starts:
tail -F logs/grpo_single80_<jobid>.out
```

---

## 6. What both scripts do at runtime

```text
1. mkdir logs/
2. module purge && load 2023 + Python 3.11.3 + CUDA 12.1 + cuDNN
3. cd <repo> && source .venv/bin/activate
4. nvidia-smi  (sanity check)
5. export HF_CACHE_DIR + HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1
6. export GRPO_MODEL_ID, GRPO_QUANTIZATION  (knobs for rl_demo.py)
7. python -u pipeline/reinforcement_algorithm/rl_demo.py
```

`rl_demo.py` currently calls `grpo.train_from_existing_batch("batch_1")`
which means the SLURM job will **only** run the RL-update step on
existing reward data, not the full generate→validate→simulate→reward
loop. Once `NGSpiceSimulator` is verified working on the HPC, flip the
commented lines in `rl_demo.py` to call `grpo.train(...)` instead.

---

## 7. Customising the model / quantization (env-var knobs)

The `overrides/rl_demo.py` we ship reads these SLURM-side env vars so
you can tune without editing python:

| Env var              | Default          | Meaning                                                  |
|----------------------|------------------|----------------------------------------------------------|
| `GRPO_MODEL_ID`      | `Qwen/Qwen3-14B` | HuggingFace model id (or local path)                     |
| `GRPO_QUANTIZATION`  | `""` (= bf16)    | `"4bit"` → NF4. **Required on Scenario C**.              |
| `GRPO_BATCH_ID`      | `batch_1`        | Which `pipeline/data/<batch>/` to use                    |
| `GRPO_N_SAMPLES`     | _unset_          | Cap samples after `_build_training_batch` (OOM escape)   |
| `GRPO_MAX_LENGTH`    | _unset_          | Override `RLConfig.max_length` (and shrink prompt+comp)  |
| `GRPO_FULL_TRAIN`    | `"0"`            | `"1"` → run `train()` (full loop) instead of existing-batch |

Example (Scenario C with extra-tight memory):
```bash
export GRPO_QUANTIZATION="4bit"
export GRPO_MAX_LENGTH=768
export GRPO_N_SAMPLES=2
```

Each SLURM file in `slurm/` already exports the sensible defaults for
its scenario — these are just the further knobs you can toggle.

---

## 8. First-run smoke test (recommended)

Before booking 2-hour jobs:

```bash
# Cut walltime to 30 minutes and submit
sed -i 's/--time=02:00:00/--time=00:30:00/' hpc_configs/slurm/train_single_80gb.slurm
sbatch hpc_configs/slurm/train_single_80gb.slurm
```

In the output log you want to see:

| Line                                                                  | Means                                      |
|-----------------------------------------------------------------------|--------------------------------------------|
| `Loading weights:` reaching 100%                                      | model loaded from HF cache OK              |
| `trainable params: XX,XXX,XXX \|\| all params: 14,XXX,XXX,XXX \|\| trainable%: 0.1X` | LoRA was applied — only ~0.1% trainable    |
| `RL samples: 4`                                                       | full batch (not the old `[:2]` hack)       |
| `Saved RL LoRA adapter to: ./checkpoints/grpo-lora/final`             | LoRA checkpoint written                    |
| `Saved metrics to .../grpo_metrics.json`                              | training metrics persisted                 |
| `DONE — exit code 0`                                                  | clean exit                                 |

If you see CUDA OOM:
1. (Scenario B only) set `GRPO_QUANTIZATION="4bit"` in the SLURM file
2. Reduce `max_length: 1024 → 512` in `hpc_configs/overrides/new_rl_updater.py`
3. As a last resort, re-introduce a sample cap

---

## 9. Common gotchas

| Symptom                                                | Likely cause                                                       | Fix                                                                   |
|--------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------|
| `srun: error: No compute budget for the partition`     | wrong `--partition` or no `--account`                              | check `accinfo` / `sacctmgr show association user=$USER`              |
| `local_files_only=True, no file found`                 | `HF_CACHE_DIR` doesn't actually contain Qwen3-14B                  | `ls $HF_CACHE_DIR/hub/` — must see `models--Qwen--Qwen3-14B`          |
| Loads model but immediately CUDA OOM                   | got a 40GB card without enabling sharding/quantization             | use `train_quad_40gb.slurm` (4×40GB) or add `GRPO_QUANTIZATION="4bit"` |
| `.venv not found — did you run env.sh?`                | the venv path doesn't match what env.sh built                      | env.sh uses `.venv` (with leading dot); scripts match that.           |
| Modules mismatch causes `ImportError: libtorch...`     | SLURM loaded `2024` stack but venv was built under `2023`          | edit `module load 2023` in both env.sh and SLURM to be identical      |
| Long ngspice failures or "Simulator executable not found" | ngspice not loaded                                              | add `module load ngspice` (check exact name via `module spider ngspice`); for now stick with `train_from_existing_batch` |

---

## 10. Custom HF cache (no `/projects/2/...` access)

If you can't read the shared Snellius cache, point `HF_CACHE_DIR` at your
own scratch directory and download the model once:

```bash
# In an interactive CPU job (don't do this on int5)
srun -p rome -A scur2545 --time=02:00:00 --pty bash
source .venv/bin/activate
mkdir -p /scratch/$USER/hf_cache
export HF_CACHE_DIR=/scratch/$USER/hf_cache
huggingface-cli download Qwen/Qwen3-14B --cache-dir "$HF_CACHE_DIR"
```

Then change the two SLURM scripts:
```bash
export HF_CACHE_DIR=/scratch/$USER/hf_cache    # ← your path
```

---

## 11. TL;DR

```bash
# one-time setup
bash env.sh

# install overrides (3 files now: rl_demo.py reads SLURM env vars)
cp hpc_configs/overrides/*.py pipeline/reinforcement_algorithm/

# pick your scenario, edit --account in the SLURM file, submit
sbatch hpc_configs/slurm/train_single_80gb.slurm   # 1× 80GB
sbatch hpc_configs/slurm/train_quad_40gb.slurm     # 4× 40GB
sbatch hpc_configs/slurm/train_single_40gb.slurm   # 1× 40GB ← most common

# watch
squeue -u $USER
tail -F logs/grpo_*_<jobid>.out
```

**For the single-40GB case** (which is what you'll usually get):
- read [`TUNING_ON_40GB.md`](TUNING_ON_40GB.md) for the memory ladder,
  smoke test, and what to sweep first
- the SLURM script already sets `GRPO_QUANTIZATION="4bit"` for you
