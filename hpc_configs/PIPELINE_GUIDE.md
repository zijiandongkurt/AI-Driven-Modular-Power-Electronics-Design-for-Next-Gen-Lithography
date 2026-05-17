# Full Pipeline Operation Guide — Snellius HPC

End-to-end command-level instructions for the SFT → GRPO recipe.
**Interactive only — no `sbatch`.** All commands assume you already
have an `salloc`/`srun` GPU session (gcn*).

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0  ssh + GPU node                                         │
│  Stage 1  pull latest code (LLM_Syh)                             │
│  Stage 2  set env vars (one-time per shell)                      │
│  Stage 3  re-train SFT on clean data (~10 min)                   │
│  Stage 4  GRPO smoke (~30 s, no simulator)                       │
│  Stage 5  GRPO full loop (~5-30 min, needs LTspice container)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 0 — SSH + grab a GPU

```bash
# from your laptop
ssh snellius
# request a single H100 80GB for 4 hours
salloc --partition=gpu_h100 --gpus=1 --time=4:00:00 \
       --mem=120G --cpus-per-task=16 --account=scur2545
# you should land on a node like gcn149, with nvidia-smi showing H100 95830 MiB
nvidia-smi --query-gpu=name,memory.total --format=csv
```

---

## Stage 1 — Pull latest code

```bash
cd ~/Documents/LLM_test_hpc/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
git pull origin LLM_Syh
# verify the new files are there
ls -lh scripts/grpo_smoke.py scripts/grpo_full.py scripts/run_pipeline.sh \
       scripts/strip_probe_lines.py \
       pipeline/simulation/ltspice_runner_snellius.py \
       pipeline/reward_evaluation/reward_function_norm.py 2>&1
```

You should see all 6 files listed without errors.

---

## Stage 2 — Activate venv + env vars

This is the **single source of truth** for cache/auth env vars.
Every other script reads from these.

```bash
source .venv/bin/activate
unset HF_HOME                           # critical — never point at shared dir
export PYTHONPATH=$(pwd)
export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export TRANSFORMERS_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
```

---

## Stage 3 — Re-train SFT (~10 min)

The training data was cleaned of `.probe` lines (LTspice-incompatible)
so the model no longer learns to emit them.  We also use the OOM-safe
config from before (`bs=2, seq=1024`).

```bash
# nuke the old checkpoint folder
rm -rf checkpoints/sft-lora

# one-shot training (run_pipeline.sh handles env-var defaults)
bash scripts/run_pipeline.sh sft
```

**Expected log markers**:
```
[pipeline] 80GB-class GPU (95830 MiB) → bf16
Loading tokenizer from cache: /projects/2/managed_datasets/hf_cache_dir
Loading weights: 100%|...| 443/443 [00:30<00:00]
trainable params: 64,225,280 || all params: 14,832,532,480 || trainable%: 0.4330
[sft] training on 647 samples, val=71, epochs=5, batch_size=2, lr=0.0002
...
[sft] ── epoch 3/5 ── train_loss=0.0X val_loss=0.0X (~120s)
...
```

**Expected output**:
```
checkpoints/sft-lora/
├─ epoch-1/  epoch-2/  epoch-3/  epoch-4/  epoch-5/  final/
└─ history.json                         # loss curve
```

```bash
# verify
cat checkpoints/sft-lora/history.json
ls -lh checkpoints/sft-lora/epoch-3/
```

---

## Stage 4 — GRPO smoke (~30 s, no simulator)

Verifies that GRPO can pick up the SFT-trained LoRA in **trainable mode**
and run one policy-gradient step on the pre-computed reward data in
`pipeline/data/batch_2/`.  **No LTspice required for this stage.**

```bash
bash scripts/run_pipeline.sh smoke
```

**Expected log markers**:
```
[grpo_smoke] loading TopologyLLM ...
[grpo_smoke] loading SFT LoRA adapter (trainable=True) from ./checkpoints/sft-lora/epoch-3
[grpo_smoke] trainable params: 64,225,280 / 14,832,532,480 (0.433%)

[grpo_smoke] running train_from_existing_batch('batch_2') ...
=== [GRPO/existing-batch] batch_2 ===
Loaded top1: reward=-144.02 (fitness_score)
Loaded top2: reward=-149.02 (fitness_score)
Loaded top3: reward=-20062.40 (fitness_score)
Loaded top4: reward=-137.51 (fitness_score)
RL samples: 4
Saved RL LoRA adapter to: ./checkpoints/grpo-lora/final

  num_valid_samples : 4
  mean_reward       : -5123.X
  policy_loss       : <finite number, not NaN>
  advantages        : [+0.50, +0.50, -1.49, +0.50]

  Verdict: ✅ PASS
```

If `Verdict: ✅ PASS`, the SFT → GRPO hand-off works.  Move on.

**If `Verdict: ❌ FAIL`**: scroll up to find the actual error; common causes:
- `trainable params: 0` — `load_adapter` didn't accept `trainable=True`
  (you forgot to `git pull` Stage 1)
- `No reward file: batch_2/reward_results.json` — batch_2 missing on disk;
  bring it from `origin/main` via `git checkout origin/main -- pipeline/data/batch_2/`

---

## Stage 5 — GRPO full loop (needs LTspice container)

This stage **actually trains the model on simulated rewards**, the main
event of the SFT → GRPO recipe.  All container assets are in this repo
under `containers/ltspice/` — **no team-private files needed**.

### 5.1 Provision the container (~3 min via pull, or ~15 min via build)

```bash
module load 2023
module spider Apptainer       # find the version your HPC has
module load Apptainer/1.2.5-GCCcore-12.3.0    # adjust to whatever spider shows

# Option A (recommended): pull pre-built image from Docker Hub
bash containers/ltspice/build.sh
# → produces ~/container_custom.sif

# Option B: build from Apptainer.def (Debian + Wine 11 + LTspice 64 MSI)
##MODE=build bash containers/ltspice/build.sh
```

Either way you end up with `$HOME/container_custom.sif` (~1.2-1.5 GB).
See `containers/ltspice/README.md` for full details on both modes and
the container's internals.

### 5.2 Wire the container to the Python pipeline

⚠️ **Use the launcher that matches the build mode:**
- `MODE=pull`  (default) → **`run_ltspice_docker.sh`**
- `MODE=build`           → `run_ltspice_snellius.sh`

```bash
mkdir -p $HOME/ltspice-files
export LTSPICE_SIF=$HOME/container_custom.sif

# Default (pull from Docker Hub):
export LTSPICE_LAUNCHER=$PWD/containers/ltspice/run_ltspice_docker.sh

# If you used MODE=build instead:
##export LTSPICE_LAUNCHER=$PWD/containers/ltspice/run_ltspice_snellius.sh

export LTSPICE_FILES_DIR=$HOME/ltspice-files

# Standalone container smoke test (without our Python runner)
cp pipeline/data/batch_sft_expanded_idx0/llm_output/top1.net $LTSPICE_FILES_DIR/
bash $LTSPICE_LAUNCHER /sim/top1.net
ls -lh $LTSPICE_FILES_DIR/top1.raw    # should appear after ~10-30 s
```

If the .raw doesn't appear, fall back to:
```bash
# Run the container interactively to see what's happening
apptainer shell --writable-tmpfs --bind $LTSPICE_FILES_DIR:/sim $LTSPICE_SIF
# inside container:
ls /opt/wineprefix-template/drive_c/Program\ Files/ADI/LTspice/
```
See `containers/ltspice/README.md` "Troubleshooting" for full table.

### 5.3 Run the GRPO full loop

```bash
# default: 50 GRPO iterations, n=4 candidates each, constraint idx 0 (12V → 5V)
bash scripts/run_pipeline.sh full
```

To customise:
```bash
GRPO_N_STEPS=100 GRPO_N_PER_STEP=4 GRPO_CONSTRAINT_IDX=2 \
  bash scripts/run_pipeline.sh full
```

**Expected per-step output**:
```
################################################################################
# Step 1/50  batch=batch_grpo_run_step1
################################################################################
=== [GRPO/full] batch_grpo_run_step1 (n=4) ===
[stage] 1/5  generate
  generated 4 netlists
[stage] 2/5  validate
  Valid: 4/4
[stage] 3/5  simulate
  Batch 'batch_grpo_run_step1' done — 4/4 successful
[stage] 4/5  reward
[stage] 5/5  RL update
  RL samples: 4
  Rewards: [-12.3, -18.7, -8.4, -25.1]
=== GRPO Training Done ===
  policy_loss: 0.XX, kl_loss: 0.0Y, mean_reward: -16.1, ...
```

**Expected total wall time on H100 + container**:
- Per step: ~60–120 s (gen 30 s + sim 30 s + RL 10 s + housekeeping)
- 50 steps ≈ **50 min – 1.5 h**

Results land in:
```
checkpoints/grpo-lora/final/            # final RL-tuned LoRA
checkpoints/grpo-lora/step-{5,10,15,...}/   # intermediate checkpoints
checkpoints/grpo-lora/history.json      # per-step metrics
pipeline/data/batch_grpo_run_step{1..N}/
    ├── llm_output/top{1..4}.net
    ├── validation_results.json
    ├── simulation_results.csv
    ├── reward_results.json
    └── grpo_metrics.json
```

---

## Stage 6 — Tuning GRPO when it isn't learning

This section is the distilled lesson from the 20-step diagnostic run
that produced a flat mean_reward, periodic all-zero reward batches, and
policy_loss swinging in [-6.8, +8.7].  v2 of the updater
(`hpc_configs/overrides/new_rl_updater.py`) ships with defaults that
address all three root causes, but the knobs are exposed so you can
push further if you still don't see learning.

### 6.1 What v2 changed

| Knob              | v1 default | **v2 default** | Why                                                                                            |
|-------------------|------------|----------------|------------------------------------------------------------------------------------------------|
| Policy loss       | sum-of-log-probs | **per-token mean** | TRL/PPO standard. Length-invariant. Compatible scale with KL so kl_beta is meaningful.   |
| `GRPO_LR`         | 2e-5       | **1e-5**       | v1 was unstable (loss ∈ [-6.8, +8.7]). v2's per-token loss is naturally smaller, so 1e-5 here ≈ what a 5e-6 would be on v1 sum-loss. Bump to 3e-5 if reward is flat. |
| `GRPO_KL_BETA`    | 0 (no-op)  | **0.05**       | Was hardcoded to 0 AND not implemented. Now anchors π to π_base via PEFT `disable_adapter()`. |
| `GRPO_ENTROPY_BETA`| n/a       | **0.0**        | Available; bump to 0.01 if you still see mode collapse (3+ identical candidates).             |
| `GRPO_N_STEPS`    | 5          | **50**         | 5-step runs are too short to see learning; 20 was borderline; 50 is the realistic floor.       |
| `GRPO_TEMPERATURE`| 0.5        | **0.7**        | Diversity at sample time fights group-collapse before the gradient even runs.                  |
| Prompt @ log-prob | system_prompt.txt + JSON | **`make_prompt()`** | v1 used a different prompt at log-prob time than at generation time — gradient mis-aligned. |

### 6.2 Diagnose with `grpo_metrics.json`

After each run, `checkpoints/grpo-lora/history.json` has the per-step
metrics.  Read it with:
```bash
python - <<'PY'
import json, statistics
hist = json.load(open("checkpoints/grpo-lora/history.json"))
print(f"steps={len(hist)}")
for h in hist:
    m = h["metrics"]
    flag = "ALL-SAME-REWARD" if m.get("all_same_reward") else ""
    print(f"  step {m['step']:3d}  reward={m['mean_reward']:+.3f}  "
          f"policy={m['policy_loss']:+.3f}  kl={m['kl_loss']:+.4f}  "
          f"|g|={m.get('grad_norm','?')}  {flag}")
rewards = [h["metrics"]["mean_reward"] for h in hist
           if h["metrics"].get("mean_reward") is not None]
if len(rewards) >= 10:
    early = statistics.mean(rewards[:5])
    late  = statistics.mean(rewards[-5:])
    print(f"early-5 mean: {early:+.3f}   late-5 mean: {late:+.3f}   delta: {late-early:+.3f}")
PY
```

What to look for:
- **`mean_reward` trending up** = learning. If flat after 30 steps,
  bump `GRPO_LR` to 1e-5 OR drop `GRPO_KL_BETA` to 0.02.
- **`kl_loss` exploding (>1.0)** = policy diverging from base.  Raise
  `GRPO_KL_BETA` to 0.1 OR lower `GRPO_LR`.
- **`ALL-SAME-REWARD` flagged on >25% of steps** = mode collapse.
  Raise `GRPO_TEMPERATURE` to 0.9 AND `GRPO_ENTROPY_BETA` to 0.01.
- **`|g|` (grad_norm) clipping every step** = LR way too high; drop it.
- **`num_valid_samples=0` (skipped)** = simulator failed all 4; this
  is now non-fatal (v2 skip-gracefully) but if it happens often,
  re-check the LTspice container.

### 6.3 Recommended escalation ladder

If 50-step run shows no learning at v2 defaults:

```bash
# Step 1: more steps + slightly higher LR
GRPO_N_STEPS=100 GRPO_LR=1e-5 \
  bash scripts/run_pipeline.sh full

# Step 2: also fight mode collapse
GRPO_N_STEPS=100 GRPO_LR=1e-5 GRPO_TEMPERATURE=0.9 GRPO_ENTROPY_BETA=0.01 \
  bash scripts/run_pipeline.sh full

# Step 3: relax the KL anchor (let the policy explore further from base)
GRPO_N_STEPS=100 GRPO_LR=1e-5 GRPO_KL_BETA=0.02 GRPO_TEMPERATURE=0.9 \
  bash scripts/run_pipeline.sh full
```

If at this point `mean_reward` is still not climbing, the bottleneck is
upstream — likely the reward signal itself.  Spot-check by running:
```bash
ls pipeline/data/batch_grpo_run_step*/reward_results.json | tail -5 \
  | xargs -I{} python -c "import json,sys; d=json.load(open(sys.argv[1]));
    print(sys.argv[1], {k: v.get('grpo_reward') for k,v in d['circuits'].items()})" {}
```
If rewards are bunched (all within 0.01 of each other), reward signal
has no contrast → GRPO advantage = 0 → no gradient.  Tune the reward
weights in `grpo_trainer.py::train()` (the `weights={...}` dict).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PermissionError ../token` | `HF_HOME` accidentally set to shared dir | `unset HF_HOME` and re-run |
| `LocalEntryNotFoundError ... Qwen3-14B` | cache_dir wrong | `ls /projects/2/managed_datasets/hf_cache_dir/models--Qwen--Qwen3-14B` to confirm |
| `OOM at forward` | seq+batch too big | `SFT_BATCH_SIZE=2 SFT_MAX_LENGTH=1024 bash scripts/run_pipeline.sh sft` |
| `No trainable parameters found for RL update` | `load_adapter` didn't get `trainable=True` | `git pull` (this commit fixes it) |
| smoke step says `Loaded ... missing` | batch_2 not on LLM_Syh | `git checkout origin/main -- pipeline/data/batch_2/` |
| full step crashes in simulate | LTspice container missing/broken | Stage 5.1 troubleshoot |
| validation pass-rate drops to 0/4 | model regressed | retry with `GRPO_TEMPERATURE=0.3` or revert to SFT epoch-3 |

## What's where (file-level cheatsheet)

| File | Purpose |
|---|---|
| `scripts/run_pipeline.sh` | one-shot orchestrator (sft / smoke / full / all) |
| `scripts/run_sft.py` + `scripts/run_sft_smoke.sh` | SFT runner |
| `scripts/grpo_smoke.py` | single-step RL update on existing batch (no simulator) |
| `scripts/grpo_full.py` | full SFT→GRPO loop with simulator |
| `scripts/strip_probe_lines.py` | one-time data cleanup utility (already run) |
| `pipeline/reinforcement_algorithm/sft_trainer.py` | SFT loss + LoRA wrap |
| `pipeline/reinforcement_algorithm/new_rl_updater.py` | GRPO RL updater |
| `pipeline/reinforcement_algorithm/grpo_trainer.py` | GRPO orchestrator (gen→val→sim→reward→update) |
| `pipeline/simulation/ltspice_runner_snellius.py` | calls container via `run_ltspice_snellius.sh` |
| `pipeline/reward_evaluation/reward_function_norm.py` | GRPO-compatible normalized reward |
| `pipeline/netlist_validation/validator.py` | 23 checks + inject `.save V(*) I(*)` (LTspice mode) |
| `data/sft/sft_train.jsonl` + `sft_val.jsonl` | 647 train + 71 val samples (no `.probe`) |
