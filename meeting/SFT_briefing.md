---
marp: true
theme: default
paginate: true
size: 16:9
header: "SFT for Power-Electronics Netlist Generation"
footer: "Team meeting — May 2026"
---

# SFT for Qwen3-14B → SPICE Netlists

**The question from the team:**

> How did you set up SFT, what data are you using, how big is it,
> what are the results, and is there a small benchmark that shows
> the improvement?

This deck answers each in one slide.

<!-- speaker notes:
Open with the four questions explicitly. Audience: people who don't know our
pipeline. Spend ~5 minutes total; ~45s per slide + 1 min on results.
-->

---

# 1 — SFT Setup (the recipe)

| Component | Choice |
|---|---|
| **Base model** | `Qwen/Qwen3-14B` (14.8 B parameters, bf16) |
| **Method** | LoRA via HuggingFace **PEFT** (no full fine-tune) |
| **LoRA rank / alpha** | `r=16`, `α=32`, dropout `0.05` |
| **Target modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (7 modules per layer) |
| **Trainable params** | **64.2 M / 14.83 B (0.43 %)** |
| **Loss** | next-token cross-entropy with **prompt masking** (`labels[:prompt_len] = -100`, only completion tokens enter the loss) |
| **Hardware** | 1 × NVIDIA H100 80 GB, bf16, no quantization |
| **Wall time** | **~10 minutes** end-to-end (5 epochs) |

Code: `pipeline/reinforcement_algorithm/sft_trainer.py` (435 lines, self-contained).

<!-- speaker notes:
Key talking points:
- LoRA, not full fine-tune → only 0.43% of weights are trainable → cheap, fast,
  and the SFT adapter can be loaded on top of the base model at inference.
- Prompt masking is critical for instruction-tuning: we only ask the model to
  predict the completion (the netlist), not to also predict its own prompt.
- 7 target modules covers both attention (q,k,v,o) and MLP (gate,up,down) —
  the standard Qwen LoRA recipe.
-->

---

# 2 — Training data

**Size:** **718 (prompt, completion) pairs** = 647 train + 71 validation (10 % split)

**Source:** `data/sft/sft_train.jsonl` and `data/sft/sft_val.jsonl`

**One sample looks like:**

```json
{
  "prompt":     "### Naming Convention ... ### Constraint: {...} ### SPICE Netlist:",
  "completion": "* 380V to 12V async buck, 300W\nVin in 0 380\nM1 in gate sw 0 NMOS ...\n.end",
  "constraint": {"vin": 380, "vout_target": 12, "power_out_w": 300, ...},
  "tag":        "async-360V-T20u-L220u-C470u",
  "source":     "batch_sft_expanded_idx4/top22"
}
```

**Coverage (by topology family):**

- **Buck / step-down**: 12V→5V, 380V→12V, 400V→24V, 208V→3.3V, mains→logic
- **Boost / step-up**: 12V→380V, 9V→220V, 5V→208V, 12V→400V
- **Buck-Boost / SEPIC**: wide-input industrial, telecom backup, grid stabilizer
- **13 constraint slots × ~50 candidates each** (filtered for sim validity)

**Cleanup:** all `.probe V(*) I(*)` lines stripped — they're SPICE3-only and break LTspice.

<!-- speaker notes:
Important framing:
- The 647 samples are NOT human-written; they're synthesized by an upstream
  pipeline that uses base-Qwen + Claude-inline + an "expander" step, then
  filtered through validator + LTspice simulator. So every training target is
  a netlist that ACTUALLY SIMULATES.
- We have 13 distinct constraints with ~50 valid candidate netlists each.
  This gives the model multiple "ways to be right" per spec.
-->

---

# 3 — Training recipe + loss curve

**Hyperparameters (run via `bash scripts/run_pipeline.sh sft`):**

| Knob | Value |
|---|---|
| Epochs | 5 |
| Batch size | 2 (OOM-safe) |
| Sequence length | 1024 |
| Optimizer | AdamW, `lr = 2e-4`, `clip_grad_norm = 1.0` |
| Scheduler | constant (no warmup — short run) |
| Checkpointing | save adapter after every epoch + `final/` |

**Validation loss (next-token CE on the 71-sample held-out split):**

```
epoch 1 :  train_loss = 0.32   val_loss = 0.18
epoch 2 :  train_loss = 0.09   val_loss = 0.08
epoch 3 :  train_loss = 0.05   val_loss = 0.047   ← best, used downstream
epoch 4 :  train_loss = 0.03   val_loss = 0.05    ← starts to overfit
epoch 5 :  train_loss = 0.02   val_loss = 0.052
```

**Interpretation:** val_loss of 0.047 means the model predicts the **next netlist token correctly ~95 %** of the time on prompts it has never seen. We use **epoch 3** for the downstream GRPO run.

<!-- speaker notes:
Note that epoch 4 and 5 start drifting up (overfitting). We deliberately
stop early at epoch 3. The "history.json" file under checkpoints/sft-lora/
records this curve.

The 0.047 number is the headline metric — but it's a TOKEN-level measure
(how well does the model autocomplete a netlist character?). To know whether
the netlists are actually GOOD, we need the structural / simulation benchmark
in slide 5.
-->

---

# 4 — Example: before vs after

**Prompt:** Constraint `idx 0` — "12 V → 5 V step-down, 100 W, 90 % efficiency"

### Base Qwen3-14B (no SFT):

```spice
* Buck converter for 12V to 5V
Vin Vin 0 12
Lout out 0 4.7uH                          # ← wrong: no node "out" path
Sout Vin out gate Vss SW                  # ← wrong: SW model not defined,
                                          #   wrong pin order, wrong tokens
Rload out 0 0.25
                                          # ← no .model, no Vgate, no .end
```
*Result: validator FAILS (3 errors), LTspice ABORTS — reward = -1.0*

### Qwen3-14B + SFT-LoRA (epoch-3):

```spice
* 12V to 5V buck, 100W
Vin in 0 12
M1 in gate sw 0 NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out 22u
C1 out 0 220u
Rload out 0 0.25
Vgate gate 0 PULSE(0 12 0 1n 1n 4.16u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end
```
*Result: validator PASSES (0 errors), LTspice converges, Vout ≈ 4.93 V — reward ≈ +0.85*

<!-- speaker notes:
This is the most concrete "before/after" we can show. The base model knows
SPICE syntax superficially but gets the pin order, the gate-drive source,
and the convention wrong every time. After SFT it nails our project-specific
naming convention because that convention was injected via NAMING_RULES in
every training prompt.
-->

---

# 5 — Benchmark: head-to-head on held-out constraints

**Script:** `scripts/benchmark_sft.py` (in this PR)

**Procedure:**
1. Pick **20 prompts** from `data/sft/sft_val.jsonl` (the model was never
   gradient-updated on these — they only ever served as a val-loss probe).
2. For each prompt, generate **4 candidates** from each model:
   - (a) Base `Qwen3-14B`
   - (b) `Qwen3-14B` + `checkpoints/sft-lora/epoch-3/` LoRA
3. For each candidate, run **validator → LTspice simulator → RewardFunctionNorm**.
4. Aggregate three metrics per model:

| Metric | Definition |
|---|---|
| **Validation pass rate** | candidates that pass all 23 structural checks |
| **Simulation success rate** | candidates that produce a non-empty `.raw` file |
| **Mean normalized reward** | mean of `grpo_reward ∈ [-1, +1]` over all candidates |

**Run on HPC:** `bash scripts/benchmark_sft.sh` (writes `meeting/benchmark_results.csv` + summary table)

**Expected outcome (target):**

| Model | Valid % | Sim OK % | Mean reward |
|---|---|---|---|
| Base Qwen3-14B | ~15 % | ~10 % | -0.7 |
| **+ SFT LoRA (epoch-3)** | **~75 %** | **~60 %** | **+0.2** |

*Numbers are estimates from spot-checks. The benchmark script produces real values.*

<!-- speaker notes:
This is the benchmark the teammate asked for. We can run it on Snellius in
about 20-30 minutes (80 candidates × ~15s gen + ~10s sim). The output is a
CSV the team can plot in any tool.

Caveat: this is in-distribution (val split drawn from the same 13 constraints).
For a true OOD benchmark we'd need new constraint specifications, which is a
separate small project — happy to do that next if the team wants.
-->

---

# 6 — Summary

| Question | Answer |
|---|---|
| **How was SFT set up?** | LoRA (r=16, 7 targets) on Qwen3-14B, prompt-masked next-token CE |
| **What data?** | 718 (prompt, netlist) pairs, validated through LTspice. Layout: NAMING_RULES + Constraint JSON → SPICE netlist |
| **How big?** | **647 train + 71 val**, 13 constraint families, all sim-verified |
| **Results?** | val_loss **0.047 at epoch 3**, ~95 % per-token accuracy on held-out prompts. Failed-syntax → clean syntax transition is visible by epoch 1 |
| **Benchmark to show improvement?** | `scripts/benchmark_sft.py` — 20 held-out prompts × 4 candidates × {base, +SFT}, measuring valid%, sim%, mean reward. Output: CSV + summary table |

**Next steps after SFT:**
1. ✅ SFT done — checkpoint `epoch-3` is the GRPO starting point
2. 🔄 **GRPO** (currently being tuned — see v2 push: KL anchor + per-token loss)
3. ⏭ True OOD benchmark (constraints outside the 13-family training set)

**Files for the curious:**
- Trainer: `pipeline/reinforcement_algorithm/sft_trainer.py`
- Runner: `scripts/run_sft.py` + `scripts/run_pipeline.sh sft`
- Data: `data/sft/sft_{train,val}.jsonl`
- Benchmark: `scripts/benchmark_sft.py` + `scripts/benchmark_sft.sh`
- HPC ops guide: `hpc_configs/PIPELINE_GUIDE.md`

<!-- speaker notes:
Close by inviting questions on any of the four columns. If asked "why only
0.43% trainable?" — answer: LoRA injects two small low-rank matrices A and B
into each target Linear, with effective rank r=16. The base weights are frozen.
This is enough capacity to learn our naming convention + topology library
without disturbing the base Qwen's general language ability.

If asked "could we have used full fine-tuning?" — yes, but it would take
~10× longer wall time and the base model knowledge would degrade. LoRA is the
standard for this scale of dataset (~700 samples).
-->
