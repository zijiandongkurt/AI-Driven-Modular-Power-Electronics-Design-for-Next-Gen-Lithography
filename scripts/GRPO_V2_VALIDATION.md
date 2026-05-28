# GRPO v2 — Local Validation Report

**Verdict: PASS (20/20 checks).** The GRPO v2 mechanism was validated on a
real model before deployment.

## What was tested

The harness `scripts/validate_grpo_v2.py` loads the **actual deployed file**
(`pipeline/reinforcement_algorithm/new_rl_updater.py`, identical to
`hpc_configs/overrides/new_rl_updater.py`) and exercises every v2 change
against a live PEFT-LoRA model. It does not test a mock — it runs the same
forward / KL-reference / backward / clip / optimizer-step path that the HPC
run uses.

## Environment

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (same architecture family as Qwen3-14B; identical LoRA target modules) |
| Device | CPU (local RTX 5070 is `sm_120`, unsupported by the installed torch 2.4; `torch.cuda.is_available` forced `False`) |
| Env | conda `circuit-rl` — torch 2.4.0, transformers 4.45.0, peft 0.13.0 |
| LoRA applied | 8.8M / 502M params trainable (1.75%) — wrapping succeeded on a real PEFT model |
| Runtime | < 1 min on CPU |

A 0.5B stand-in is sufficient because the test targets the **mechanism**
(advantage → gradient → log-prob change, KL computation, loss scaling), not
language quality. The same code path runs unchanged on Qwen3-14B.

## Results

| # | Check | Validates | Result |
|---|---|---|---|
| T1 | `update()` returns `kl_loss`, `entropy`, `grad_norm`, `all_same_reward`, `total_loss`; `num_valid_samples==4` | v2 metrics surface for diagnosis | PASS (6 checks) |
| T2 | KL ≈ 0 at step 1 (policy == base); KL grows after the LoRA drifts; KL is exactly 0 when `kl_beta=0` | **KL penalty is really implemented** via `disable_adapter()` (v1 hard-coded it to 0) | PASS (3 checks) |
| T3 | `\|policy_loss\|` stays in O(0.01–1) | **per-token mean loss** (v1 used sum-of-log-probs, O(100s)) | PASS |
| T4 | `entropy > 0` when `entropy_beta>0` and it enters `total_loss` | **entropy bonus** wired | PASS |
| T5 | after 20 steps with reward(GOOD)=+1, reward(BAD)=−1, `logp(GOOD) − logp(BAD)` increases | **learning direction is correct** (advantage → gradient sign) | PASS |
| T6 | all-equal rewards → `all_same_reward=True` and `grad_norm ≈ 0` | **mode-collapse detector**; zero advantage ⇒ zero gradient | PASS (2 checks) |
| T7 | single-sample batch → `skipped=True`, `policy_loss=None`, no crash | **n<2 graceful skip** | PASS (2 checks) |
| T8 | `grpo_trainer.py` uses `make_prompt()`, old `system_prompt.txt` path gone, `make_prompt` emits naming-rules + netlist header | **prompt-alignment fix** (train/inference prompt now identical) | PASS (3 checks) |

**Total: 20 / 20 passed.**

## Why this matters

The team's `inference-demo` branch still ships the **v1** updater
(`kl_beta=0.0`, KL not implemented, sum-of-log-probs loss, no entropy, no
prompt alignment) — the exact configuration that produced the flat
mean-reward and mode-collapse seen in the 20-step diagnostic run. This
validation confirms the v2 fixes work on a real model and are safe to deploy.

## How to reproduce

```bash
# in the circuit-rl env (or any env with torch+transformers+peft)
python scripts/validate_grpo_v2.py
# expected tail: "GRPO v2 validation: 20/20 checks passed  /  VERDICT: PASS"
```
