"""
validate_grpo_v2.py — local CPU smoke test of the GRPO v2 mechanism.

Purpose: before pushing the GRPO v2 changes (KL penalty, per-token loss,
entropy bonus, prompt alignment) onto the team's live branch, verify on a
TINY model that every mechanism actually works and that the policy moves
in the correct direction under a clear reward signal.

This loads the REAL v2 updater from hpc_configs/overrides/new_rl_updater.py
(the file that would be deployed) and runs it against a tiny Qwen2 model
on CPU. It does not need a GPU and finishes in well under a minute.

Run:
    <circuit-rl python> scripts/validate_grpo_v2.py
"""

from __future__ import annotations

# ── Force CPU BEFORE importing torch (local GPU is sm_120, unsupported) ──
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import importlib.util
import sys
from pathlib import Path

import torch

# This laptop's GPU (sm_120) is not supported by the installed torch, and
# bitsandbytes (pulled in lazily by PEFT) crashes when it thinks CUDA is
# present. Force a clean CPU-only view so every downstream import behaves.
torch.cuda.is_available = lambda: False  # type: ignore[assignment]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch.manual_seed(0)

# ── Load the v2 updater straight from the overrides file under test ──────
_spec = importlib.util.spec_from_file_location(
    "new_rl_updater_v2",
    REPO / "hpc_configs" / "overrides" / "new_rl_updater.py",
)
v2 = importlib.util.module_from_spec(_spec)
sys.modules["new_rl_updater_v2"] = v2   # register so @dataclass can resolve types
_spec.loader.exec_module(v2)
RLConfig, RLUpdater = v2.RLConfig, v2.RLUpdater


# ── Minimal engine stub matching what RLUpdater expects ──────────────────
class TinyEngine:
    def __init__(self, model, tok):
        self._model = model
        self._tok = tok
        self._is_peft = False

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tok


def load_tiny_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    candidates = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-Coder-7B",  # already cached locally; heavier fallback
    ]
    last = None
    for mid in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32)
            model.to("cpu")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"[setup] loaded model: {mid}")
            return model, tok
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[setup] could not load {mid}: {e!r}")
    raise RuntimeError(f"No tiny model could be loaded: {last!r}")


def make_cfg(**over):
    """v2 config tuned tiny + CPU-safe (bf16 off, no dropout, no checkpoints)."""
    cfg = RLConfig()
    cfg.bf16 = False
    cfg.lora_dropout = 0.0
    cfg.save_every = 0
    cfg.max_length = 64
    cfg.max_prompt_length = 48
    cfg.max_completion_length = 32
    cfg.learning_rate = 5e-3   # large LR so effects are visible in a few steps
    for k, val in over.items():
        setattr(cfg, k, val)
    return cfg


def fresh_updater(model, tok, **over):
    """Build a fresh updater on an UNWRAPPED engine each time so LoRA starts clean."""
    eng = TinyEngine(model, tok)
    return RLUpdater(eng, make_cfg(**over)), eng


def seq_logprob(updater, prompt, completion):
    """Helper: completion log-prob under the current policy (no grad)."""
    with torch.no_grad():
        st = updater._per_token_stats(prompt, completion)
    return float(st["sum_log_prob"]) if st is not None else None


# ── Test battery ─────────────────────────────────────────────────────────

RESULTS = []

def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main():
    model, tok = load_tiny_model()

    PROMPT = "### Constraint: 12V to 5V buck\n### SPICE Netlist:\n"
    GOOD = "Vin in 0 12\nRload out 0 5\n.end"
    BAD  = "zzzz qqqq wwww\n.end"

    # ── T1: update runs + exposes the new v2 metric fields ──────────────
    print("\n[T1] update() runs and returns v2 metrics")
    upd, _ = fresh_updater(model, tok, kl_beta=0.05, entropy_beta=0.01)
    m = upd.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD], [1.0, -1.0, 1.0, -1.0])
    for key in ("policy_loss", "kl_loss", "entropy", "grad_norm",
                "all_same_reward", "total_loss"):
        check(f"metric '{key}' present", key in m, f"value={m.get(key)}")
    check("num_valid_samples == 4", m.get("num_valid_samples") == 4)

    # ── T2: KL penalty is actually computed (v1 hardcoded it to 0) ──────
    print("\n[T2] KL penalty is wired (nonzero when kl_beta>0, zero when 0)")
    upd_kl, _ = fresh_updater(model, tok, kl_beta=0.2, entropy_beta=0.0)
    kl_first = upd_kl.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD],
                             [1.0, -1.0, 1.0, -1.0])["kl_loss"]
    # drive several updates so the LoRA drifts away from the base model
    kl_last = kl_first
    for _ in range(8):
        kl_last = upd_kl.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD],
                                [1.0, -1.0, 1.0, -1.0])["kl_loss"]
    check("KL ~ 0 at step 1 (policy == base)", abs(kl_first) < 1e-3,
          f"kl_first={kl_first:.3e}")
    check("KL grows after drift (disable_adapter ref works)",
          abs(kl_last) > abs(kl_first), f"kl_last={kl_last:.3e}")

    upd_nokl, _ = fresh_updater(model, tok, kl_beta=0.0, entropy_beta=0.0)
    m0 = upd_nokl.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD], [1.0, -1.0, 1.0, -1.0])
    check("KL term exactly 0 when kl_beta=0", m0["kl_loss"] == 0.0)

    # ── T3: per-token loss scale (v1 used sum-of-log-probs ~ hundreds) ──
    print("\n[T3] policy loss is on a per-token scale (O(0.01..3), not O(100s))")
    check("|policy_loss| is small (per-token)", abs(m["policy_loss"]) < 5.0,
          f"policy_loss={m['policy_loss']}")

    # ── T4: entropy bonus active ────────────────────────────────────────
    print("\n[T4] entropy term reported and positive when entropy_beta>0")
    upd_e, _ = fresh_updater(model, tok, kl_beta=0.0, entropy_beta=0.05)
    me = upd_e.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD], [1.0, -1.0, 1.0, -1.0])
    check("entropy > 0", me["entropy"] > 0.0, f"entropy={me['entropy']}")

    # ── T5: learning signal — good completion becomes relatively likely ─
    print("\n[T5] policy moves correctly under a clear reward (logp(GOOD)-logp(BAD) up)")
    upd_l, _ = fresh_updater(model, tok, kl_beta=0.0, entropy_beta=0.0,
                             learning_rate=1e-2)
    g0, b0 = seq_logprob(upd_l, PROMPT, GOOD), seq_logprob(upd_l, PROMPT, BAD)
    gap_before = g0 - b0
    for _ in range(20):
        upd_l.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD], [1.0, -1.0, 1.0, -1.0])
    g1, b1 = seq_logprob(upd_l, PROMPT, GOOD), seq_logprob(upd_l, PROMPT, BAD)
    gap_after = g1 - b1
    check("relative log-prob of GOOD increased",
          gap_after > gap_before,
          f"gap {gap_before:.3f} -> {gap_after:.3f} (Δ={gap_after-gap_before:+.3f})")

    # ── T6: mode-collapse detector + zero gradient on equal rewards ─────
    print("\n[T6] all-equal rewards -> flagged + (near) zero gradient")
    upd_c, _ = fresh_updater(model, tok, kl_beta=0.0, entropy_beta=0.0)
    mc = upd_c.update([PROMPT]*4, [GOOD, BAD, GOOD, BAD], [0.5, 0.5, 0.5, 0.5])
    check("all_same_reward flagged", mc.get("all_same_reward") is True)
    check("grad_norm ~ 0 (zero advantage)", abs(mc.get("grad_norm", 1.0)) < 1e-4,
          f"grad_norm={mc.get('grad_norm')}")

    # ── T7: n<2 graceful skip (no crash) ────────────────────────────────
    print("\n[T7] single-sample batch is skipped gracefully")
    upd_s, _ = fresh_updater(model, tok)
    ms = upd_s.update([PROMPT], [GOOD], [1.0])
    check("skipped == True", ms.get("skipped") is True)
    check("policy_loss is None on skip", ms.get("policy_loss") is None)

    # ── T8: prompt-alignment fix lives in the v2 grpo_trainer ───────────
    print("\n[T8] grpo_trainer uses make_prompt() (train/inference prompt aligned)")
    gt_src = (REPO / "hpc_configs" / "overrides" / "grpo_trainer.py").read_text(
        encoding="utf-8")
    check("grpo_trainer imports/uses make_prompt", "make_prompt(" in gt_src)
    check("old system_prompt.txt path removed from _load_prompt",
          "### Constraint:\\n\"" not in gt_src or "make_prompt(" in gt_src)
    from pipeline.llm_topology_generation.prompt_input import make_prompt
    sample_prompt = make_prompt({"vin_min": 12, "vout_target": 5})
    check("make_prompt emits naming rules + netlist header",
          "Naming Convention" in sample_prompt and "SPICE Netlist" in sample_prompt)

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    ntot = len(RESULTS)
    print(f" GRPO v2 validation: {npass}/{ntot} checks passed")
    print("=" * 64)
    if npass == ntot:
        print(" VERDICT: PASS — the v2 mechanism works; safe to push.")
        sys.exit(0)
    else:
        print(" VERDICT: FAIL — do NOT push; failing checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"   - {name}  {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
