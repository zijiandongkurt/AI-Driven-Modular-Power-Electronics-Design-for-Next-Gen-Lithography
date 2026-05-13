"""
generate_sft_via_claude.py
==========================

Use Anthropic's Claude API to generate high-quality SFT samples that
respect NAMING_RULES + are electrically reasonable (proper D, L, C, R
values based on physics).

Why this is better than local Qwen2.5-Coder-7B (which `generate_sft_candidates.py`
uses):
  - Sonnet 4.5 / Opus 4.7 understands power-electronics physics directly
  - Computes D = Vout/Vin, R = V^2/P, L ≈ Vout(1-D)T/(0.3*Iout), etc.
  - Distinguishes buck / boost / buck-boost topologies correctly
  - Follows multi-rule prompts reliably (no JSON-pseudo-code regressions)

Expected validator pass-rate: ~98 % (vs Coder-2.5's 82 %)
Expected electrical plausibility (vout close to target):  ~75 %
                                                          (Coder-2.5: unknown,
                                                          probably 30-50 %)

Pricing (rough, varies by model):
  Sonnet 4.5     ~$0.006/sample → 200 samples ≈ $1.20
  Opus 4.x       ~$0.030/sample → 200 samples ≈ $6.00

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

CLI:
    python scripts/generate_sft_via_claude.py \
        --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 \
        --n-per-constraint 16 \
        --model claude-sonnet-4-5 \
        --out data/sft/sft_from_claude.jsonl

Outputs:
  data/sft/sft_from_claude.jsonl           validator-passing pairs
  pipeline/data/batch_sft_claude_idx<N>/   canonical batch layout for
                                           later HPC fitness scoring
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_topology_generation.prompt_input import (
    NAMING_RULES, load_constraints, make_prompt,
)
from pipeline.llm_topology_generation.net_writer import (
    write_netlists, get_llm_output_dir,
)


# ────────────────────────────────────────────────────────────────────────
# Few-shot example pool, hand-picked to demonstrate format + physics
# ────────────────────────────────────────────────────────────────────────

FEWSHOT_BUCK = """\
* 12V to 5V buck, 100W, async, 100kHz
* D=5/12=0.417, Rload=V^2/P=0.25, t_on=4.17u
Vin in 0 12
M1 in gate sw 0 NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out 100u
C1 out 0 470u
Rload out 0 0.25
Vgate gate 0 PULSE(0 12 0 1n 1n 4.17u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end"""

FEWSHOT_BOOST = """\
* 24V to 380V boost, 400W, 100kHz
* D = 1 - Vin/Vout = 0.937, t_on=9.37u
* L1 sits on the input side (in -> sw), M1 is low-side (source=0)
Vin in 0 24
L1 in sw 100u
M1 sw gate 0 0 NMOS W=1 L=1
D1 sw out DIODE
C1 out 0 100u
Rload out 0 361
Vgate gate 0 PULSE(0 12 0 1n 1n 9.37u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end"""


def _classify_topology(constraint: dict) -> str:
    """Return 'buck' / 'boost' / 'buck-boost' from a constraint dict.

    Decision: if Vin can be both above and below Vout → buck-boost.
              if Vin always > Vout → buck.
              if Vin always < Vout → boost.
    """
    vin_min = constraint.get("vin_min", 0)
    vin_max = constraint.get("vin_max", 0)
    vout    = constraint.get("vout_target", 0)
    if vin_min >= vout:
        return "buck"
    if vin_max <= vout:
        return "boost"
    return "buck-boost"


def _build_messages(constraint: dict, n: int):
    """Build the Anthropic messages payload."""
    topo = _classify_topology(constraint)
    fewshots = []
    if topo in ("buck", "buck-boost"):
        fewshots.append(FEWSHOT_BUCK)
    if topo in ("boost", "buck-boost"):
        fewshots.append(FEWSHOT_BOOST)

    system = (
        "You are an expert power-electronics designer who writes valid "
        "ngspice / LTspice netlists for non-isolated DC/DC converters.\n\n"
        + NAMING_RULES
        + "\n\nALWAYS compute component values from physics first, then "
        "write the netlist.  D = Vout/Vin for buck, D = 1 - Vin/Vout for "
        "boost.  Rload = Vout^2 / Pout.  Inductor L ~ Vout*(1-D)*T / (0.3*Iout). "
        "Output ONLY the requested JSON — no Markdown fences, no other prose."
    )

    user_parts = [
        f"## Worked examples (format you must match)\n"
    ]
    for i, ex in enumerate(fewshots, 1):
        user_parts.append(f"### Example {i}\n```\n{ex}\n```\n")

    user_parts.append(
        f"\n## Target constraint\n"
        f"```json\n{json.dumps({k: v for k, v in constraint.items() if not k.startswith('_')}, indent=2)}\n```\n"
    )
    user_parts.append(
        f"\n## Task\n"
        f"Produce **{n} distinct, valid SPICE netlists** that satisfy the "
        f"target constraint.  Use {topo} topology (most appropriate for "
        f"the Vin range vs Vout_target).  Vary across these axes:\n"
        f"  - inductor value (e.g. 47u / 68u / 100u / 220u)\n"
        f"  - capacitor value (e.g. 100u / 220u / 470u / 1000u)\n"
        f"  - switching period (e.g. 10u / 20u / 50u corresponding to 100k / 50k / 20kHz)\n"
        f"  - single-phase vs synchronous (with M2) vs interleaved 2-phase\n"
        f"\nFor every netlist, compute D from Vin and Vout_target, set "
        f"`t_on = D * period`, set `Rload = Vout^2 / P_in`.\n"
        f"\nReturn STRICT JSON of the form:\n"
        f'  {{"netlists": ["netlist1 text...", "netlist2 text...", ...]}}\n'
        f"Each netlist must end with `.end`.  No commentary.  No Markdown fences."
    )

    return system, "\n".join(user_parts)


def _extract_netlists(text: str) -> list[str]:
    """Parse the Claude response into a list of netlist strings."""
    # Strip Markdown code fences if Claude added them
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    try:
        obj = json.loads(text)
        nets = obj.get("netlists", [])
        return [n.strip() for n in nets if isinstance(n, str) and n.strip()]
    except json.JSONDecodeError:
        # Fall-back: try to find array of strings
        m = re.search(r'\{[^}]*"netlists"\s*:\s*\[(.*)\]\s*\}', text, re.DOTALL)
        if not m:
            return []
        try:
            arr = json.loads("[" + m.group(1) + "]")
            return [n.strip() for n in arr if isinstance(n, str) and n.strip()]
        except Exception:
            return []


def _call_claude(client, model: str, system: str, user: str, max_tokens: int):
    """Send one chat-completion request to Anthropic's API."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # response.content is a list of blocks; we want the text block
    out_parts = [b.text for b in resp.content if hasattr(b, "text")]
    return "".join(out_parts), resp.usage


def _run_validator(batch_id: str) -> dict:
    try:
        from pipeline.netlist_validation.validator import validator
    except ImportError as e:
        print(f"  WARN: validator unavailable ({e}); keeping all non-empty")
        return {}
    val = validator()
    val.validate(batch_id)
    p = REPO_ROOT / "pipeline" / "data" / batch_id / "validation_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def generate_for_constraint(client, model, max_tokens, constraint, idx, n,
                            batch_prefix):
    print(f"\n══════ Constraint #{idx} "
          f"({_classify_topology(constraint)}): "
          f"vin {constraint.get('vin_min')}-{constraint.get('vin_max')}V "
          f"→ {constraint.get('vout_target')}V / {constraint.get('power_in')}W ══════")

    batch_id = f"{batch_prefix}_idx{idx}"
    system, user = _build_messages(constraint, n)

    # Approximate max-tokens budget: 400 tokens per netlist
    cap = max(max_tokens, n * 500)
    t0 = time.time()
    raw, usage = _call_claude(client, model, system, user, cap)
    dt = time.time() - t0
    print(f"[api] {dt:.1f}s; "
          f"in={usage.input_tokens} out={usage.output_tokens}")

    netlists = _extract_netlists(raw)
    print(f"[parse] got {len(netlists)} netlist(s) back")

    if not netlists:
        print("  RAW (truncated):", raw[:300])
        return []

    # Write to canonical batch layout, validate
    written = write_netlists(netlists=netlists, batchID=batch_id)
    print(f"[write] {len(written)} files → {get_llm_output_dir(batch_id)}")

    val_results = _run_validator(batch_id)
    n_passed = sum(1 for v in val_results.values() if v.get("passed"))
    print(f"[validate] {n_passed}/{len(val_results)} passed all 23 checks")

    prompt = make_prompt(constraint)
    pairs = []
    for path in written:
        info = val_results.get(path.stem) or val_results.get(path.name) or {}
        if not info.get("passed"):
            continue
        pairs.append({
            "prompt":     prompt,
            "completion": path.read_text(encoding="utf-8"),
            "fitness":    None,
            "source":     f"{batch_id}/{path.stem}",
            "constraint": {k: v for k, v in constraint.items() if not k.startswith("_")},
            "generator":  f"claude-api/{model}",
        })
    print(f"[sft] kept {len(pairs)} validator-passing pair(s)")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", type=Path,
                    default=REPO_ROOT / "pipeline" / "data" / "datasets" / "constraints.json")
    ap.add_argument("--indices", type=int, nargs="+", default=list(range(13)))
    ap.add_argument("--n-per-constraint", type=int, default=16)
    ap.add_argument("--model", type=str, default="claude-sonnet-4-5",
                    help="Anthropic model id, e.g. claude-sonnet-4-5, claude-opus-4-5")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="Output token budget per API call")
    ap.add_argument("--batch-prefix", type=str, default="batch_sft_claude")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "sft" / "sft_from_claude.jsonl")
    args = ap.parse_args()

    # ── Anthropic client ──────────────────────────────────────────────
    try:
        from anthropic import Anthropic
    except ImportError:
        print("FATAL: pip install anthropic", file=sys.stderr); sys.exit(2)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("FATAL: set ANTHROPIC_API_KEY env var", file=sys.stderr); sys.exit(2)
    client = Anthropic(api_key=api_key)

    # ── Load constraints ──────────────────────────────────────────────
    constraints = load_constraints(str(args.constraints))
    print(f"[setup] loaded {len(constraints)} constraints from {args.constraints}")
    print(f"[setup] indices={args.indices}, n_per_constraint={args.n_per_constraint}, "
          f"model={args.model}")

    # ── Generate per constraint ───────────────────────────────────────
    all_pairs: list[dict] = []
    total_in_tokens = 0
    total_out_tokens = 0
    for idx in args.indices:
        if idx >= len(constraints):
            print(f"WARN: index {idx} out of range"); continue
        try:
            pairs = generate_for_constraint(
                client, args.model, args.max_tokens,
                constraints[idx], idx, args.n_per_constraint,
                args.batch_prefix,
            )
        except Exception as e:
            print(f"  ERROR on constraint #{idx}: {e!r}")
            continue
        all_pairs.extend(pairs)

    # ── Save JSONL ────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in all_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(all_pairs)} validator-passing pair(s)")
    print(f"[done] written to {args.out}")


if __name__ == "__main__":
    main()
