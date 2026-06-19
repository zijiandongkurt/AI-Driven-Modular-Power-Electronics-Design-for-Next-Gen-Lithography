"""
expand_sft_samples.py
=====================

Template-driven SFT sample expansion.  For each constraint in
constraints.json, iterates Vin × period × L × C × topology to produce a
diverse set of physics-consistent netlists, then runs the validator to
keep only those that pass all 23 checks.

Differences from claude_inline_samples.py:
  - Programmatic, not hand-written (scales to hundreds of samples)
  - All values derived from physics:
      Buck:   D = Vout / Vin
      Boost:  D = 1 - Vin / Vout
      SEPIC:  D = Vout / (Vin + Vout)
      Rload = Vout^2 / Pin
  - Targets LTspice syntax (`.tran 1n 5m`, `Meg`, etc.); validator-passing
    netlists are guaranteed LTspice-runnable.

Expected output: ~250-350 validator-passing pairs across all 13 constraints.

Run:
    python scripts/expand_sft_samples.py
Output:
    data/sft/sft_from_template.jsonl              SFT pairs (validator-only)
    pipeline/data/batch_sft_expanded_idx<N>/      canonical batch layout
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_topology_generation.prompt_input import load_constraints, make_prompt
from pipeline.llm_topology_generation.net_writer import write_netlists


#  Format helpers 

def f(v, prec=3):
    """Format a numeric value with reasonable precision + SPICE suffix."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6:
        return f"{v/1e6:.{prec}g}Meg"
    if a >= 1e3:
        return f"{v/1e3:.{prec}g}k"
    if a >= 1:
        return f"{v:.{prec}g}"
    if a >= 1e-3:
        return f"{v*1e3:.{prec}g}m"
    if a >= 1e-6:
        return f"{v*1e6:.{prec}g}u"
    if a >= 1e-9:
        return f"{v*1e9:.{prec}g}n"
    return f"{v:.{prec}g}"


def n(x):
    """Format integer-like floats without trailing .0 for tags."""
    return str(int(x)) if x == int(x) else str(x)


#  Topology templates 

def buck_async(vin, vout, p, T, L, C):
    D = vout / vin
    t_on = D * T
    Rload = vout * vout / p
    return f"""* {f(vin)}V to {f(vout)}V async buck, {f(p)}W
* D={f(D)}, t_on={f(t_on)}, Rload={f(Rload)}
Vin in 0 {f(vin)}
M1 in gate sw 0 NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out {f(L)}
C1 out 0 {f(C)}
Rload out 0 {f(Rload)}
Vgate gate 0 PULSE(0 12 0 1n 1n {f(t_on)} {f(T)})
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end"""


def buck_sync(vin, vout, p, T, L, C):
    D = vout / vin
    t_on_hi = D * T
    delay_lo = t_on_hi + T * 0.05      # 5 % dead time
    t_on_lo = T - delay_lo - T * 0.02   # leave 2 % final dead time
    Rload = vout * vout / p
    return f"""* {f(vin)}V to {f(vout)}V sync buck (M2 replaces D1), {f(p)}W
* D={f(D)}, Rload={f(Rload)}; dead time ~5% each end
Vin in 0 {f(vin)}
M1 in gate sw 0 NMOS W=1 L=1
M2 sw gate2 0 0 NMOS W=1 L=1
L1 sw out {f(L)}
C1 out 0 {f(C)}
Rload out 0 {f(Rload)}
Rbleed sw 0 1Meg
Vgate  gate  0 PULSE(0 12 0 1n 1n {f(t_on_hi)} {f(T)})
Vgate2 gate2 0 PULSE(0 12 {f(delay_lo)} 1n 1n {f(t_on_lo)} {f(T)})
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.tran 1n 5m
.end"""


def buck_interleaved(vin, vout, p, T, L, C):
    D = vout / vin
    t_on = D * T
    delay2 = T / 2
    Rload = vout * vout / p
    return f"""* {f(vin)}V to {f(vout)}V interleaved 2-phase buck, {f(p)}W
* D={f(D)}, t_on={f(t_on)}, 180-deg phase shift
Vin in 0 {f(vin)}
M1 in gate1 sw1 0 NMOS W=1 L=1
M2 in gate2 sw2 0 NMOS W=1 L=1
D1 0 sw1 DIODE
D2 0 sw2 DIODE
L1 sw1 out {f(L)}
L2 sw2 out {f(L)}
C1 out 0 {f(C)}
Rload out 0 {f(Rload)}
Vgate1 gate1 0 PULSE(0 12 0 1n 1n {f(t_on)} {f(T)})
Vgate2 gate2 0 PULSE(0 12 {f(delay2)} 1n 1n {f(t_on)} {f(T)})
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end"""


def boost(vin, vout, p, T, L, C):
    D = 1 - vin / vout
    t_on = D * T
    Rload = vout * vout / p
    return f"""* {f(vin)}V to {f(vout)}V boost, {f(p)}W
* D=1-{f(vin)}/{f(vout)}={f(D)}, t_on={f(t_on)}, Rload={f(Rload)}
* L1 on input side, M1 is low-side switch
Vin in 0 {f(vin)}
L1 in sw {f(L)}
M1 sw gate 0 0 NMOS W=1 L=1
D1 sw out DIODE
C1 out 0 {f(C)}
Rload out 0 {f(Rload)}
Rbleed sw 0 1Meg
Vgate gate 0 PULSE(0 12 0 1n 1n {f(t_on)} {f(T)})
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end"""


def sepic(vin, vout, p, T, L1v, L2v, Cc, Cout):
    D = vout / (vin + vout)
    t_on = D * T
    Rload = vout * vout / p
    return f"""* {f(vin)}V to {f(vout)}V SEPIC, {f(p)}W
* D=Vout/(Vin+Vout)={f(D)}, t_on={f(t_on)}
* L1 input, M1 grounds sw1, C1 coupling sw1<->sw2, L2 sw2->0, D1 sw2->out
Vin in 0 {f(vin)}
L1 in sw1 {f(L1v)}
M1 sw1 gate 0 0 NMOS W=1 L=1
C1 sw1 sw2 {f(Cc)}
L2 sw2 0 {f(L2v)}
D1 sw2 out DIODE
C2 out 0 {f(Cout)}
Rload out 0 {f(Rload)}
Rbleed sw1 0 1Meg
Rbleed1 sw2 0 1Meg
Vgate gate 0 PULSE(0 12 0 1n 1n {f(t_on)} {f(T)})
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1n 5m
.end"""


#  Per-constraint topology + Vin sweep config 
#
# Choose: (topology_class, list of Vin values within [vin_min, vin_max])

PRESETS: dict[int, tuple[str, list[float]]] = {
    # IDX     class      Vin sweep within [vin_min, vin_max]
    0:  ('buck',  [12, 18, 24, 36, 50, 75, 100]),
    1:  ('buck',  [208, 240, 280, 320, 360, 400]),
    2:  ('buck',  [220, 250, 290, 330, 380]),
    3:  ('buck',  [230, 270, 310, 360, 400]),
    4:  ('buck',  [208, 230, 250, 277]),
    5:  ('boost', [12, 15, 18, 21, 24]),
    6:  ('boost', [9, 10, 11, 12]),
    7:  ('boost', [5, 6, 7, 8, 9]),
    8:  ('boost', [12, 16, 20, 24]),
    9:  ('sepic', [9, 12, 18, 24, 30, 36]),
    10: ('sepic', [208, 280, 350, 420, 480]),
    11: ('sepic', [12, 24, 30, 36, 42, 48]),
    12: ('sepic', [100, 150, 200, 240, 277]),
}

# Sweep dimensions
PERIODS  = [10e-6, 20e-6]                  # 100 kHz, 50 kHz
LS_BUCK  = [100e-6, 220e-6]                # buck L
CS_BUCK  = [220e-6, 470e-6]                # buck C
LS_BOOST = [100e-6, 220e-6]                # boost L
CS_BOOST = [100e-6, 220e-6]                # boost C
LS_SEPIC = [100e-6, 220e-6]                # SEPIC L1 = L2
CC_SEPIC = [22e-6, 47e-6]                  # SEPIC coupling C


#  Generate per constraint 

def generate_for_constraint(idx: int, constraint: dict) -> list[dict]:
    """Generate all template-driven netlist candidates for one constraint.

    Args:
        idx (int): Constraint index used to look up the topology class and
            Vin sweep from ``PRESETS``.
        constraint (dict): Constraint specification; must have
            ``"vout_target"`` and ``"power_in"`` keys.

    Returns:
        list[dict]: Candidate dicts with ``"net"`` (SPICE netlist string)
            and ``"tag"`` (human-readable variant identifier).
    """
    topo_class, vins = PRESETS[idx]
    vout = constraint['vout_target']
    p    = constraint['power_in']
    out: list[dict] = []

    if topo_class == 'buck':
        # async-buck sweep
        for vin in vins:
            for T in PERIODS:
                for L in LS_BUCK:
                    for C in CS_BUCK:
                        out.append({
                            "net": buck_async(vin, vout, p, T, L, C),
                            "tag": f"async-{n(vin)}V-T{int(T*1e6)}u-L{int(L*1e6)}u-C{int(C*1e6)}u",
                        })
        # 4 sync variants (only at min and median Vin)
        for vin in [vins[0], vins[len(vins)//2]]:
            for L, C in [(100e-6, 220e-6), (220e-6, 470e-6)]:
                out.append({
                    "net": buck_sync(vin, vout, p, 10e-6, L, C),
                    "tag": f"sync-{n(vin)}V-L{int(L*1e6)}u-C{int(C*1e6)}u",
                })
        # 4 interleaved variants
        for vin in [vins[0], vins[len(vins)//2]]:
            for L, C in [(100e-6, 220e-6), (220e-6, 470e-6)]:
                out.append({
                    "net": buck_interleaved(vin, vout, p, 10e-6, L, C),
                    "tag": f"intlv-{n(vin)}V-L{int(L*1e6)}u-C{int(C*1e6)}u",
                })

    elif topo_class == 'boost':
        for vin in vins:
            for T in PERIODS:
                for L in LS_BOOST:
                    for C in CS_BOOST:
                        out.append({
                            "net": boost(vin, vout, p, T, L, C),
                            "tag": f"boost-{n(vin)}V-T{int(T*1e6)}u-L{int(L*1e6)}u-C{int(C*1e6)}u",
                        })

    elif topo_class == 'sepic':
        for vin in vins:
            for T in PERIODS:
                for L in LS_SEPIC:
                    for Cc in CC_SEPIC:
                        # Use same L1 = L2 (most common SEPIC)
                        Cout = 220e-6 if Cc == 22e-6 else 470e-6
                        out.append({
                            "net": sepic(vin, vout, p, T, L, L, Cc, Cout),
                            "tag": f"sepic-{n(vin)}V-T{int(T*1e6)}u-L{int(L*1e6)}u-Cc{int(Cc*1e6)}u",
                        })

    return out


#  Validator (lazy import) + JSONL writer 

def _run_validator(batch_id: str) -> dict:
    try:
        from pipeline.netlist_validation.validator import validator
    except ImportError as e:
        print(f"  WARN: validator unavailable ({e})")
        return {}
    val = validator()
    val.validate(batch_id)
    p = REPO_ROOT / "pipeline" / "data" / batch_id / "validation_results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    constraints = load_constraints(
        str(REPO_ROOT / "pipeline" / "data" / "datasets" / "constraints.json")
    )
    print(f"[setup] {len(constraints)} constraints loaded")

    # Generate per constraint
    all_groups: dict[int, list[dict]] = defaultdict(list)
    for idx in range(len(constraints)):
        if idx not in PRESETS:
            continue
        samples = generate_for_constraint(idx, constraints[idx])
        all_groups[idx] = samples
        print(f"  idx={idx} ({PRESETS[idx][0]}):  {len(samples)} candidates queued")

    total_candidates = sum(len(g) for g in all_groups.values())
    print(f"[setup] {total_candidates} total candidate netlists")

    # Write + validate per batch
    pairs: list[dict] = []
    total_passed = 0
    for idx, group in sorted(all_groups.items()):
        constraint = constraints[idx]
        batch_id = f"batch_sft_expanded_idx{idx}"
        netlists = [g["net"] for g in group]
        written = write_netlists(netlists=netlists, batchID=batch_id)

        val_results = _run_validator(batch_id)
        n_passed = sum(1 for v in val_results.values() if v.get("passed"))
        total_passed += n_passed
        print(f"\n idx={idx}  candidates={len(written)}  passed={n_passed}  ")

        prompt = make_prompt(constraint)
        for path, sample in zip(written, group):
            info = val_results.get(path.stem) or val_results.get(path.name) or {}
            if not info.get("passed"):
                continue
            pairs.append({
                "prompt":     prompt,
                "completion": path.read_text(encoding="utf-8"),
                "fitness":    None,
                "source":     f"{batch_id}/{path.stem}",
                "constraint": {k: v for k, v in constraint.items() if not k.startswith("_")},
                "generator":  "claude-template-expansion",
                "tag":        sample["tag"],
            })

    out_path = REPO_ROOT / "data" / "sft" / "sft_from_template.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for r in pairs:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print(f"[done] generated {total_candidates}, validator-passed {total_passed} "
          f"({100*total_passed/max(total_candidates,1):.1f}%)")
    print(f"[done] {len(pairs)} pairs written to {out_path}")


if __name__ == "__main__":
    main()
