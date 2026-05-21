"""
prompt_input.py — Constraint → prompt template module.

Reads a constraint JSON file and converts each entry into an LLM prompt.
The prompt explicitly enforces the SPICE naming convention used by the
downstream evaluator/simulator, so even the un-fine-tuned base model has
a structural target to aim at.

Public API
----------
- load_constraints(json_path)  → list[dict]
- make_prompt(constraint)      → str   (full prompt with naming rules)
- slug(constraint, idx)        → str   (filename-safe label)
"""

from __future__ import annotations

import json
from pathlib import Path


# ── System prompt ────────────────────────────────────────────────────────
#
# Defines the model's role, allowed components, fixed rules, required nodes,
# and output format. Injected into every prompt.
#
# MUST stay in sync with the netlist validator and reward function.
# Do not rename nodes or component prefixes without updating those too.

SYSTEM_PROMPT = """You are an AI electrical engineer that designs non-isolated DC/DC converters.
Given a set of design constraints, output a single SPICE netlist that attempts to meet them.

ALLOWED COMPONENTS:
  Voltage sources : Vin (input), Vgate / Vgate1..N (gate drives)
  MOSFETs         : M1, M2, ...   — switching elements
  Diodes          : D1, D2, ...   — freewheeling or rectification
  Inductors       : L1, L2, ...   — energy storage
  Capacitors      : C1, C2, ...   — filtering or energy transfer
  Resistors       : Rload (mandatory load), Rbleed / Rbleed1..N (floating nodes only)

FIXED RULES — do not deviate:
  - Input source   : Vin in 0 <vin>
  - Load resistor  : Rload out 0 <value>
  - MOSFET model   : .model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
  - MOSFET pins    : M<n> drain gate source bulk NMOS
      High-side    : drain=in,  gate=gate,  source=sw,  bulk=0
      Low-side     : drain=sw,  gate=gate2, source=0,   bulk=0
  - Gate drive     : Vgate gate 0 PULSE(0 12 0 1n 1n <t_on> <period>)
                     — exactly 7 parameters, choose t_on and period to meet switching frequency
  - Floating nodes : any node connected only to reactive elements needs Rbleed <node> 0 1Meg
  - Simulation     : .tran 1n 5m

REQUIRED NODES:
  in   — positive terminal of Vin
  out  — output node, positive terminal of Rload
  sw   — switch node (use sw1, sw2, ... for multiple switches)
  gate — gate drive node (use gate1, gate2, ... for multiple switches)
  0    — GND reference

OUTPUT FORMAT:
  Output raw SPICE netlist text only.
  No markdown fences, no explanation, no prose.
  Start with an optional single-line title comment (* <title>).
  End the file with exactly: .end
  DO NOT write any words on the same line as .end
  DO NOT output a single character after .end
  NO EXPLANATIONS, NO PROSE, NO CHAT.
  NEVER EXPLAIN YOURSELF. JUST RAW SPICE NETLIST TEXT.
  
  CRITICAL: Do not include design notes or explanations in SPICE comments.
  CRITICAL RULE: DO NOT copy component values or topologies directly from the examples below. You MUST calculate new component values (Vin, Rload, inductor/capacitor sizing, and Vgate pulse timing) and select the correct topology to satisfy the exact constraints given to you.

=== EXAMPLES OF PERFECT RESPONSES ===

--- START EXAMPLE 1 (Step-Down / Buck) ---
### SPICE Netlist:
* 12V to 5V Buck Converter
Vin in 0 12
M1 in gate sw 0 NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out 47u
C1 out 0 220u
Rload out 0 0.278
Vgate gate 0 PULSE(0 12 0 1n 1n 4.16u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1u 10m
.end
--- END EXAMPLE 1 ---

--- START EXAMPLE 2 (Step-Up / Boost) ---
### SPICE Netlist:
* 5V to 12V Boost Converter
Vin in 0 5
L1 in sw 22u
M1 sw gate 0 0 NMOS W=1 L=1
D1 sw out DIODE
C1 out 0 100u
Rload out 0 6
Vgate gate 0 PULSE(0 12 0 1n 1n 5.8u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1u 10m
.end
--- END EXAMPLE 2 ---
"""

# Kept for backward compatibility with llm_engine_minimal.py Constraint.to_prompt()
NAMING_RULES = SYSTEM_PROMPT


def load_constraints(json_path: str | Path) -> list[dict]:
    """Load a list of constraint dicts from a JSON file."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return data


def load_constraint(json_path: str | Path, idx: int = 0) -> dict:
    """Load a single constraint dict from a JSON file by index.

    Args:
        json_path: Path to a JSON file containing a list of constraint dicts.
        idx:       Index of the constraint to return (default: 0).

    Returns:
        A single constraint dict.
    """
    constraints = load_constraints(json_path)
    if idx >= len(constraints):
        raise IndexError(f"Index {idx} out of range — file has {len(constraints)} constraints")
    return constraints[idx]


def make_prompt(constraint: dict) -> str:
    """Build the full prompt for one constraint dict.

    Layout:
        <SYSTEM_PROMPT>

        ### Constraint:
        { ...JSON... }

        ### SPICE Netlist:

    Any field whose key starts with `_` (e.g. `_comment`) is filtered out
    before the JSON is serialized — those are bookkeeping fields used for
    filenames and reports, never shown to the model.
    """
    payload = {k: v for k, v in constraint.items() if not k.startswith("_")}
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Constraint:\n{json.dumps(payload, indent=2)}\n\n"
        f"### SPICE Netlist:\n"
    )


def slug(constraint: dict, idx: int) -> str:
    """Generate a short filename slug from the constraint comment."""
    comment = constraint.get("_comment", f"constraint_{idx:02d}")
    safe = "".join(c if c.isalnum() else "_" for c in comment)
    safe = "_".join(filter(None, safe.split("_")))  # collapse repeats
    return f"{idx:02d}_{safe[:50]}"


if __name__ == "__main__":
    # Quick self-test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_constraints.json"
    items = load_constraints(path)
    print(f"Loaded {len(items)} constraints from {path}")
    for i, c in enumerate(items[:2]):
        print(f"\n--- Slug: {slug(c, i)} ---")
        print(make_prompt(c))