"""
prompt_input.py — Constraint → prompt template module.

Reads a constraint JSON file and converts each entry into an LLM prompt.
The prompt explicitly enforces the SPICE naming convention used by the
downstream evaluator/simulator, so even the un-fine-tuned base model has
a structural target to aim at.

Public API
----------
- load_constraints(json_path)       → list[dict]
- load_constraint(json_path, idx)   → dict   (single constraint by index)
- make_prompt(constraint)           → str    (full prompt with naming rules)
"""

from __future__ import annotations

import json
from pathlib import Path


# ── Naming-convention preamble (injected into every prompt) ──────────────
#
# These rules MUST stay in sync with the netlist-evaluator regex used by
# `pipeline/netlist_validation/validator.py` and the SFT/GRPO reward
# functions. Do not rename a token here without updating those.

NAMING_RULES = """### Naming Convention (MUST follow exactly):
- Input voltage source   : Vin  (e.g. `Vin in 0 12`)
- Load resistor          : Rload  (e.g. `Rload out 0 10`)
- Gate drive source      : Vgate / Vgate1..N — PULSE with exactly 7 parameters
                           (e.g. `Vgate gate 0 PULSE(0 12 0 1n 1n <t_on> <period>)`)
- MOSFET pin order       : M<name> drain gate source bulk model
    High-side : drain=in,  gate=gate,  source=sw, bulk=0
    Low-side  : drain=sw,  gate=gate2, source=0,  bulk=0
- MOSFET model           : .model NMOS NMOS(Vto=1 Kp=2 Lambda=0)  — exactly this, no variations
- Diode model            : .model DIODE D  — exactly this line, required whenever any D* component is used
- Component prefixes     : V (voltage sources: Vin / Vgate / Vgate1..N),
                           M (MOSFETs), D (diodes), L (inductors), C (capacitors),
                           R (resistors: Rload for the load, Rbleed/Rbleed1..N for floating nodes)
- Required nodes         : `in` (Vin positive), `out` (Rload positive),
                           `sw` / `sw1, sw2, ...` (MOSFET source),
                           `gate` / `gate1, gate2, ...` (gate drive), `0` (GND)
- Floating nodes         : any node connected only to reactive elements needs
                           `Rbleed <node> 0 1Meg`
- Output ONLY the raw SPICE netlist — no Markdown fences, no explanation.
"""


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
    if idx >= len(constraints) or idx < 0:
        raise IndexError(
            f"Index {idx} out of range — file has {len(constraints)} constraints"
        )
    return constraints[idx]


def make_prompt(constraint: dict) -> str:
    """Build the full prompt for one constraint dict.

    Layout::

        ### Naming Convention (MUST follow exactly):
        - <rules ...>

        ### Constraint:
        { ...JSON... }

        ### SPICE Netlist:

    Any field whose key starts with `_` (e.g. `_comment`) is filtered out
    before the JSON is serialized — those are bookkeeping fields used for
    filenames and reports, never shown to the model.
    """
    payload = {k: v for k, v in constraint.items() if not k.startswith("_")}
    return (
        f"{NAMING_RULES}\n"
        f"### Constraint:\n{json.dumps(payload, indent=2)}\n\n"
        f"### SPICE Netlist:\n"
    )


if __name__ == "__main__":
    # Quick self-test — prints the prompt for the first 2 constraints in
    # the canonical dataset.
    import sys

    default_path = "pipeline/data/datasets/constraints.json"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    items = load_constraints(path)
    print(f"Loaded {len(items)} constraints from {path}\n")
    for i, c in enumerate(items[:2]):
        print(f"--- Constraint #{i}: {c.get('_comment', 'n/a')} ---")
        print(make_prompt(c))
