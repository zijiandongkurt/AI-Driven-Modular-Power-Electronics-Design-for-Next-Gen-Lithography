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


# ── Naming-convention preamble (injected into every prompt) ──────────────
#
# These rules MUST stay in sync with the netlist-evaluator regex used by
# `pipeline/llm_topology_generation/netlist_filter.py` and the SFT/GRPO
# reward functions. Do not rename a token here without updating those.

NAMING_RULES = """### Naming Convention (MUST follow exactly):
- The input voltage source MUST be named `Vin` (e.g. `Vin in 0 12`).
- The output load resistor MUST be named `Rload` (e.g. `Rload out 0 10`).
- Any MOSFET gate-driver pulse source MUST be named `Vgate`.
- Inductors MUST start with `L` (e.g. `L1`), capacitors with `C`, diodes with `D`.
- For every MOSFET used, include its model card:
    `.model NMOS NMOS` and/or `.model PMOS PMOS`.
- Include exactly ONE transient analysis directive: `.tran <step> <stop>`.
- The netlist MUST end with `.end` on its own line.
- Output ONLY the SPICE netlist — no Markdown fences, no commentary.
"""


def load_constraints(json_path: str | Path) -> list[dict]:
    """Load a list of constraint dicts from a JSON file."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return data


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
