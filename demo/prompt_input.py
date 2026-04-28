"""
prompt_input.py — Auto prompt input module.

Reads a constraint JSON file and converts each entry into an LLM prompt.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_constraints(json_path: str | Path) -> list[dict]:
    """Load a list of constraint dicts from a JSON file."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return data


def make_prompt(constraint: dict) -> str:
    """Build a minimal prompt from a constraint dict.

    The constraint is dumped as JSON inside a `### Constraint:` block.
    No naming convention rules are injected here — this is the
    'pre-fine-tuning' baseline prompt.
    """
    # Strip the comment field from the JSON shown to the model
    payload = {k: v for k, v in constraint.items() if not k.startswith("_")}
    return (
        f"### Constraint:\n{json.dumps(payload, indent=2)}\n\n"
        f"### SPICE Netlist:\n"
    )


def slug(constraint: dict, idx: int) -> str:
    """Generate a short filename slug from the constraint comment."""
    comment = constraint.get("_comment", f"constraint_{idx:02d}")
    # Replace anything non-alphanumeric with underscore
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
