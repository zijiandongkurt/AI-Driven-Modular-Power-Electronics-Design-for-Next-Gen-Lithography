"""
net_writer.py — .net file creation module.

Each .net file contains a header (constraint metadata) plus 4 candidate
netlists separated by `* === Candidate N ===` banners.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def write_net_file(
    path: str | Path,
    netlists: list[str],
    constraint: dict,
) -> Path:
    """Write a list of candidate netlists to a single .net file.

    Layout:
        * Generated: <timestamp>
        * Constraint: <comment>
        * <full constraint JSON>
        *
        * === Candidate 1 ===
        <netlist 1>
        * === Candidate 2 ===
        <netlist 2>
        ...
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"* Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"* Constraint: {constraint.get('_comment', 'n/a')}")
    for k, v in constraint.items():
        if k.startswith("_"):
            continue
        lines.append(f"*   {k} = {v}")
    lines.append(f"* Total candidates: {len(netlists)}")
    lines.append("*")

    for i, nl in enumerate(netlists, start=1):
        lines.append("")
        lines.append(f"* === Candidate {i} ===")
        lines.append(nl.strip())

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    # Quick self-test
    demo = ["* dummy netlist 1\n.end", "* dummy netlist 2\n.end"]
    p = write_net_file(
        "demo_out/test.net",
        demo,
        {"_comment": "Test", "vin_min": 12, "vout_target": 5},
    )
    print(f"Wrote: {p}")
    print(p.read_text())
