"""
net_writer.py — .net file creation module.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re  # NEW: moved import to top-level

_PIPELINE_ROOT = Path(__file__).parent.parent


def get_llm_output_dir(batchID: str) -> Path:
    """Return canonical output path: pipeline/data/<batchID>/LLM_output/."""
    return _PIPELINE_ROOT / "data" / batchID / "LLM_output"


def _format_header(
    constraint: dict,
    candidate_idx: int,
    total_candidates: int,
    custom_name: str | None = None,  # NEW: record grouped identity in header
) -> list[str]:
    """Build comment header lines for a single .net file."""
    lines = [
        f"* Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"* Constraint: {constraint.get('_comment', 'n/a')}",
    ]

    for k, v in constraint.items():
        if k.startswith("_"):
            continue
        lines.append(f"*   {k} = {v}")

    lines.append(f"* Candidate: {candidate_idx} of {total_candidates}")

    # NEW: useful for grouped GRPO debugging.
    if custom_name is not None:
        lines.append(f"* Grouped name: {custom_name}")

    lines.append("*")
    return lines


def write_single_netlist(
    path: str | Path,
    netlist: str,
    constraint: dict,
    candidate_idx: int = 1,
    total_candidates: int = 1,
    custom_name: str | None = None,  # NEW: pass grouped name into header
) -> Path:
    """Write one candidate netlist to one .net file.

    Args:
        path (str | Path): Destination file path.
        netlist (str): SPICE netlist text to write.
        constraint (dict): Constraint dict written into the comment header.
        candidate_idx (int): 1-based index of this candidate.
        total_candidates (int): Total number of candidates in the batch.
        custom_name (str | None): Optional grouped name recorded in the header.

    Returns:
        Path: Resolved absolute path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = _format_header(
        constraint=constraint,
        candidate_idx=candidate_idx,
        total_candidates=total_candidates,
        custom_name=custom_name,  # NEW
    )

    lines.append(netlist.strip())
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out.resolve()


def write_netlists(
    netlists: list[str],
    constraint: dict,
    label: str,
    batchID: str | None = None,
    out_dir: str | Path | None = None,
    custom_names: list[str] | None = None,  # NEW: supports g1_cand1 naming
) -> list[Path]:
    """Write each candidate netlist to its own .net file.

    Args:
        netlists (list[str]): SPICE netlist texts, one per candidate.
        constraint (dict): Constraint dict written into each comment header.
        label (str): Filename slug prefix (e.g. ``"00_Buck_12V"``).
        batchID (str | None): Batch identifier; determines output directory
            when provided.  Mutually exclusive with ``out_dir``.
        out_dir (str | Path | None): Explicit output directory; used when
            ``batchID`` is not given.
        custom_names (list[str] | None): Per-candidate name suffixes for
            grouped GRPO filenames (e.g. ``["g1_cand1", "g1_cand2"]``).
            Must have the same length as ``netlists`` when provided.

    Returns:
        list[Path]: Resolved absolute paths of all written files.

    Raises:
        ValueError: If neither ``batchID`` nor ``out_dir`` is provided, or if
            ``custom_names`` length does not match ``netlists``.
    """
    if batchID is not None:
        out = get_llm_output_dir(batchID)
    elif out_dir is not None:
        out = Path(out_dir)
    else:
        raise ValueError("Either batchID or out_dir must be provided.")

    out.mkdir(parents=True, exist_ok=True)

    batch_suffix = ""
    if batchID is not None:
        match = re.search(r"batch_(\d+)", str(batchID))
        if match:
            batch_suffix = f"_b{match.group(1)}"

    n = len(netlists)

    # NEW: protect grouped GRPO naming alignment.
    if custom_names is not None and len(custom_names) != n:
        raise ValueError("custom_names must have the same length as netlists.")

    written: list[Path] = []

    for i, nl in enumerate(netlists, start=1):
        custom_name = custom_names[i - 1] if custom_names is not None else None

        # CHANGED: grouped GRPO filenames become label_g1_cand1_b2.net.
        if custom_name is not None:
            file_path = out / f"{label}_{custom_name}{batch_suffix}.net"
        else:
            file_path = out / f"{label}_cand{i}{batch_suffix}.net"

        write_single_netlist(
            path=file_path,
            netlist=nl,
            constraint=constraint,
            candidate_idx=i,
            total_candidates=n,
            custom_name=custom_name,  # NEW
        )

        written.append(file_path.resolve())

    return written


if __name__ == "__main__":
    demo_nets = ["* dummy A\n.end", "* dummy B\n.end"]

    paths = write_netlists(
        out_dir="demo_out",
        netlists=demo_nets,
        constraint={"_comment": "Test", "vin_min": 12, "vout_target": 5},
        label="00_Test",
        custom_names=["g1_cand1", "g1_cand2"],  # NEW: grouped test
    )

    for p in paths:
        print(f"Wrote: {p}")
        print(p.read_text())
        print("-" * 40)