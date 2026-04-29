"""
net_writer.py — .net file creation module.

Writes each candidate topology to its OWN .net file (one topology per file).
Each file contains a header (constraint metadata + candidate index) followed
by the SPICE netlist body.

Public API
----------
- write_single_netlist(path, netlist, constraint, candidate_idx, total)
      Write ONE netlist to ONE .net file.
- write_netlists(out_dir, netlists, constraint, label)
      Batch helper: write N netlists to N files in out_dir.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

# pipeline/llm_topology_generation/ -> pipeline/
_PIPELINE_ROOT = Path(__file__).parent.parent


def get_llm_output_dir(batchID: str) -> Path:
    """Return the canonical output path for a batch: data/<batchID>/llm_output/"""
    return _PIPELINE_ROOT / "data" / batchID / "llm_output"


def _format_header(
    constraint: dict,
    candidate_idx: int,
    total_candidates: int,
) -> list[str]:
    """Build the comment header lines for a single .net file."""
    lines = [
        f"* Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"* Constraint: {constraint.get('_comment', 'n/a')}",
    ]
    for k, v in constraint.items():
        if k.startswith("_"):
            continue
        lines.append(f"*   {k} = {v}")
    lines.append(f"* Candidate: {candidate_idx} of {total_candidates}")
    lines.append("*")
    return lines


def write_single_netlist(
    path: str | Path,
    netlist: str,
    constraint: dict,
    candidate_idx: int = 1,
    total_candidates: int = 1,
) -> Path:
    """Write ONE candidate netlist to ONE .net file.

    The file contains a comment header (constraint metadata + candidate
    index) followed by the netlist body and a trailing newline.

    Args:
        path:             Target file path.
        netlist:          The SPICE netlist text (already cleaned).
        constraint:       The original constraint dict (used for header).
        candidate_idx:    1-based index of this candidate.
        total_candidates: Total number of candidates in the same batch
                          (used in the header for traceability).

    Returns:
        The absolute path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = _format_header(constraint, candidate_idx, total_candidates)
    lines.append(netlist.strip())
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out.resolve()


def write_netlists(
    netlists: list[str],
    constraint: dict,
    label: str,
    batchID: str | None = None,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Write each candidate to its own .net file inside data/<batchID>/llm_output/.

    Either `batchID` or `out_dir` must be provided. If both are given,
    `batchID` takes precedence.

    File naming:  <label>_cand1.net, <label>_cand2.net, ...

    Args:
        netlists:   list of cleaned netlist strings (one per candidate).
        constraint: Constraint dict (passed through to the header).
        label:      Filename prefix (typically prompt_input.slug() output).
        batchID:    Batch identifier — resolves to data/<batchID>/llm_output/.
        out_dir:    Explicit output directory override (used if batchID is None).

    Returns:
        List of absolute paths to the written files (in order).
    """
    if batchID is not None:
        out = get_llm_output_dir(batchID)
    elif out_dir is not None:
        out = Path(out_dir)
    else:
        raise ValueError("Either batchID or out_dir must be provided.")

    out.mkdir(parents=True, exist_ok=True)

    n = len(netlists)
    written: list[Path] = []
    for i, nl in enumerate(netlists, start=1):
        file_path = out / f"{label}_cand{i}.net"
        write_single_netlist(
            file_path, nl, constraint,
            candidate_idx=i, total_candidates=n,
        )
        written.append(file_path.resolve())
    return written


if __name__ == "__main__":
    # Quick self-test
    demo_nets = ["* dummy A\n.end", "* dummy B\n.end", "* dummy C\n.end"]
    paths = write_netlists(
        out_dir="demo_out",
        netlists=demo_nets,
        constraint={"_comment": "Test", "vin_min": 12, "vout_target": 5},
        label="00_Test",
    )
    for p in paths:
        print(f"Wrote: {p}")
        print(p.read_text())
        print("-" * 40)