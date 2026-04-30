"""
net_writer.py — .net file creation module.

Writes each candidate topology to its OWN .net file under
``pipeline/data/<batchID>/llm_output/topN.net``. Files are pure SPICE
(no comment header) to match the convention used by the rest of the
pipeline (validator + simulator + reward evaluator).

Public API
----------
- get_llm_output_dir(batchID)
      Resolve the canonical output directory for a batch.
- write_netlists(netlists, batchID=..., out_dir=..., start_index=1)
      Write N netlists into the canonical batch directory as
      ``top1.net``, ``top2.net``, ...
- write_single_netlist(path, netlist)
      Write ONE netlist to ONE explicit .net file path.
"""

from __future__ import annotations

from pathlib import Path

# pipeline/llm_topology_generation/ -> pipeline/
_PIPELINE_ROOT = Path(__file__).parent.parent


def get_llm_output_dir(batchID: str) -> Path:
    """Return the canonical output path for a batch.

    Layout: ``pipeline/data/<batchID>/llm_output/`` (folder name is
    intentionally lowercase, matching the validator and simulator).
    """
    return _PIPELINE_ROOT / "data" / batchID / "llm_output"


def write_single_netlist(path: str | Path, netlist: str) -> Path:
    """Write ONE netlist to ONE .net file (pure SPICE, no header).

    Args:
        path:    Target file path.
        netlist: The SPICE netlist text (already cleaned).

    Returns:
        The absolute path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(netlist.strip() + "\n", encoding="utf-8")
    return out.resolve()


def write_netlists(
    netlists: list[str],
    *,
    batchID: str | None = None,
    out_dir: str | Path | None = None,
    start_index: int = 1,
) -> list[Path]:
    """Write each candidate to its own ``topN.net`` inside the batch folder.

    Either ``batchID`` or ``out_dir`` must be provided. If both are given,
    ``batchID`` takes precedence and resolves to
    ``pipeline/data/<batchID>/llm_output/``.

    Empty / whitespace-only netlists are skipped with a warning so an
    LLM hiccup never produces a 0-byte ``.net`` that would crash the
    validator downstream. Numbering stays contiguous in the produced
    files (so a skipped candidate does NOT leave a gap on disk).

    File naming: ``top<start_index>.net``, ``top<start_index+1>.net``, ...
    Files are pure SPICE — no comment header.

    Args:
        netlists:    List of cleaned netlist strings (one per candidate).
        batchID:     Batch identifier — resolves to the canonical path.
        out_dir:     Explicit override for the output directory.
        start_index: 1-based starting index for the ``topN`` naming
                     (use this if you want to append to an existing batch).

    Returns:
        List of absolute paths to the written files, in order.
    """
    import warnings

    if batchID is not None:
        out = get_llm_output_dir(batchID)
    elif out_dir is not None:
        out = Path(out_dir)
    else:
        raise ValueError("Either batchID or out_dir must be provided.")

    out.mkdir(parents=True, exist_ok=True)

    # Drop empty/whitespace-only candidates up-front so numbering stays
    # contiguous on disk.
    cleaned: list[str] = []
    skipped = 0
    for i, nl in enumerate(netlists):
        if nl is None or not nl.strip():
            skipped += 1
            continue
        cleaned.append(nl)
    if skipped:
        warnings.warn(
            f"write_netlists: skipped {skipped}/{len(netlists)} empty netlist(s)",
            RuntimeWarning,
            stacklevel=2,
        )

    written: list[Path] = []
    for offset, nl in enumerate(cleaned):
        file_path = out / f"top{start_index + offset}.net"
        write_single_netlist(file_path, nl)
        written.append(file_path.resolve())
    return written


if __name__ == "__main__":
    # Quick self-test
    demo_nets = ["Vin in 0 12\n.end", "Vin in 0 24\n.end"]
    paths = write_netlists(netlists=demo_nets, out_dir="demo_out")
    for p in paths:
        print(f"Wrote: {p}")
        print(p.read_text())
        print("-" * 40)