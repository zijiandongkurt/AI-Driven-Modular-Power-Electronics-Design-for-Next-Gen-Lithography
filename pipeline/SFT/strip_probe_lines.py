"""
strip_probe_lines.py
====================

Remove `.probe V(*) I(*)` lines from all SFT JSONL files and from
pipeline/data/batch_*/llm_output/*.net files.

Background:
  The validator on `origin/main` was set to inject `.probe V(*) I(*)`
  (an NGSpice/PSpice directive).  The containerized LTspice runner on
  `origin/containerfix` expects no `.probe` — LTspice saves all signals
  by default.  Leaving `.probe` lines in the training set teaches the
  model to emit incompatible directives, breaking downstream LTspice
  simulation.

This script:
  1. Walks every *.jsonl under data/sft/ and rewrites the `completion`
     field with `.probe V(*) I(*)\\n` removed.
  2. Walks every .net under pipeline/data/batch_*/llm_output/ and
     strips the line in-place.

Run from repo root:
    python scripts/strip_probe_lines.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = ".probe V(*) I(*)"


def _strip(text: str) -> tuple[str, bool]:
    """Return (cleaned_text, was_changed)."""
    if TARGET not in text:
        return text, False
    cleaned = (text
               .replace(f"{TARGET}\n", "")     # whole-line removal
               .replace(TARGET, ""))           # trailing case (rare)
    return cleaned, True


def fix_jsonl(path: Path) -> int:
    """Strip .probe from every completion field in a JSONL. Returns # changed."""
    if not path.exists():
        return 0
    rows = []
    n_changed = 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "completion" in obj:
            cleaned, changed = _strip(obj["completion"])
            if changed:
                obj["completion"] = cleaned
                n_changed += 1
        rows.append(obj)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return n_changed


def fix_net_files(root: Path) -> int:
    """Strip .probe from every .net file under root. Returns # changed."""
    n_changed = 0
    for net_path in root.rglob("*.net"):
        text = net_path.read_text(encoding="utf-8")
        cleaned, changed = _strip(text)
        if changed:
            net_path.write_text(cleaned, encoding="utf-8")
            n_changed += 1
    return n_changed


def main():
    print(f"[strip_probe] repo root: {REPO_ROOT}")

    # 1. JSONL files
    sft_dir = REPO_ROOT / "data" / "sft"
    print(f"\n[strip_probe] JSONL files in {sft_dir}:")
    total_jsonl = 0
    for jsonl in sorted(sft_dir.glob("*.jsonl")):
        n = fix_jsonl(jsonl)
        total_jsonl += n
        print(f"  {jsonl.name:<35} stripped from {n} rows")
    print(f"  total: {total_jsonl} JSONL rows cleaned")

    # 2. .net files in pipeline/data/batch_*/llm_output
    data_dir = REPO_ROOT / "pipeline" / "data"
    print(f"\n[strip_probe] .net files under {data_dir}/batch_*/llm_output/:")
    total_net = 0
    for batch_dir in sorted(data_dir.glob("batch_*")):
        n = fix_net_files(batch_dir)
        if n:
            print(f"  {batch_dir.name:<40} stripped {n} files")
        total_net += n
    print(f"  total: {total_net} .net files cleaned")

    print(f"\n[done] {total_jsonl + total_net} files / rows cleaned")


if __name__ == "__main__":
    main()
