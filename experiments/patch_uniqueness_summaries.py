import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))
from pipeline.utility.topology_hasher import get_topological_hash

def patch_summaries(zycos_folder: str):
    zycos_dir = Path(zycos_folder)
    run_folders = [d for d in zycos_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")]
    
    master_total = 0
    master_unique_hashes = set()
    
    for run_dir in run_folders:
        summary_path = run_dir / "results" / "run_summary.txt"
        if not summary_path.exists(): continue
        
        text = summary_path.read_text(encoding="utf-8")
        
        # Calculate ground truth for this run
        run_hashes = set()
        run_total = 0
        
        for batch_dir in run_dir.glob("batch_*"):
            for net_file in (batch_dir / "LLM_output").glob("*.net"):
                run_total += 1
                run_hashes.add(get_topological_hash(net_file.read_text(encoding="utf-8")))
                master_unique_hashes.add(get_topological_hash(net_file.read_text(encoding="utf-8")))
        
        master_total += run_total
        run_unique = len(run_hashes)
        run_dups = run_total - run_unique
        run_u_rate = (run_unique / run_total * 100) if run_total > 0 else 0
        
        # Patch the local run_summary.txt
        # Look for the section after Invalid: XXX
        pattern = re.compile(r"(Invalid:\s*\d+\nTotal candidates in history:\s*\d+\n)")
        
        if "Unique Topologies:" not in text and pattern.search(text):
            injection = f"\nUnique Topologies: {run_unique}\nExact Duplicates:  {run_dups}\nUniqueness Rate:   {run_u_rate:.1f}%\n"
            new_text = pattern.sub(rf"\1{injection}", text)
            summary_path.write_text(new_text, encoding="utf-8")
            print(f" Patched {run_dir.name} -> Unique: {run_unique} ({run_u_rate:.1f}%)")

    # Patch the master training_run_summary.txt
    master_summary_path = zycos_dir / "training_run_summary.txt"
    if master_summary_path.exists():
        text = master_summary_path.read_text(encoding="utf-8")
        
        master_unique = len(master_unique_hashes)
        master_dups = master_total - master_unique
        master_u_rate = (master_unique / master_total * 100) if master_total > 0 else 0
        
        pattern = re.compile(r"(Total Invalid:\s*\d+\nGlobal Validity Rate:\s*[\d.]+%\n)")
        if "Global Uniqueness Rate:" not in text and pattern.search(text):
            injection = f"Total Unique Topologies: {master_unique}\nGlobal Exact Duplicates: {master_dups}\nGlobal Uniqueness Rate: {master_u_rate:.2f}%\n"
            new_text = pattern.sub(rf"\1{injection}", text)
            master_summary_path.write_text(new_text, encoding="utf-8")
            print(f" Patched Master Summary -> Global Unique: {master_unique} ({master_u_rate:.1f}%)")

if __name__ == "__main__":
    TARGET_FOLDER = PROJECT_ROOT / "pipeline" / "data" / "zycos_006"
    patch_summaries(str(TARGET_FOLDER))