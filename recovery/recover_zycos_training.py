import os
import sys
import re
import shutil
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def fix_netlist_content(content: str) -> str:
    """The 3-Pass Physics Fixer (Bulk, Floating Gate, .tran normalization)"""
    lines = content.split('\n')
    gate_to_source_map = {}
    fixed_lines_pass1 = []
    
    # PASS 1: Map MOSFETs and fix Bulk connections
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].upper().startswith('M'):
            gate_node, source_node, bulk_node = parts[2], parts[3], parts[4]
            gate_to_source_map[gate_node] = source_node
            if bulk_node != source_node:
                parts[4] = source_node
            leading_space = line[:len(line) - len(line.lstrip())]
            fixed_lines_pass1.append(leading_space + " ".join(parts))
        else:
            fixed_lines_pass1.append(line)
            
    # PASS 2 & 3: Fix Gate Drives & Normalize .tran
    final_lines = []
    for line in fixed_lines_pass1:
        parts = line.strip().split(maxsplit=3)
        if not parts:
            final_lines.append(line)
            continue
            
        if len(parts) >= 4 and parts[0].upper().startswith('V') and 'PULSE' in line.upper():
            name, pos_node, neg_node, rest = parts[0], parts[1], parts[2], parts[3]
            if pos_node in gate_to_source_map:
                expected_ref = gate_to_source_map[pos_node]
                if neg_node != expected_ref:
                    parts[2] = expected_ref
                    leading_space = line[:len(line) - len(line.lstrip())]
                    line = leading_space + f"{name} {pos_node} {parts[2]} {rest}"
                    
        elif parts[0].upper() == '.TRAN':
            leading_space = line[:len(line) - len(line.lstrip())]
            line = leading_space + ".tran 10n 1m"
            
        final_lines.append(line)
    return '\n'.join(final_lines)

def main():
    legacy_dir = PROJECT_ROOT / "pipeline" / "data_legacy_broken"
    clean_dir = PROJECT_ROOT / "pipeline" / "data"

    zycos_folders = [d for d in legacy_dir.iterdir() if d.is_dir() and d.name.startswith("zycos_")]
    zycos_folders = sorted(zycos_folders, key=lambda x: x.name)

    if not zycos_folders:
        print("❌ No zycos_XXX folders found in legacy data.")
        return

    total_netlists = 0
    total_runs = 0

    print(f"\n{'='*70}\n Catching up {len(zycos_folders)} Zycos Training Folders...\n{'='*70}")

    for zycos_dir in zycos_folders:
        run_folders = [d for d in zycos_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")]
        
        for run_dir in run_folders:
            new_run_dir = clean_dir / zycos_dir.name / run_dir.name
            new_run_dir.mkdir(parents=True, exist_ok=True)
            
            # --- 1. DEEP SEARCH RECOVER CONSTRAINTS ---
            batch_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                                   key=lambda x: int(x.name.split('_')[1]))
            
            for b_dir in batch_folders:
                legacy_reward_file = b_dir / "reward_results.json"
                if legacy_reward_file.exists():
                    try:
                        with open(legacy_reward_file, "r", encoding="utf-8") as f:
                            legacy_data = json.load(f)
                        constraints = legacy_data.get("active_constraints", {})
                        if constraints:
                            weights = legacy_data.get("weights_used", legacy_data.get("weights", None))
                            rec_pkg = {"active_constraints": constraints}
                            if weights: rec_pkg["weights"] = weights
                            
                            with open(new_run_dir / "active_constraint.json", "w", encoding="utf-8") as f:
                                json.dump(rec_pkg, f, indent=4)
                            break # Found it, stop searching this run
                    except Exception:
                        pass

            # --- 2. RESTRUCTURE & FIX BATCHES ---
            for batch_dir in batch_folders:
                legacy_raw = batch_dir / "raw_output.txt"
                if not legacy_raw.exists(): continue

                new_batch_dir = new_run_dir / batch_dir.name
                new_batch_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_raw, new_batch_dir / "raw_output.txt")

                new_llm_out = new_batch_dir / "LLM_output"
                new_llm_out.mkdir(parents=True, exist_ok=True)

                text = legacy_raw.read_text(encoding="utf-8")
                pattern = re.compile(r"={10,}\n\s*CANDIDATE (\d+)\s*\n={10,}\n")
                parts = pattern.split(text)
                
                if len(parts) > 1:
                    for i in range(1, len(parts), 2):
                        cand_id = parts[i]
                        cand_text = parts[i+1].strip()
                        if cand_text:
                            fixed = fix_netlist_content(cand_text)
                            (new_llm_out / f"cand_{cand_id}.net").write_text(fixed, encoding="utf-8")
                            total_netlists += 1
            total_runs += 1
            print(f" ✔ Processed {zycos_dir.name}/{run_dir.name}")

    print(f"\n{'='*70}\n Caught up {total_runs} training runs and fixed {total_netlists} netlists!\n{'='*70}")

if __name__ == "__main__":
    main()