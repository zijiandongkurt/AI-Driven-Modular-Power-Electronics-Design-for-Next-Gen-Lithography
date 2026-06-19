import os
import sys
import re
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def fix_netlist_content(content: str) -> str:
    """
    Parses SPICE text to fix MOSFET bulk pins, Gate Drive reference nodes,
    and forcibly normalizes the simulation time.
    """
    lines = content.split('\n')
    gate_to_source_map = {}
    fixed_lines_pass1 = []
    
    # === PASS 1: Map MOSFETs and fix Bulk connections ===
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0].upper().startswith('M'):
            gate_node = parts[2]
            source_node = parts[3]
            bulk_node = parts[4]
            
            gate_to_source_map[gate_node] = source_node
            
            if bulk_node != source_node:
                parts[4] = source_node
                
            leading_space = line[:len(line) - len(line.lstrip())]
            fixed_lines_pass1.append(leading_space + " ".join(parts))
        else:
            fixed_lines_pass1.append(line)
            
    # === PASS 2 & 3: Fix Gate Drives & Normalize .tran ===
    final_lines = []
    for line in fixed_lines_pass1:
        parts = line.strip().split(maxsplit=3)
        if not parts:
            final_lines.append(line)
            continue
            
        # Pass 2: Fix Gate Drive References
        if len(parts) >= 4 and parts[0].upper().startswith('V') and 'PULSE' in line.upper():
            name = parts[0]
            pos_node = parts[1]
            neg_node = parts[2]
            rest = parts[3]
            
            if pos_node in gate_to_source_map:
                expected_ref = gate_to_source_map[pos_node]
                
                if neg_node != expected_ref:
                    parts[2] = expected_ref
                    leading_space = line[:len(line) - len(line.lstrip())]
                    line = leading_space + f"{name} {pos_node} {parts[2]} {rest}"
                    
        # Pass 3: Normalize Simulation Time
        elif parts[0].upper() == '.TRAN':
            leading_space = line[:len(line) - len(line.lstrip())]
            line = leading_space + ".tran 10n 1m"
            
        final_lines.append(line)
        
    return '\n'.join(final_lines)

def process_run_dir(legacy_run_dir: Path, new_data_dir: Path):
    """
    Reads the legacy run, creates the clean mirror in the new data dir,
    and extracts/fixes netlists purely from raw_output.txt.
    """
    new_run_dir = new_data_dir / legacy_run_dir.name
    new_run_dir.mkdir(parents=True, exist_ok=True)
    
    batch_folders = sorted(
        [d for d in legacy_run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
        key=lambda x: int(x.name.split('_')[1])
    )
    
    total_fixed = 0
    for batch_dir in batch_folders:
        legacy_raw_file = batch_dir / "raw_output.txt"
        
        if not legacy_raw_file.exists():
            continue
            
        # 1. Setup new batch folder
        new_batch_dir = new_run_dir / batch_dir.name
        new_batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Copy ONLY the raw_output.txt
        new_raw_file = new_batch_dir / "raw_output.txt"
        shutil.copy2(legacy_raw_file, new_raw_file)
        
        # 3. Setup clean LLM_output folder
        new_llm_output = new_batch_dir / "LLM_output"
        new_llm_output.mkdir(parents=True, exist_ok=True)
        
        # 4. Parse candidates directly from the text
        text = new_raw_file.read_text(encoding="utf-8")
        
        pattern = re.compile(r"={10,}\n\s*CANDIDATE (\d+)\s*\n={10,}\n")
        parts = pattern.split(text)
        
        if len(parts) > 1:
            fixed_count = 0
            for i in range(1, len(parts), 2):
                cand_id = parts[i]
                cand_text = parts[i+1].strip()
                
                if not cand_text:
                    continue
                    
                # Apply physics fixes and .tran normalization
                fixed_content = fix_netlist_content(cand_text)
                
                net_filename = f"cand_{cand_id}.net"
                (new_llm_output / net_filename).write_text(fixed_content, encoding="utf-8")
                fixed_count += 1
                
            if fixed_count > 0:
                print(f"   -> {batch_dir.name}: Extracted and fixed {fixed_count} netlists.")
                total_fixed += fixed_count
                
    if total_fixed > 0:
        print(f" ✔ Successfully rebuilt {legacy_run_dir.name} ({total_fixed} total netlists)")

def main():
    original_data_dir = PROJECT_ROOT / "pipeline" / "data"
    legacy_data_dir = PROJECT_ROOT / "pipeline" / "data_legacy_broken"
    
    # 1. Root-level Rename (The big safety switch)
    if original_data_dir.exists() and not legacy_data_dir.exists():
        print(f"Renaming current 'data' folder to 'data_legacy_broken'...")
        original_data_dir.rename(legacy_data_dir)
    elif not legacy_data_dir.exists():
        print("❌ Error: Could not find 'data' or 'data_legacy_broken'.")
        return
        
    # 2. Create the pristine new data folder
    original_data_dir.mkdir(parents=True, exist_ok=True)
        
    # 3. Find target folders in the LEGACY directory
    target_folders = []
    zyco_targets = ["zyco_8", "zyco_9", "zyco_10"]
    
    for d in legacy_data_dir.iterdir():
        if not d.is_dir():
            continue
            
        name_lower = d.name.lower()
        if name_lower.startswith("bench_") or name_lower in zyco_targets:
            target_folders.append(d)
            
    target_folders = sorted(target_folders, key=lambda x: x.name)
    
    if not target_folders:
        print(f" No target folders found in {legacy_data_dir}")
        return
        
    print(f"\n{'='*60}")
    print(f" Found {len(target_folders)} historical runs.")
    print(f" Rebuilding pristine data architecture and parsing raw text...")
    print(f"{'='*60}\n")
    
    for run_dir in target_folders:
        print(f"Processing {run_dir.name}...")
        process_run_dir(run_dir, original_data_dir)
        
    print(f"\n{'='*60}")
    print(" Master Clean Slate Rebuild Complete!")
    print(" Your 'pipeline/data' folder is now 100% clean, standardized to .tran 10n 1m, and ready for resimulation.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()