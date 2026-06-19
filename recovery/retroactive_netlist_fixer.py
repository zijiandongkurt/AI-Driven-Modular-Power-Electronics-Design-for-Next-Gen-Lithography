import os
import sys
import re
import shutil
from pathlib import Path

# Add project root to path (assuming script is in /experiments)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def fix_netlist_content(content: str) -> str:
    """
    Parses SPICE text to fix MOSFET bulk pins and Gate Drive reference nodes.
    """
    lines = content.split('\n')
    gate_to_source_map = {}
    fixed_lines = []
    
    # === PASS 1: Map MOSFETs and fix Bulk connections ===
    for line in lines:
        parts = line.strip().split()
        # Look for MOSFET definitions: M<name> <drain> <gate> <source> <bulk> ...
        if len(parts) >= 5 and parts[0].upper().startswith('M'):
            gate_node = parts[2]
            source_node = parts[3]
            bulk_node = parts[4]
            
            # Map this gate to its corresponding source
            gate_to_source_map[gate_node] = source_node
            
            # Fix the bulk node if it doesn't match the source
            if bulk_node != source_node:
                parts[4] = source_node
                
            # Reconstruct the line while preserving leading whitespace
            leading_space = line[:len(line) - len(line.lstrip())]
            fixed_lines.append(leading_space + " ".join(parts))
        else:
            fixed_lines.append(line)
            
    # === PASS 2: Fix Gate Drive references ===
    final_lines = []
    for line in fixed_lines:
        parts = line.strip().split(maxsplit=3)
        # Look for Voltage sources using PULSE: V<name> <pos> <neg> PULSE(...)
        if len(parts) >= 4 and parts[0].upper().startswith('V') and 'PULSE' in line.upper():
            name = parts[0]
            pos_node = parts[1]
            neg_node = parts[2]
            rest = parts[3]
            
            # If this voltage source is driving a known gate
            if pos_node in gate_to_source_map:
                expected_ref = gate_to_source_map[pos_node]
                
                # Fix the negative reference if it's pointing to 0/ground inappropriately
                if neg_node != expected_ref:
                    parts[2] = expected_ref
                    leading_space = line[:len(line) - len(line.lstrip())]
                    line = leading_space + f"{name} {pos_node} {parts[2]} {rest}"
                    
        final_lines.append(line)
        
    return '\n'.join(final_lines)

def process_run_dir(run_dir: Path):
    batch_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                           key=lambda x: int(x.name.split('_')[1]))
    
    total_fixed = 0
    for batch_dir in batch_folders:
        old_output_dir = batch_dir / "LLM_output"
        broken_output_dir = batch_dir / "LLM_output_broken"
        
        # Check if this batch even has outputs, or if we already backed it up
        if not old_output_dir.exists() and not broken_output_dir.exists():
            continue
            
        # Step 1: Backup original as LLM_output_broken
        if old_output_dir.exists() and not broken_output_dir.exists():
            old_output_dir.rename(broken_output_dir)
            
        # Step 2: Create a fresh LLM_output folder
        new_output_dir = batch_dir / "LLM_output"
        new_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Process all .net files from the broken folder into the new folder
        fixed_count = 0
        for net_file in broken_output_dir.glob("*.net"):
            original_content = net_file.read_text(encoding="utf-8")
            fixed_content = fix_netlist_content(original_content)
            
            (new_output_dir / net_file.name).write_text(fixed_content, encoding="utf-8")
            fixed_count += 1
            
        if fixed_count > 0:
            print(f"   -> {batch_dir.name}: Restored {fixed_count} .net files.")
            total_fixed += fixed_count
            
    if total_fixed > 0:
        print(f" ✔ Successfully fixed {total_fixed} total files in {run_dir.name}")

def main():
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found at {data_dir}")
        return
        
    # Find all Bench_* folders AND Zyco 8, 9, 10
    target_folders = []
    zyco_targets = ["zyco_8", "zyco_9", "zyco_10"]
    
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
            
        name_lower = d.name.lower()
        if name_lower.startswith("bench_") or name_lower in zyco_targets:
            target_folders.append(d)
            
    # Sort folders alphabetically for clean terminal output
    target_folders = sorted(target_folders, key=lambda x: x.name)
    
    if not target_folders:
        print(f" No target folders found in {data_dir}")
        return
        
    print(f"\n{'='*60}\n Found {len(target_folders)} historical benchmark runs.\n Applying physics fixes (Bulk Tie + Floating Gates)...\n{'='*60}\n")
    
    for run_dir in target_folders:
        print(f"Processing {run_dir.name}...")
        process_run_dir(run_dir)
        
    print(f"\n{'='*60}\n Retroactive Netlist Fix Complete!\n{'='*60}")
    print("All original folders have been backed up as 'LLM_output_broken'.")

if __name__ == "__main__":
    main()