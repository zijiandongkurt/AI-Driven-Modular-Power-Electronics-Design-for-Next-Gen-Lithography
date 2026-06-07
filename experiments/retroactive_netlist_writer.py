import os
import sys
import json
import re
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import your new net_writer module
from pipeline.llm_topology_generation.net_writer import write_netlists

def process_zycos_folder(zycos_dir: Path):
    run_folders = sorted([d for d in zycos_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")])
    if not run_folders:
        return
        
    print(f"\n{'='*60}\n Retroactively generating .net files for {zycos_dir.name}\n{'='*60}")

    for run_dir in run_folders:
        # Sort batches numerically
        batch_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                               key=lambda x: int(x.name.split('_')[1]))
        
        for batch_dir in batch_folders:
            raw_file = batch_dir / "raw_output.txt"
            reward_file = batch_dir / "reward_results.json"
            
            if not raw_file.exists() or not reward_file.exists():
                continue
                
            text = raw_file.read_text(encoding="utf-8")
            # Split using the exact visual delimiter LLM_API outputs
            blocks = re.split(r"={10,}\n\s*CANDIDATE \d+\s*\n={10,}\n", text)
            candidate_texts = [b.strip() for b in blocks[1:] if b.strip()]
            
            if not candidate_texts:
                continue
                
            try:
                with open(reward_file, "r", encoding="utf-8") as f:
                    reward_data = json.load(f)
            except Exception as e:
                print(f"   Error reading JSON in {batch_dir.name}: {e}")
                continue
                
            constraint = reward_data.get("active_constraints", {})
            circuits = reward_data.get("circuits", {})
            
            if not circuits:
                continue
                
            # Extract custom names (e.g., g1_cand1) from the dictionary keys
            custom_names = []
            for cid in circuits.keys():
                match = re.search(r'_(g\d+_cand\d+)', cid)
                if match:
                    custom_names.append(match.group(1))
                    
            # Remove duplicates and sort them logically (g1_cand1, g1_cand2, g2_cand1...)
            # This ensures they perfectly align with Candidate 1, 2, 3... in the text file
            def sort_key(name):
                g, c = name.split('_')
                return (int(g[1:]), int(c[4:]))
                
            custom_names = sorted(list(set(custom_names)), key=sort_key)
            
            # Ensure 1-to-1 mapping
            if len(custom_names) != len(candidate_texts):
                print(f"   Mismatch in {run_dir.name}/{batch_dir.name}: {len(custom_names)} names vs {len(candidate_texts)} texts. Truncating to fit.")
                min_len = min(len(custom_names), len(candidate_texts))
                custom_names = custom_names[:min_len]
                candidate_texts = candidate_texts[:min_len]

            if not candidate_texts:
                continue
                
            label_raw = constraint.get("_comment", f"Topology_{run_dir.name}")
            clean_label = "00_" + re.sub(r'[^a-zA-Z0-9]', '_', label_raw)
            clean_label = re.sub(r'_+', '_', clean_label).strip('_')
            
            # Format exactly how net_writer expects it (relative path from 'data')
            batch_id = f"{zycos_dir.name}/{run_dir.name}/{batch_dir.name}"
            
            try:
                write_netlists(
                    netlists=candidate_texts,
                    constraint=constraint,
                    label=clean_label,
                    batchID=batch_id,
                    custom_names=custom_names
                )
                print(f"   Restored {len(candidate_texts)} .net files -> {batch_id}/LLM_output")
            except Exception as e:
                print(f"   Error writing netlists for {batch_id}: {e}")

def main():
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    zycos_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("zycos_")])
    
    if not zycos_folders:
        print(f" No zycos_XXX folders found in {data_dir}")
        return
        
    print(f" Found {len(zycos_folders)} historical training runs. Reconstructing LLM_output folders...")
    
    for zycos_dir in zycos_folders:
        process_zycos_folder(zycos_dir)
        
    print("\n Master reconstruction complete! All physical .net files are restored.")

if __name__ == "__main__":
    main()