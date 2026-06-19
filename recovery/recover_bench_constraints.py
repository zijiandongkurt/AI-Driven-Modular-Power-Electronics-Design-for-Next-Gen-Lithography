import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def main():
    legacy_dir = PROJECT_ROOT / "pipeline" / "data_legacy_broken"
    clean_dir = PROJECT_ROOT / "pipeline" / "data"

    if not legacy_dir.exists():
        print(f"❌ Error: Could not find legacy directory at {legacy_dir}")
        return

    # Only target Bench folders (Zycos is handled in the next script)
    run_folders = [d for d in clean_dir.iterdir() if d.is_dir() and d.name.startswith("Bench_")]
    run_folders = sorted(run_folders, key=lambda x: x.name)

    print(f"\n{'='*60}\n Deep-searching constraints for {len(run_folders)} Bench runs...\n{'='*60}\n")

    recovered_count = 0

    for run_dir in run_folders:
        run_name = run_dir.name
        legacy_run_dir = legacy_dir / run_name
        
        # Sort batches so we search chronologically
        legacy_batches = sorted([d for d in legacy_run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                                key=lambda x: int(x.name.split('_')[1]))
        
        found = False
        for batch_dir in legacy_batches:
            legacy_reward_file = batch_dir / "reward_results.json"
            
            if legacy_reward_file.exists():
                try:
                    with open(legacy_reward_file, "r", encoding="utf-8") as f:
                        legacy_data = json.load(f)

                    constraints = legacy_data.get("active_constraints", {})
                    if not constraints:
                        continue # Empty constraints, check next batch

                    weights = legacy_data.get("weights_used", legacy_data.get("weights", None))

                    recovery_package = {"active_constraints": constraints}
                    if weights:
                        recovery_package["weights"] = weights

                    # Save it to the root of the NEW run folder
                    output_file = run_dir / "active_constraint.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(recovery_package, f, indent=4)

                    print(f"   -> {run_name}: Recovered constraints from {batch_dir.name} (Vout: {constraints.get('vout_target', 'N/A')}V).")
                    recovered_count += 1
                    found = True
                    break # Stop searching this run once we find it!

                except Exception as e:
                    print(f" ❌ Error parsing {legacy_reward_file}: {e}")
                    
        if not found:
            print(f" ⚠️ Warning: Checked {len(legacy_batches)} batches in {run_name}, but found NO valid reward_results.json.")

    print(f"\n{'='*60}\n ✔ Successfully recovered constraints for {recovered_count}/{len(run_folders)} Bench runs!\n{'='*60}")

if __name__ == "__main__":
    main()