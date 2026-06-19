import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def main():
    legacy_dir = PROJECT_ROOT / "pipeline" / "data_legacy_broken"
    clean_dir = PROJECT_ROOT / "pipeline" / "data"

    if not legacy_dir.exists():
        print(f"❌ Error: Could not find legacy directory at {legacy_dir}")
        return

    # Look for all Bench_ and zycos_ folders in the CLEAN directory
    run_folders = [d for d in clean_dir.iterdir() if d.is_dir() and (d.name.startswith("Bench_") or d.name.startswith("zyco"))]
    run_folders = sorted(run_folders, key=lambda x: x.name)

    print(f"\n{'='*60}")
    print(f" Recovering constraints for {len(run_folders)} runs...")
    print(f"{'='*60}\n")

    recovered_count = 0

    for run_dir in run_folders:
        run_name = run_dir.name
        
        # Point to the first batch in the legacy folder to steal the JSON data
        legacy_batch_1 = legacy_dir / run_name / "batch_1"
        legacy_reward_file = legacy_batch_1 / "reward_results.json"

        if not legacy_reward_file.exists():
            print(f" ⚠️ Warning: No reward_results.json found for {run_name}/batch_1. Cannot recover constraints.")
            continue

        try:
            with open(legacy_reward_file, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)

            constraints = legacy_data.get("active_constraints", {})
            
            if not constraints:
                print(f" ⚠️ Warning: 'active_constraints' key missing in {run_name}/batch_1.")
                continue

            # Check if weights were also saved in the legacy JSON (useful for later)
            weights = legacy_data.get("weights_used", legacy_data.get("weights", None))

            # Package it up
            recovery_package = {
                "active_constraints": constraints
            }
            if weights:
                recovery_package["weights"] = weights

            # Save it to the root of the NEW run folder
            output_file = run_dir / "active_constraint.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(recovery_package, f, indent=4)

            print(f"   -> {run_name}: Recovered constraints (Vout: {constraints.get('vout_target', 'N/A')}V).")
            recovered_count += 1

        except Exception as e:
            print(f" ❌ Error parsing {legacy_reward_file}: {e}")

    print(f"\n{'='*60}")
    print(f" ✔ Successfully recovered constraints for {recovered_count}/{len(run_folders)} runs!")
    print(f" Saved as 'active_constraint.json' in each run directory.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()