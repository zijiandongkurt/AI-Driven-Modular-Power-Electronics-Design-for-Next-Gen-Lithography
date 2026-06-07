import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Setup paths so it can find the pipeline utility folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the duplicate checker we built
from pipeline.utility.check_duplicates import get_duplicates_per_batch

def generate_master_duplicate_plot(zycos_folder: str):
    base_dir = Path(zycos_folder)
    
    # Find all Run folders
    run_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")])
    
    if not run_folders:
        print(f"❌ No Run_XXX folders found inside {base_dir}")
        return

    print(f"🔍 Analyzing mode collapse across {len(run_folders)} runs...")

    # Dictionary to aggregate duplicate counts
    # Structure: { batch_number: [dup_count_run1, dup_count_run2, ...] }
    global_duplicates = {}

    for run_dir in run_folders:
        try:
            # Fetch the duplicate counts for this specific run
            dup_data = get_duplicates_per_batch(str(run_dir))
            
            for b_num, count in dup_data.items():
                if b_num not in global_duplicates:
                    global_duplicates[b_num] = []
                global_duplicates[b_num].append(count)
                
        except Exception as e:
            print(f"⚠️ Error reading duplicates for {run_dir.name}: {e}")

    if not global_duplicates:
        print("❌ Could not extract any duplicate data.")
        return

    # Sort batches chronologically
    batches = sorted(global_duplicates.keys())
    
    # Calculate Mean and Standard Deviation across all runs
    mean_dups = np.array([np.mean(global_duplicates[b]) for b in batches])
    std_dups = np.array([np.std(global_duplicates[b]) for b in batches])

    plt.figure(figsize=(10, 6))
    
    # Plot the average trend line
    plt.plot(batches, mean_dups, label='Mean Duplicates', color='coral', linewidth=2.5, marker='o')
    
    # Add the shaded standard deviation region
    # np.clip prevents the lower bound from dropping below 0 (impossible to have negative duplicates)
    plt.fill_between(batches, 
                     np.clip(mean_dups - std_dups, 0, None), 
                     mean_dups + std_dups, 
                     color='lightsalmon', alpha=0.3, label='± 1 Std Dev')

    plt.title(f'Master Mode Collapse Tracker: {base_dir.name}\n(Mean ± Std over {len(run_folders)} Runs)', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Number of Duplicate Circuits', fontsize=12)
    plt.xticks(batches)
    
    # Dynamic Y-axis scaling
    max_y = max(mean_dups + std_dups) if len(mean_dups) > 0 else 4
    plt.yticks(range(0, int(max_y) + 2))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=11)
    
    # Save to the root of the zycos folder
    out_file = base_dir / "master_duplicate_topologies.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Success! Master duplicate plot saved to: {out_file}")

if __name__ == "__main__":
    # Point this to your main training folder
    TARGET_FOLDER = PROJECT_ROOT / "pipeline" / "data" / "zycos_005"
    generate_master_duplicate_plot(TARGET_FOLDER)