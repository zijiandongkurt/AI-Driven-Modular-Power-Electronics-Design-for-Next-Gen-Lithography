import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Setup paths so it can find the pipeline utility folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the duplicate checker we just updated
from pipeline.utility.check_duplicates import get_duplicates_per_batch

def plot_duplicates_for_zycos(zycos_folder: str):
    """Crawl a training folder and generate a duplicate-topology bar chart for each run.

    Args:
        zycos_folder (str): Path to the top-level zycos training folder containing
            Run_XXX subdirectories.
    """
    zycos_path = Path(zycos_folder)
    
    # Find all Run folders and sort them numerically
    run_folders = sorted([d for d in zycos_path.iterdir() if d.is_dir() and d.name.startswith("Run_")])
    
    if not run_folders:
        print(f"❌ No Run_XXX folders found inside {zycos_path}")
        return

    print(f"📊 Generating duplicate topology plots for {len(run_folders)} runs in {zycos_path.name}...\n")

    for run_dir in run_folders:
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Fetch the duplicate counts dictionary from our utility script
            dup_data = get_duplicates_per_batch(str(run_dir))
            
            if not dup_data:
                print(f"  ⚠️ Skipping {run_dir.name} (No batch data found)")
                continue
                
            # Extract and sort batches to ensure X-axis is chronological
            batches = sorted(dup_data.keys())
            dup_counts = [dup_data[b] for b in batches]
            
            plt.figure(figsize=(9, 6))
            plt.bar(batches, dup_counts, color='coral', alpha=0.8, edgecolor='black')
            plt.title(f'Duplicate Topologies Per Batch ({run_dir.name})', fontsize=14, fontweight='bold')
            plt.xlabel('Batch Number', fontsize=12)
            plt.ylabel('Duplicate Circuits', fontsize=12)
            plt.xticks(batches)
            
            # Scale Y axis dynamically based on max duplicates
            max_dup = max(dup_counts) if dup_counts and max(dup_counts) > 0 else 4
            plt.yticks(range(0, int(max_dup) + 2))
            plt.grid(axis='y', alpha=0.3)
            
            # Save the plot
            out_path = results_dir / '0_summary_duplicate_topologies.png'
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved duplicate plot -> {run_dir.name}/results/{out_path.name}")
            
        except Exception as e:
            print(f"  ❌ Failed to plot {run_dir.name}: {e}")

    print("\n🎉 Finished plotting all runs!")

if __name__ == '__main__':
    # Point this to the main training folder containing Run_001, Run_002, etc.
    TARGET_FOLDER = PROJECT_ROOT / "pipeline" / "data" / "zycos_005"
    plot_duplicates_for_zycos(TARGET_FOLDER)