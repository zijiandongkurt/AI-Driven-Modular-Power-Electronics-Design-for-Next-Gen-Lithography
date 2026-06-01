import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))
from pipeline.utility.topology_hasher import get_topological_hash

def plot_uniqueness_for_run(run_folder_path: str):
    data_dir = Path(run_folder_path)
    batch_folders = sorted([d for d in data_dir.iterdir() if d.name.startswith("batch_")], 
                           key=lambda x: int(x.name.split('_')[1]))
    
    batches, valid_rates, unique_rates = [], [], []
    global_hashes = set()
    cumulative_total = 0
    
    for batch_dir in batch_folders:
        b_idx = int(batch_dir.name.split('_')[1])
        val_file = batch_dir / "validation_results.json"
        llm_dir = batch_dir / "LLM_output"
        
        if not val_file.exists(): continue
        
        with open(val_file, "r") as f: val_data = json.load(f)
        
        batch_total = len(val_data)
        batch_valid = sum(1 for v in val_data.values() if v.get("passed", False))
        
        cumulative_total += batch_total
        
        # Hash new netlists
        for cid in val_data.keys():
            net_path = llm_dir / f"{cid}.net"
            if net_path.exists():
                t_hash = get_topological_hash(net_path.read_text(encoding="utf-8"))
                global_hashes.add(t_hash)
                
        batches.append(b_idx)
        valid_rates.append((batch_valid / batch_total * 100) if batch_total > 0 else 0)
        unique_rates.append((len(global_hashes) / cumulative_total * 100) if cumulative_total > 0 else 0)

    # Plotting
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(batches, valid_rates, color='mediumseagreen', marker='o', linewidth=2.5, label='Validity Rate (Per Batch)')
    ax1.set_xlabel('Batch Number', fontsize=12); ax1.set_ylabel('Validity Rate (%)', color='darkgreen')
    ax1.set_ylim(0, 105); ax1.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(batches, unique_rates, color='coral', marker='s', linestyle='--', linewidth=2.5, label='Cumulative Uniqueness')
    ax2.set_ylabel('Uniqueness Rate (%)', color='orangered'); ax2.set_ylim(0, 105)
    
    plt.title(f'Exploration vs. Yield: Uniqueness & Validity ({data_dir.name})', fontweight='bold')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    plt.savefig(results_dir / '0_summary_yield_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Yield/Uniqueness plot -> {data_dir.name}/results/")

if __name__ == "__main__":
    TARGET_ZYCOS = PROJECT_ROOT / "pipeline" / "data" / "zycos_006"
    
    # Find all Run_XXX folders and sort them
    run_folders = sorted([d for d in TARGET_ZYCOS.iterdir() if d.is_dir() and d.name.startswith("Run_")])
    
    print(f"🔍 Found {len(run_folders)} runs in {TARGET_ZYCOS.name}. Generating yield/uniqueness plots...")
    
    for run_dir in run_folders:
        print(f"\n--- Processing {run_dir.name} ---")
        plot_uniqueness_for_run(str(run_dir))