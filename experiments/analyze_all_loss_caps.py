import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.utility.topology_hasher import get_topological_hash

def analyze_all_caps():
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    zycos_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("zycos_")])
    
    if not zycos_folders:
        print(f"❌ No zycos_XXX folders found in {data_dir}")
        return

    print("🔍 Crawling historical data for all unique, valid penalties...")
    
    global_hashes = set()
    
    # Store the raw, unweighted penalties
    penalties = {
        "voltage": [],
        "efficiency": [],
        "volume": [],
        "components": []
    }
    
    for zycos_dir in zycos_folders:
        for run_dir in zycos_dir.glob("Run_*"):
            if not run_dir.is_dir(): continue
                
            for batch_dir in run_dir.glob("batch_*"):
                if not batch_dir.is_dir(): continue
                    
                val_file = batch_dir / "validation_results.json"
                rew_file = batch_dir / "reward_results.json"
                llm_dir = batch_dir / "LLM_output"
                
                if not (val_file.exists() and rew_file.exists() and llm_dir.exists()):
                    continue
                    
                try:
                    with open(val_file, "r", encoding="utf-8") as f:
                        validations = json.load(f)
                    with open(rew_file, "r", encoding="utf-8") as f:
                        reward_data = json.load(f)
                        rewards = reward_data.get("circuits", {})
                        constraints = reward_data.get("active_constraints", {})
                except Exception:
                    continue

                for cid, cdata in rewards.items():
                    # 1. Check Validity
                    if not validations.get(cid, {}).get("passed", False):
                        continue
                        
                    # 2. Check Topological Uniqueness
                    net_file = llm_dir / f"{cid}.net"
                    if not net_file.exists():
                        continue
                        
                    t_hash = get_topological_hash(net_file.read_text(encoding="utf-8"))
                    if t_hash in global_hashes:
                        continue
                    global_hashes.add(t_hash)

                    # 3. Extract Metrics
                    raw_metrics = cdata.get("raw_metrics", {})
                    v_out = raw_metrics.get("simulation_output_voltage")
                    eff = raw_metrics.get("efficiency")
                    vol = raw_metrics.get("total_volume_cm3")
                    comps = raw_metrics.get("total_components")
                    
                    target_vout = constraints.get("vout_target", 5.0)
                    target_eff = constraints.get("efficiency_target", 0.90)

                    # 4. Calculate Exact Raw Penalties (matching RewardFunctionNorm)
                    if v_out is not None and v_out < 10000.0:  # Ignore extreme physics explosions
                        penalties["voltage"].append((v_out - target_vout) ** 2)
                        
                    if eff is not None:
                        safe_eff = max(0.0, min(1.0, eff))
                        penalties["efficiency"].append(max(0.0, target_eff - safe_eff))
                        
                    if vol is not None and vol > 0 and vol < 9000.0:
                        penalties["volume"].append(vol)
                        
                    if comps is not None and comps > 0:
                        penalties["components"].append(comps)

    if not penalties["volume"]:
        print("❌ No valid data found.")
        return

    print("\n" + "="*60)
    print(f"📊 LOSS CAP DISTRIBUTION REPORT (Unique Valid: {len(global_hashes)})")
    print("="*60)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Raw Penalty Distributions for LOSS_CAP Calibration\n({len(global_hashes)} Unique Valid Topologies)", fontsize=16, fontweight='bold')
    
    plot_configs = [
        ("voltage", "Voltage Tracking Penalty (V_error²)", "purple", axs[0, 0]),
        ("efficiency", "Efficiency Penalty (Target - Actual)", "seagreen", axs[0, 1]),
        ("volume", "Volume Penalty (cm³)", "royalblue", axs[1, 0]),
        ("components", "Component Cost (Count)", "crimson", axs[1, 1])
    ]

    for key, title, color, ax in plot_configs:
        arr = np.array(penalties[key])
        if len(arr) == 0: continue
            
        v_min, v_max, v_mean, v_median = np.min(arr), np.max(arr), np.mean(arr), np.median(arr)
        p75, p90, p95, p99 = np.percentile(arr, [75, 90, 95, 99])
        
        print(f"\n--- {title} ---")
        print(f"Min: {v_min:.4f} | Mean: {v_mean:.4f} | Max: {v_max:.4f}")
        print(f"Median:   {v_median:.4f}")
        print(f"90th %:   {p90:.4f}  <-- (Conservative Cap)")
        print(f"95th %:   {p95:.4f}  <-- (Standard Cap)")

        # Plotting (Cap visual bounds to 99th percentile to avoid outlier squish)
        plot_max = p99 if p99 > 0 else v_max
        # Give a little breathing room on the x-axis, unless it's a discrete count like components
        if key == "components":
            bins = np.arange(v_min - 0.5, plot_max + 1.5, 1)
        else:
            bins = 50
            
        filtered_arr = arr[arr <= plot_max]
        
        ax.hist(filtered_arr, bins=bins, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(v_median, color='black', linestyle='dashed', linewidth=2, label=f'Median ({v_median:.2f})')
        ax.axvline(p90, color='orange', linestyle='dashed', linewidth=2, label=f'90th % ({p90:.2f})')
        ax.axvline(p95, color='red', linestyle='dashed', linewidth=2, label=f'95th % ({p95:.2f})')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = data_dir / "all_loss_caps_analysis.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print(f"✅ 2x2 Histogram grid saved to: {out_file.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    analyze_all_caps()