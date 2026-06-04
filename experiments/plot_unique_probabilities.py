import os
import sys
import json
import numpy as np
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

# Add project root to path to import hasher
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from pipeline.utility.topology_hasher import get_topological_hash

def get_unique_database(data_dir: Path, target_batch: int):
    """Rebuilds the database up to target_batch, keeping ONLY unique topologies."""
    records = {}
    seen_hashes = set()

    for i in range(1, target_batch):
        batch_folder = data_dir / f"batch_{i}"
        reward_file = batch_folder / "reward_results.json"
        llm_out_dir = batch_folder / "LLM_output"
        
        if reward_file.exists():
            with open(reward_file, "r", encoding="utf-8") as f:
                rewards = json.load(f).get("circuits", {})
                for cid, data in rewards.items():
                    fit = data.get("fitness_score")
                    if fit is not None and fit > -1.0:
                        net_file = llm_out_dir / f"{cid}.net"
                        if net_file.exists():
                            net_text = net_file.read_text(encoding="utf-8")
                            t_hash = get_topological_hash(net_text)
                            if t_hash not in seen_hashes:
                                seen_hashes.add(t_hash)
                                records[cid] = fit
    return records

def plot_unique_probabilities(run_folder: str, target_batch: int, temp=0.05, top_k=15, epsilon=0.15):
    """Generates both the Softmax and Cumulative probability plots for unique topologies."""
    data_dir = Path(run_folder)
    records = get_unique_database(data_dir, target_batch)
    
    if not records:
        print(f"⚠️ Database empty before Batch {target_batch} in {data_dir.name}")
        return

    # Setup the new output directory
    out_dir = data_dir / "results" / "probability_no_duplicates"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort candidates
    sorted_cands = sorted(records.items(), key=lambda x: x[1], reverse=True)
    cids = [x[0] for x in sorted_cands]
    scores = np.array([x[1] for x in sorted_cands])

    elite_scores = scores[:top_k]
    tail_scores = scores[top_k:]
    probs = np.zeros(len(scores))
    has_tail = len(tail_scores) > 0
    actual_epsilon = epsilon if has_tail else 0.0

    # Math
    if len(elite_scores) > 0:
        exp_scores = np.exp((elite_scores - np.max(elite_scores)) / temp)
        probs[:len(elite_scores)] = (exp_scores / np.sum(exp_scores)) * (1.0 - actual_epsilon)
    if has_tail:
        probs[len(elite_scores):] = actual_epsilon / len(tail_scores)

    # --- 1. PLOT SOFTMAX LOLLIPOP ---
    plt.figure(figsize=(12, 7))
    elite_y = probs[:len(elite_scores)] * 100.0
    tail_y = probs[len(elite_scores):] * 100.0 if has_tail else []
    bar_width = max(0.005, (max(scores) - min(scores)) * 0.005) if len(scores) > 1 else 0.01

    plt.bar(elite_scores, elite_y, width=bar_width, color='#58d68d', alpha=0.7, edgecolor='black')
    if has_tail: plt.bar(tail_scores, tail_y, width=bar_width, color='#f5b041', alpha=0.6, edgecolor='gray')
    
    plt.scatter(elite_scores, elite_y, color='#58d68d', s=80, edgecolor='black', zorder=3)
    if has_tail: plt.scatter(tail_scores, tail_y, color='#f5b041', s=35, edgecolor='white', alpha=0.9, zorder=3)

    plt.title(f'Softmax Probabilities (Unique Only) - Entering Batch {target_batch}\nTemp: {temp} | Top-K: {top_k} | ε: {epsilon}', fontweight='bold')
    plt.ylabel('Sampling Probability'); plt.xlabel('Raw Fitness Score')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.savefig(out_dir / f"softmax_entering_batch_{target_batch}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. PLOT CUMULATIVE MASS ---
    cumulative_probs = np.cumsum(probs)
    fig, (ax1, ax_fit) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    plt.subplots_adjust(hspace=0.08)
    x_pos = np.arange(len(probs))

    ax1.bar(x_pos, probs, color='steelblue', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(x_pos, cumulative_probs, color='crimson', marker='.', linewidth=2)
    
    boundary_x = min(top_k - 0.5, len(cids) - 0.5)
    ax1.axvline(x=boundary_x, color='black', linestyle='--', linewidth=2)
    ax_fit.axvline(x=boundary_x, color='black', linestyle='--', linewidth=2)

    ax1.set_yticklabels(['{:,.1%}'.format(x) for x in ax1.get_yticks()]); ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()])
    ax1.set_title(f"Cumulative Mass (Unique Only) - Entering Batch {target_batch}", fontweight='bold')
    
    ax_fit.plot(x_pos, scores, color='forestgreen', marker='x')
    ax_fit.set_ylabel('Fitness'); ax_fit.set_xlabel('Unique Candidate Rank')
    
    plt.savefig(out_dir / f"cumulative_entering_batch_{target_batch}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Unique Probability Plots -> {out_dir.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    TARGET_ZYCOS = PROJECT_ROOT / "pipeline" / "data" / "zycos_006"
    
    # Find all Run_XXX folders and sort them
    run_folders = sorted([d for d in TARGET_ZYCOS.iterdir() if d.is_dir() and d.name.startswith("Run_")])
    
    print(f"🔍 Found {len(run_folders)} runs in {TARGET_ZYCOS.name}. Generating probability plots...")
    
    for run_dir in run_folders:
        print(f"\n--- Processing {run_dir.name} ---")
        
        # Count how many batches exist in this specific run
        batch_folders = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
        num_batches = len(batch_folders)
        
        if num_batches == 0:
            print(f"  ⚠️ No batches found in {run_dir.name}")
            continue
            
        # Generate a plot for entering Batch 2, 3, 4 ... up to (Total Batches + 1)
        for b_idx in range(2, num_batches + 2):
            plot_unique_probabilities(str(run_dir), target_batch=b_idx)