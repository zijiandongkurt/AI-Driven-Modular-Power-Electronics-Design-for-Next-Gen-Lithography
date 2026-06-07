import os
import sys
import json
import numpy as np
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg') # Thread-safe backend
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from pipeline.utility.topology_hasher import get_topological_hash


def plot_cumulative_probabilities(run_id: str, target_batch: int, temperature: float = 0.05, top_k: int = 15, epsilon: float = 0.15, label: str = ""):
    """Calculate and plot cumulative sampling probabilities for unique topologies.

    Computes Epsilon-Greedy Top-K Softmax probabilities using pure fitness
    and renders a 2-panel chart showing probability mass accumulation.

    Args:
        run_id (str): Run identifier matching a folder under ``pipeline/data/``.
        target_batch (int): Only batches strictly before this index are used
            to reconstruct the database.
        temperature (float): Softmax temperature for elite exploitation.
        top_k (int): Number of elite candidates for softmax exploitation.
        epsilon (float): Fraction of probability mass reserved for uniform
            tail exploration.
        label (str): Optional label appended to the output filename.
    """
    data_dir = Path("pipeline/data") / run_id
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} not found.")
        return

    records = {}
    seen_hashes = set()

    # 1. Reconstruct database
    for b_idx in range(1, target_batch):
        batch_folder = data_dir / f"batch_{b_idx}"
        reward_file = batch_folder / "reward_results.json"
        valid_file = batch_folder / "validation_results.json"
        llm_out_dir = batch_folder / "LLM_output"

        if not reward_file.exists() or not valid_file.exists():
            continue

        with open(valid_file, "r", encoding="utf-8") as f:
            validations = json.load(f)
            
        with open(reward_file, "r", encoding="utf-8") as f:
            rewards = json.load(f).get("circuits", {})

        for cand_id, metrics in rewards.items():
            is_valid = validations.get(cand_id, {}).get("passed", False)
            fitness = metrics.get("fitness_score", -1.0)
            
            if fitness is not None and fitness > -1.0:
                net_file = llm_out_dir / f"{cand_id}.net"
                if net_file.exists():
                    net_text = net_file.read_text(encoding="utf-8")
                    try:
                        t_hash = get_topological_hash(net_text)
                    except (ValueError, Exception):
                        t_hash = f"invalid_{cand_id}"
                    if t_hash not in seen_hashes:
                        seen_hashes.add(t_hash)
                        records[cand_id] = {"fitness": fitness}

    if not records:
        print(f"Database is empty before Batch {target_batch}.")
        return

    # 2. Score All Candidates (Pure Fitness)
    keys = np.array(list(records.keys()))
    scores = np.array([records[k]["fitness"] for k in keys])
    
    # Sort descending by score to identify Top-K
    sorted_by_score_idx = np.argsort(scores)[::-1]
    
    top_k_indices = sorted_by_score_idx[:top_k]
    long_tail_indices = sorted_by_score_idx[top_k:]

    probs = np.zeros(len(keys))

    # 3. Apply Softmax to Elite (85% of mass)
    if len(top_k_indices) > 0:
        elite_scores = scores[top_k_indices]
        exp_scores = np.exp((elite_scores - np.max(elite_scores)) / temperature)
        elite_probs = exp_scores / np.sum(exp_scores)
        
        # Scale to reserve Epsilon for the tail
        actual_epsilon = epsilon if len(long_tail_indices) > 0 else 0.0
        probs[top_k_indices] = elite_probs * (1.0 - actual_epsilon)

    # 4. Apply Uniform Distribution to Long Tail (15% of mass)
    if len(long_tail_indices) > 0:
        tail_prob = epsilon / len(long_tail_indices)
        probs[long_tail_indices] = tail_prob

    # 5. Sort for the Pareto Chart
    if len(top_k_indices) > 0:
        elite_sorted_idx = top_k_indices[np.argsort(probs[top_k_indices])[::-1]]
    else:
        elite_sorted_idx = np.array([], dtype=int)
        
    if len(long_tail_indices) > 0:
        tail_sorted_idx = long_tail_indices[np.argsort(scores[long_tail_indices])[::-1]]
    else:
        tail_sorted_idx = np.array([], dtype=int)
        
    sorted_indices = np.concatenate((elite_sorted_idx, tail_sorted_idx))
    
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    sorted_fitness = scores[sorted_indices]

    # 6. Generate the 2-Panel Chart
    fig, (ax1, ax_fit) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    plt.subplots_adjust(hspace=0.08)

    x_pos = np.arange(len(sorted_probs))

    ax1.bar(x_pos, sorted_probs, color='steelblue', alpha=0.7, label='Individual Probability')
    ax1.set_ylabel('Individual Sampling Probability', color='steelblue', fontsize=12, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.plot(x_pos, cumulative_probs, color='crimson', marker='.', linestyle='-', linewidth=2, label='Cumulative Mass')
    ax2.set_ylabel('Cumulative Probability', color='crimson', fontsize=12, fontweight='bold')
    
    boundary_x = min(top_k - 0.5, len(keys) - 0.5)
    ax1.axvline(x=boundary_x, color='black', linestyle='--', alpha=0.8, linewidth=2, label="Top-K Boundary")
    ax_fit.axvline(x=boundary_x, color='black', linestyle='--', alpha=0.8, linewidth=2)

    ax1.text(boundary_x + 1, ax1.get_ylim()[1]*0.8, f"Exploration Zone\n(Epsilon = {epsilon*100:.0f}%)", 
             color='black', fontsize=11, alpha=0.8, va='top')
    ax1.text(boundary_x - 1, ax1.get_ylim()[1]*0.8, f"Exploitation Zone\n(Top {top_k})", 
             color='black', fontsize=11, alpha=0.8, ha='right', va='top')

    ax1.set_yticklabels(['{:,.1%}'.format(x) for x in ax1.get_yticks()])
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()])

    ax1.set_title(f"Cumulative Sampling (Unique Topologies Only)\nBefore Batch {target_batch} | Temp: {temperature}, K: {top_k}, ε: {epsilon}", fontsize=14, fontweight='bold')
    
    ax_fit.plot(x_pos, sorted_fitness, color='forestgreen', marker='x', linestyle='-', linewidth=1.5, alpha=0.8)
    ax_fit.set_ylabel('Raw Fitness Score', color='forestgreen', fontsize=12, fontweight='bold')
    ax_fit.set_xlabel('Unique Candidate Rank (Most Likely -> Least Likely)', fontsize=12)
    ax_fit.grid(True, linestyle='--', alpha=0.4)

    # Save to disk
    output_dir = data_dir / "results" / "probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_label = f"_{label}" if label else ""
    filename = output_dir / f"cumulative_probs_topk_batch_{target_batch}{filename_label}.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()