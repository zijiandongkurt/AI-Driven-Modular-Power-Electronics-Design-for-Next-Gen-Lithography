import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from pathlib import Path

def plot_softmax_probabilities(run_id: str, target_batch: int, temperature: float = 0.05, top_k: int = 15, epsilon: float = 0.15):
    """
    Plots the Epsilon-Greedy Top-K Softmax probability distribution.
    Uses a Lollipop chart against Raw Fitness to visualize the exponential curve.
    """
    data_dir = Path("pipeline/data") / run_id
    if not data_dir.exists():
        return

    # 1. Rebuild the cumulative database up to the PREVIOUS batch
    cumulative_fitness = {}
    for i in range(1, target_batch):
        reward_file = data_dir / f"batch_{i}" / "reward_results.json"
        if reward_file.exists():
            with open(reward_file, "r", encoding="utf-8") as f:
                rewards = json.load(f).get("circuits", {})
                for cid, data in rewards.items():
                    fit = data.get("fitness_score")
                    if fit is not None:
                        cumulative_fitness[cid] = fit

    if not cumulative_fitness:
        return # Database is empty, nothing to plot

    # 2. Sort candidates by fitness (Highest to Lowest)
    sorted_cands = sorted(cumulative_fitness.items(), key=lambda x: x[1], reverse=True)
    cids = [x[0] for x in sorted_cands]
    scores = np.array([x[1] for x in sorted_cands])

    # 3. Apply the Epsilon-Greedy Top-K Math
    elite_scores = scores[:top_k]
    tail_scores = scores[top_k:]
    
    probabilities = np.zeros(len(scores))
    colors = []

    has_tail = len(tail_scores) > 0
    actual_epsilon = epsilon if has_tail else 0.0

    # Calculate Top-K Softmax (Exploitation)
    if len(elite_scores) > 0:
        exp_scores = np.exp((elite_scores - np.max(elite_scores)) / temperature)
        elite_probs = (exp_scores / np.sum(exp_scores)) * (1.0 - actual_epsilon)
        probabilities[:len(elite_scores)] = elite_probs
        colors.extend(['#58d68d'] * len(elite_scores)) # Green for Elite

    # Calculate Long Tail Uniform (Exploration)
    if has_tail:
        tail_prob = actual_epsilon / len(tail_scores)
        probabilities[len(elite_scores):] = tail_prob
        colors.extend(['#f5b041'] * len(tail_scores)) # Orange for Exploration

    # Generate short labels (e.g. Phase1_cons4_b2_cand1 -> b2_c1)
    short_labels = []
    for cid in cids:
        match = re.search(r'_b(\d+)_cand(\d+)', cid)
        short_labels.append(f"b{match.group(1)}_c{match.group(2)}" if match else cid[:8])

    # 4. Generate the Plot
    plt.figure(figsize=(12, 7))
    
    elite_x = elite_scores
    elite_y = probabilities[:len(elite_scores)] * 100.0
    
    tail_x = tail_scores if has_tail else []
    tail_y = probabilities[len(elite_scores):] * 100.0 if has_tail else []

    # Adaptive bar width based on the spread of the data
    x_range = max(scores) - min(scores) if len(scores) > 1 else 1.0
    bar_width = max(0.005, x_range * 0.005)

    # Draw the "bars" up from the X-axis
    plt.bar(elite_x, elite_y, width=bar_width, color='#58d68d', alpha=0.7, edgecolor='black', linewidth=0.5, zorder=2)
    if has_tail:
        plt.bar(tail_x, tail_y, width=bar_width, color='#f5b041', alpha=0.6, edgecolor='gray', linewidth=0.5, zorder=2)

    # Draw the scatter dots on top of the bars to match your reference image
    plt.scatter(elite_x, elite_y, color='#58d68d', s=80, edgecolor='black', zorder=3)
    if has_tail:
        plt.scatter(tail_x, tail_y, color='#f5b041', s=35, edgecolor='white', alpha=0.9, zorder=3)

    # Add text labels ONLY to the Top 7 to prevent text overlapping on the curve
    for i in range(min(7, len(elite_x))):
        plt.annotate(
            short_labels[i], 
            (elite_x[i], elite_y[i]), 
            xytext=(10, 0), 
            textcoords='offset points', 
            fontsize=9, 
            va='center', 
            zorder=4
        )

    # 5. Formatting (Matching the image style)
    plt.title(f'Epsilon-Greedy Probability vs. Raw Fitness (Entering Batch {target_batch})\nTemp: {temperature} | Top-K: {top_k} | Epsilon: {epsilon}', fontsize=14, fontweight='bold')
    plt.ylabel('Sampling Probability', fontsize=12)
    plt.xlabel('Raw Fitness Score', fontsize=12)
    
    # Format Y-axis as actual percentages (0.5%, 1.0%, etc.)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    
    # Light dashed grid exactly like the image
    plt.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=1)

    # Add Legend
    legend_elements = [
        Patch(facecolor='#58d68d', edgecolor='black', label=f'Elite Top-{top_k} (Softmax)'),
        Patch(facecolor='#f5b041', edgecolor='gray', label=f'Long Tail (Uniform {epsilon*100}%)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    
    # Save to the results folder
    out_dir = data_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"probabilities_entering_batch_{target_batch}.png", dpi=300)
    plt.close()