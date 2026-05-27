import os
import json
import numpy as np
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg') # Thread-safe backend for headless execution
import matplotlib.pyplot as plt

def shorten_cid(cid):
    """Shortens Phase1_cons4_b2_cand1 into b2_c1 for cleaner plot labels."""
    match = re.search(r'_b(\d+)_cand(\d+)', cid)
    if match:
        return f"b{match.group(1)}_c{match.group(2)}"
    return cid[:10]

def plot_softmax_probabilities(run_id: str, target_batch: int, temperature: float = 0.05):
    """
    Reconstructs the database right before 'target_batch' begins,
    calculates probabilities using pure fitness, and plots Probability vs. Fitness.
    """
    data_dir = Path("pipeline/data") / run_id
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} not found.")
        return

    records = {}

    # 1. Reconstruct database from Batch 1 up to (target_batch - 1)
    for b_idx in range(1, target_batch):
        batch_folder = data_dir / f"batch_{b_idx}"
        reward_file = batch_folder / "reward_results.json"
        valid_file = batch_folder / "validation_results.json"

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
                records[cand_id] = {
                    "fitness": fitness
                }

    if not records:
        print(f"Database is empty before Batch {target_batch}.")
        return

    # 2. Calculate Probabilities (Pure Fitness)
    keys = list(records.keys())
    scores = np.array([records[k]["fitness"] for k in keys])
        
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / np.sum(exp_scores)

    # Zip together and sort by probability (descending)
    results = list(zip(keys, scores, probs))
    results.sort(key=lambda x: x[2], reverse=True)

    cids = [r[0] for r in results]
    fitness_list = [r[1] for r in results]
    prob_list = [r[2] for r in results]

    # 3. Generate the Scatter Plot
    plt.figure(figsize=(14, 8))
    
    # Plot all dots in standard color
    plt.scatter(fitness_list, prob_list, color='steelblue', alpha=0.6, s=50, edgecolor='white', zorder=2)
    
    # Highlight top 5 dots in red
    plt.scatter(fitness_list[:5], prob_list[:5], color='crimson', alpha=0.9, s=90, edgecolor='black', label="Top 5 Candidates", zorder=3)

    # Label only the top 15 to avoid cluttering the bottom left
    for i in range(min(15, len(cids))):
        x, y = fitness_list[i], prob_list[i]
        short_id = shorten_cid(cids[i])
        plt.annotate(short_id, (x, y), xytext=(8, 4), textcoords="offset points", fontsize=9, fontweight='bold' if i < 5 else 'normal', alpha=0.9, zorder=4)

    # Formatting
    plt.title(f"Softmax Probability vs. Raw Fitness (Before Batch {target_batch})\nTemp: {temperature}", fontsize=14, fontweight='bold')
    plt.xlabel('Raw Fitness Score', fontsize=12)
    plt.ylabel('Sampling Probability', fontsize=12)
    
    # Convert y-axis to percentages for readability
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    
    plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
    plt.legend(loc='upper left')

    # Save to disk
    output_dir = data_dir / "results" / "probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"probabilities_batch_{target_batch}.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Generated plot: {filename}")

if __name__ == "__main__":
    plot_softmax_probabilities(run_id="Run_012", target_batch=12, temperature=0.05)
    plot_softmax_probabilities(run_id="Run_012", target_batch=13, temperature=0.05)