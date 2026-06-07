import os
import json
import glob
import re
import numpy as np

# Use thread-safe, headless backend to prevent Tkinter crashes during ML loops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from pipeline.utility.topology_hasher import get_topological_hash

def extract_batch_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else -1

def shorten_cid(cid):
    """Shorten a candidate id to a compact label for plot annotations.

    Args:
        cid (str): Full candidate identifier (e.g. ``Phase1_cons4_b2_cand1``).

    Returns:
        str: Short label such as ``b2_c1``, or first 10 characters if the
            pattern is not found.
    """
    match = re.search(r'_b(\d+)_cand(\d+)', cid)
    if match:
        return f"b{match.group(1)}_c{match.group(2)}"
    return cid[:10]

def get_sampled_parents(batch_folder):
    """Find which parents were sampled to generate a given batch.

    Args:
        batch_folder (str): Path to the batch directory.

    Returns:
        list[str]: Parent circuit identifiers extracted from
            ``prompt_parent_*.txt`` filenames.
    """
    parent_files = glob.glob(os.path.join(batch_folder, 'prompt_parent_*.txt'))
    return [os.path.basename(pf).replace('prompt_parent_', '').replace('.txt', '') for pf in parent_files]

def plot_run_results(run_dir):
    batch_folders = sorted(glob.glob(os.path.join(run_dir, 'batch_*')), key=lambda x: extract_batch_number(os.path.basename(x)))
    if not batch_folders:
        print(f"  ⚠️ No batch folders found in {run_dir}. Skipping.")
        return

    results_dir = os.path.join(run_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    batches = []
    history = {}
    cumulative_db = {}
    
    valid_counts = []
    unique_valid_counts = []   
    
    batch_validity_rates = []
    cumulative_validity_rates = []
    batch_uniqueness_rates = []
    cumulative_uniqueness_rates = []

    cumulative_total_attempted = 0
    cumulative_total_valid = 0
    exploration_seen_hashes = set() # Tracks ALL topologies (Valid + Invalid)
    global_seen_hashes = set()      # Tracks ONLY rewarded topologies for the swarm plots
    
    batch_avg_fitness, batch_max_fitness = [], []
    global_avg_fitness, global_max_fitness = [], []
    
    target_voltage = None
    target_efficiency = None

    for folder in batch_folders:
        b_idx = extract_batch_number(os.path.basename(folder))
        batches.append(b_idx)
        
        reward_file = os.path.join(folder, 'reward_results.json')
        valid_file = os.path.join(folder, 'validation_results.json')
        llm_out_dir = os.path.join(folder, 'LLM_output')
        
        new_cands = {}
        b_fit_batch = [] 
        batch_unique_valid = 0
        val_data = {}
        batch_attempted = 0
        batch_valid = 0

        # A. Safely Calculate Batch Validity from the True Validation Ledger
        if os.path.exists(valid_file):
            with open(valid_file, 'r') as f:
                val_data = json.load(f)
            batch_attempted = len(val_data)
            batch_valid = sum(1 for v in val_data.values() if v.get('passed', False))
            
        cumulative_total_attempted += batch_attempted
        cumulative_total_valid += batch_valid

        # B. Calculate True Structural Exploration (Uniqueness) across ALL generated .net files
        hashes_before = len(exploration_seen_hashes)
        net_files = glob.glob(os.path.join(llm_out_dir, '*.net'))
        if not net_files:
            net_files = glob.glob(os.path.join(folder, '*.net')) # Fallback
            
        for net_file in net_files:
            try:
                net_text = open(net_file, "r", encoding="utf-8").read()
                t_hash = get_topological_hash(net_text)
                exploration_seen_hashes.add(t_hash)
            except Exception:
                continue
                
        hashes_after = len(exploration_seen_hashes)
        new_unique_this_batch = hashes_after - hashes_before

        # Calculate Rates (Protect against ZeroDivisionErrors)
        b_val_rate = (batch_valid / batch_attempted * 100) if batch_attempted > 0 else 0
        c_val_rate = (cumulative_total_valid / cumulative_total_attempted * 100) if cumulative_total_attempted > 0 else 0
        
        b_uniq_rate = (new_unique_this_batch / batch_attempted * 100) if batch_attempted > 0 else 0
        c_uniq_rate = (len(exploration_seen_hashes) / cumulative_total_attempted * 100) if cumulative_total_attempted > 0 else 0

        batch_validity_rates.append(b_val_rate)
        cumulative_validity_rates.append(c_val_rate)
        batch_uniqueness_rates.append(b_uniq_rate)
        cumulative_uniqueness_rates.append(c_uniq_rate)
        valid_counts.append(batch_valid)

        # C. Process Rewards for the Swarm Plots
        if os.path.exists(reward_file):
            with open(reward_file, 'r') as f:
                reward_data = json.load(f)

                if target_voltage is None:
                    ac = reward_data.get('active_constraints', {})
                    target_voltage = ac.get('vout_target')
                    target_efficiency = ac.get('efficiency_target')

                for cid, cdata in reward_data.get('circuits', {}).items():
                    metrics = cdata.get('raw_metrics', {}).copy()
                    metrics['fitness'] = cdata.get('fitness_score')

                    if metrics['fitness'] is not None:
                        net_path = os.path.join(llm_out_dir, f"{cid}.net")
                        if not os.path.exists(net_path):
                             net_path = os.path.join(folder, f"{cid}.net") 
                             
                        if os.path.exists(net_path):
                            net_text = open(net_path, "r", encoding="utf-8").read()
                            t_hash = get_topological_hash(net_text)

                            # This logic ensures the swarm plot only shows each topology once
                            if t_hash not in global_seen_hashes:
                                global_seen_hashes.add(t_hash)
                                new_cands[cid] = metrics
                                cumulative_db[cid] = metrics
                                b_fit_batch.append(metrics['fitness'])

                                if val_data.get(cid, {}).get("passed", False):
                                    batch_unique_valid += 1
                                    
        unique_valid_counts.append(batch_unique_valid)

        history[b_idx] = {
            'db_state': cumulative_db.copy(),
            'new_cands': list(new_cands.keys()),
            'sampled_parents': get_sampled_parents(folder)
        }
        
        batch_avg_fitness.append(sum(b_fit_batch) / len(b_fit_batch) if b_fit_batch else None)
        batch_max_fitness.append(max(b_fit_batch) if b_fit_batch else None)
        
        b_fit_global = [m['fitness'] for m in cumulative_db.values()]
        global_avg_fitness.append(sum(b_fit_global) / len(b_fit_global) if b_fit_global else None)
        global_max_fitness.append(max(b_fit_global) if b_fit_global else None)


    def plot_swarm(metric_key, title, ylabel, yscale='linear', target_val=None, y_limits=None, filename=''):
        plt.figure(figsize=(14, 8)) 
        
        node_coords = {b: {} for b in batches}
        
        for b_idx in batches:
            for cid, m in history[b_idx]['db_state'].items():
                val = m.get(metric_key)
                if val is not None:
                    node_coords[b_idx][cid] = (b_idx, val)

        for i, b_idx in enumerate(batches):
            data = history[b_idx]
            
            if i > 0:
                prev_b_idx = batches[i-1]
                for pid in data['sampled_parents']:
                    parent_match = next((c for c in node_coords[prev_b_idx] if pid in c or c in pid), None)
                    if parent_match:
                        p_x, p_y = node_coords[prev_b_idx][parent_match]
                        for child_cid in data['new_cands']:
                            if child_cid in node_coords[b_idx]:
                                c_x, c_y = node_coords[b_idx][child_cid]
                                plt.plot([p_x, c_x], [p_y, c_y], color='gray', linestyle='--', alpha=0.35, zorder=1)

            next_b_idx = batches[i+1] if i + 1 < len(batches) else None
            sampled_for_next = history[next_b_idx]['sampled_parents'] if next_b_idx else []

            for cid, (x, y) in node_coords[b_idx].items():
                is_sampled = any(sp in cid or cid in sp for sp in sampled_for_next)
                
                color = 'crimson' if is_sampled else 'dimgray'
                alpha = 0.9 if is_sampled else 0.8 
                size = 80 if is_sampled else 40
                
                plt.scatter(x, y, color=color, alpha=alpha, s=size, edgecolor='white', linewidth=0.5, zorder=2)
                
                short_id = shorten_cid(cid)
                plt.annotate(short_id, (x, y), xytext=(6, 0), textcoords="offset points", 
                             fontsize=7, alpha=0.85, va='center', zorder=4)
                
        if target_val is not None:
            plt.axhline(y=target_val, color='red', linestyle='--', linewidth=2, label=f'Target ({target_val})', zorder=3)
            plt.legend(loc='upper right')

        if y_limits:
            plt.ylim(y_limits)
            
        plt.yscale(yscale)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Batch Number', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(batches)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    plot_swarm('fitness', 'Database Fitness Progression (Full Scale)', 'Fitness Score', y_limits=(-1.05, 1.05), filename='0_swarm_fitness_full.png')
    plot_swarm('fitness', 'Database Fitness Progression (Valid Only)', 'Fitness Score', y_limits=(0.48, 1.02), filename='0_swarm_fitness_zoomed.png')
    plot_swarm('simulation_output_voltage', 'Output Voltage Spread', 'Voltage (V)', target_val=target_voltage, filename='0_swarm_voltage.png')
    plot_swarm('efficiency', 'Efficiency Spread', 'Efficiency', target_val=target_efficiency, filename='0_swarm_efficiency.png')
    plot_swarm('total_volume_cm3', 'Volume Spread (Log Scale)', 'Volume (cm³)', yscale='log', filename='0_swarm_volume.png')
    plot_swarm('total_components', 'Component Count Spread', 'Number of Components', filename='0_swarm_components.png')

    plt.figure(figsize=(9, 6))
    plt.bar(batches, valid_counts, color='lightgray', edgecolor='black', label='Total Valid Generated')
    plt.bar(batches, unique_valid_counts, color='mediumseagreen', alpha=0.9, edgecolor='black', label='Unique Valid Topologies')
    plt.title('Topology Yield: Unique vs Total Valid', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Valid Circuits', fontsize=12)
    plt.xticks(batches)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(results_dir, '0_summary_valid_topologies.png'), dpi=300, bbox_inches='tight')
    plt.close()

    try:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        
        # Axis 1: Validity (Greens)
        ax1.plot(batches, batch_validity_rates, color='mediumseagreen', marker='o', linewidth=2.5, label='Batch Validity Rate')
        ax1.plot(batches, cumulative_validity_rates, color='darkgreen', marker='^', linestyle='--', linewidth=2, label='Cumulative Validity Rate')
        
        ax1.set_xlabel('Batch Number', fontsize=12)
        ax1.set_ylabel('Validity Rate (%)', fontsize=12, color='darkgreen')
        ax1.tick_params(axis='y', labelcolor='darkgreen')
        ax1.set_ylim(0, 105) # Lock strictly to 100% since denominator is fixed
        ax1.set_xticks(batches)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Axis 2: Uniqueness (Reds)
        ax2 = ax1.twinx()
        ax2.plot(batches, batch_uniqueness_rates, color='lightcoral', marker='d', linestyle='-', linewidth=2.5, label='Batch Uniqueness Rate')
        ax2.plot(batches, cumulative_uniqueness_rates, color='crimson', marker='s', linestyle='--', linewidth=2.5, label='Cumulative Uniqueness Rate')
        
        ax2.set_ylabel('Uniqueness Rate (%)', fontsize=12, color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        # Ensure the Uniqueness axis scales cleanly
        max_uniq = max(max(batch_uniqueness_rates, default=0), max(cumulative_uniqueness_rates, default=0))
        ax2.set_ylim(0, max_uniq * 1.2 if max_uniq > 0 else 100)
        
        plt.title('Exploration vs. Yield: Uniqueness & Validity Rates', fontsize=14, fontweight='bold')
        
        # Combine legends neatly below the plot
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        
        plt.savefig(os.path.join(results_dir, '0_summary_yield_rates.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  ⚠️ Could not generate yield plot: {e}")

    plt.figure(figsize=(11, 7))
    plt.plot(batches, batch_avg_fitness, label='Batch Avg Fitness', marker='o', linestyle='-', color='royalblue', alpha=0.7)
    plt.plot(batches, batch_max_fitness, label='Batch Max Fitness', marker='^', linestyle='-', color='darkblue', alpha=0.7)
    
    plt.plot(batches, global_avg_fitness, label='Global Avg Fitness', marker='s', linestyle='--', color='mediumseagreen')
    plt.plot(batches, global_max_fitness, label='Global Best Fitness', marker='*', linestyle='-', color='gold', markeredgecolor='black', markersize=12)
    
    plt.title('Overall Summary: Fitness Progression (Batch vs Global)', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.xticks(batches)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, '0_summary_fitness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Completed swarm visualizations for {os.path.basename(run_dir)}")

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    target_session = "zycos_010" 
    session_path = os.path.join(DATA_DIR, target_session)
    
    if not os.path.exists(session_path):
        print(f"❌ Could not find session folder at: {session_path}")
    else:
        print(f"🔍 Found session {target_session}. Extracting data from all runs...")
        run_folders = sorted(glob.glob(os.path.join(session_path, 'Run_*')))
        
        for run_dir in run_folders:
            print(f"\n[{os.path.basename(run_dir)}]")
            plot_run_results(run_dir)
            
        print(f"\n🎉 Done! All {len(run_folders)} runs have been fully visualized.")