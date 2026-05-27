import os
import json
import glob
import re
import numpy as np

# Use thread-safe, headless backend to prevent Tkinter crashes during ML loops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_batch_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else -1


def shorten_cid(cid):
    """Shortens Phase1_cons4_b2_cand1 into b2_cand1 for cleaner plot labels."""
    # Notice we swapped the order in the regex to look for _b# followed by _cand#
    match = re.search(r'_b(\d+)_cand(\d+)', cid)
    if match:
        return f"b{match.group(1)}_c{match.group(2)}"
    
    return cid[:10] # Fallback


def get_sampled_parents(batch_folder):
    """Finds which parents were sampled to generate this batch."""
    parent_files = glob.glob(os.path.join(batch_folder, 'prompt_parent_*.txt'))
    return [os.path.basename(pf).replace('prompt_parent_', '').replace('.txt', '') for pf in parent_files]


def plot_run_results(run_dir):
    batch_folders = sorted(glob.glob(os.path.join(run_dir, 'batch_*')), key=lambda x: extract_batch_number(os.path.basename(x)))
    if not batch_folders:
        print(f"No batch folders found in {run_dir}. Skipping plotting.")
        return

    results_dir = os.path.join(run_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📊 Generating Swarm Plots with Labels in: {results_dir}")

    batches = []
    history = {}
    cumulative_db = {}
    
    valid_counts = []
    
    # Track Batch-specific vs Global Database metrics
    batch_avg_fitness, batch_max_fitness = [], []
    global_avg_fitness, global_max_fitness = [], []
    
    target_voltage = None
    target_efficiency = None

    # --- 1. DATA EXTRACTION ---
    for folder in batch_folders:
        b_idx = extract_batch_number(os.path.basename(folder))
        batches.append(b_idx)
        
        reward_file = os.path.join(folder, 'reward_results.json')
        valid_file = os.path.join(folder, 'validation_results.json')
        
        new_cands = {}
        b_fit_batch = [] # Only the fitness of circuits IN THIS BATCH
        
        # Read Rewards and Metrics
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
                        new_cands[cid] = metrics
                        cumulative_db[cid] = metrics
                        b_fit_batch.append(metrics['fitness'])

        # Read Validation Yield
        val_count = 0
        if os.path.exists(valid_file):
            with open(valid_file, 'r') as f:
                val_data = json.load(f)
                val_count = sum(1 for v in val_data.values() if v.get('passed', False))
        valid_counts.append(val_count)

        # Snapshot history for this batch
        history[b_idx] = {
            'db_state': cumulative_db.copy(),
            'new_cands': list(new_cands.keys()),
            'sampled_parents': get_sampled_parents(folder)
        }
        
        # Calculate Batch-specific fitness
        batch_avg_fitness.append(sum(b_fit_batch) / len(b_fit_batch) if b_fit_batch else None)
        batch_max_fitness.append(max(b_fit_batch) if b_fit_batch else None)
        
        # Calculate Global (Cumulative) fitness
        b_fit_global = [m['fitness'] for m in cumulative_db.values()]
        global_avg_fitness.append(sum(b_fit_global) / len(b_fit_global) if b_fit_global else None)
        global_max_fitness.append(max(b_fit_global) if b_fit_global else None)


    # --- 2. SWARM PLOT GENERATOR ---
    def plot_swarm(metric_key, title, ylabel, yscale='linear', target_val=None, y_limits=None, filename=''):
        plt.figure(figsize=(14, 8)) # Slightly wider to accommodate labels
        
        node_coords = {b: {} for b in batches}
        
        # Pass 1: Assign exact X and Y coordinates to every valid dot
        for b_idx in batches:
            for cid, m in history[b_idx]['db_state'].items():
                val = m.get(metric_key)
                if val is not None:
                    node_coords[b_idx][cid] = (b_idx, val)

        # Pass 2: Draw ancestry lines and plot dots
        for i, b_idx in enumerate(batches):
            data = history[b_idx]
            
            # Draw Ancestry Lines
            if i > 0:
                prev_b_idx = batches[i-1]
                for pid in data['sampled_parents']:
                    parent_match = next((c for c in node_coords[prev_b_idx] if pid in c or c in pid), None)
                    if parent_match:
                        p_x, p_y = node_coords[prev_b_idx][parent_match]
                        for child_cid in data['new_cands']:
                            if child_cid in node_coords[b_idx]:
                                c_x, c_y = node_coords[b_idx][child_cid]
                                plt.plot([p_x, c_x], [p_y, c_y], color='gray', linestyle='--', alpha=0.25, zorder=1)

            next_b_idx = batches[i+1] if i + 1 < len(batches) else None
            sampled_for_next = history[next_b_idx]['sampled_parents'] if next_b_idx else []

            # Draw Dots & Text Labels
            for cid, (x, y) in node_coords[b_idx].items():
                is_sampled = any(sp in cid or cid in sp for sp in sampled_for_next)
                
                color = 'crimson' if is_sampled else 'dimgray'
                alpha = 0.9 if is_sampled else 0.8 
                size = 80 if is_sampled else 40
                
                plt.scatter(x, y, color=color, alpha=alpha, s=size, edgecolor='white', linewidth=0.5, zorder=2)
                
                # --- NEW: Add the Shortened Text Label ---
                short_id = shorten_cid(cid)
                plt.annotate(short_id, (x, y), xytext=(6, 0), textcoords="offset points", 
                             fontsize=7, alpha=0.85, va='center', zorder=4)
                
        # Draw Target Lines
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


    # --- 3. EXECUTE PLOTS ---
    # Swarm Plots
    plot_swarm('fitness', 'Database Fitness Progression (Full Scale)', 'Fitness Score', y_limits=(-1.05, 1.05), filename='0_swarm_fitness_full.png')
    plot_swarm('fitness', 'Database Fitness Progression (Valid Only)', 'Fitness Score', y_limits=(0.48, 1.02), filename='0_swarm_fitness_zoomed.png')
    plot_swarm('simulation_output_voltage', 'Output Voltage Spread', 'Voltage (V)', target_val=target_voltage, filename='0_swarm_voltage.png')
    plot_swarm('efficiency', 'Efficiency Spread', 'Efficiency', target_val=target_efficiency, filename='0_swarm_efficiency.png')
    plot_swarm('total_volume_cm3', 'Volume Spread (Log Scale)', 'Volume (cm³)', yscale='log', filename='0_swarm_volume.png')
    plot_swarm('total_components', 'Component Count Spread', 'Number of Components', filename='0_swarm_components.png')

    # Summary: Valid Topologies Bar Chart
    plt.figure(figsize=(9, 6))
    plt.bar(batches, valid_counts, color='mediumseagreen', alpha=0.8, edgecolor='black')
    plt.title('Valid Topologies Per Batch', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Valid Circuits', fontsize=12)
    plt.xticks(batches)
    max_valid = max(valid_counts) if valid_counts else 4
    plt.yticks(range(0, int(max_valid) + 2))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(results_dir, '0_summary_valid_topologies.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- NEW: Multi-line Fitness Progress (Batch vs Global) ---
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

    print("✅ Finished plotting all swarm and summary metrics.")


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    target_run = None 
    
    if target_run:
        run_path = os.path.join(DATA_DIR, target_run)
    else:
        run_folders = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if re.match(r"Run_\d+", d)]
        run_path = max(run_folders, key=lambda x: int(x.split('_')[-1])) if run_folders else None

    if run_path and os.path.exists(run_path):
        plot_run_results(run_path)
    else:
        print(f"❌ Error: Could not find valid run data in {DATA_DIR}")