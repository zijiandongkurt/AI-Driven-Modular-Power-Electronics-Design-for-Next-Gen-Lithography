import os
import json
import glob
import matplotlib.pyplot as plt
import re

def extract_batch_number(folder_name):
    """Extracts the integer batch number from the folder name."""
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else -1

def plot_run_results(run_dir):
    """
    Scans the specified run_dir for batch folders, aggregates the JSON results, 
    and saves the plots into run_dir/results/.
    """
    # Find all batch folders in the specific Run_XXX directory and sort them
    batch_folders = glob.glob(os.path.join(run_dir, 'batch_*'))
    batch_folders.sort(key=lambda x: extract_batch_number(os.path.basename(x)))

    if not batch_folders:
        print(f"No batch folders found in {run_dir}. Skipping plotting.")
        return

    # Create the results directory inside the run folder
    results_dir = os.path.join(run_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📊 Generating plots with consistent candidate tracking in: {results_dir}")

    # Master list of all batches to keep the x-axis consistent
    batches = [extract_batch_number(os.path.basename(folder)) for folder in batch_folders]
    
    avg_fitness_history, best_fitness_history = [], []
    valid_counts_history = [] 
    
    target_voltage = None
    target_efficiency = None

    # Track metrics by candidate lineage: cand_data[cand_key][metric] = [val1, val2, ...]
    cand_data = {}

    # --- DATA EXTRACTION ---
    for i, folder in enumerate(batch_folders):
        reward_file = os.path.join(folder, 'reward_results.json')
        valid_file = os.path.join(folder, 'validation_results.json')

        # 1. Read Reward Data
        b_fit = []
        if os.path.exists(reward_file):
            try:
                with open(reward_file, 'r') as f:
                    reward_data = json.load(f)
                
                # Grab targets on the first successful read
                if target_voltage is None:
                    active_const = reward_data.get('active_constraints', {})
                    target_voltage = active_const.get('vout_target')
                    target_efficiency = active_const.get('efficiency_target')

                circuits = reward_data.get('circuits', {})

                for circ_id, circ_data in circuits.items():
                    # Extract the candidate identifier (e.g., 'cand1')
                    cand_match = re.search(r'cand\d+', circ_id)
                    cand_key = cand_match.group() if cand_match else circ_id
                    
                    # Initialize history arrays with None for this candidate if new
                    if cand_key not in cand_data:
                        cand_data[cand_key] = {
                            'fitness': [None] * len(batches),
                            'voltage': [None] * len(batches),
                            'volume': [None] * len(batches),
                            'components': [None] * len(batches),
                            'efficiency': [None] * len(batches),
                        }
                    
                    # Extract Fitness
                    fit = circ_data.get('fitness_score')
                    if fit is not None:
                        b_fit.append(fit)
                        cand_data[cand_key]['fitness'][i] = fit
                    
                    # Extract simulation metrics
                    raw_metrics = circ_data.get('raw_metrics')
                    if raw_metrics:
                        cand_data[cand_key]['voltage'][i] = raw_metrics.get('simulation_output_voltage')
                        cand_data[cand_key]['volume'][i] = raw_metrics.get('total_volume_cm3')
                        cand_data[cand_key]['components'][i] = raw_metrics.get('total_components')
                        cand_data[cand_key]['efficiency'][i] = raw_metrics.get('efficiency')

            except json.JSONDecodeError:
                pass

        # Overall summary tracking (ignoring missing values)
        avg_fitness_history.append(sum(b_fit) / len(b_fit) if b_fit else None)
        best_fitness_history.append(max(b_fit) if b_fit else None)

        # 2. Read Validation Data
        if os.path.exists(valid_file):
            try:
                with open(valid_file, 'r') as vf:
                    valid_data = json.load(vf)
                valid_count = sum(1 for circ in valid_data.values() if circ.get('passed') is True)
                valid_counts_history.append(valid_count)
            except json.JSONDecodeError:
                valid_counts_history.append(0)
        else:
            valid_counts_history.append(0)

    # --- PLOTTING PREPARATION ---
    x_ticks = batches

    # Create a consistent color mapping for all candidates found
    cand_keys = sorted(list(cand_data.keys()))
    cmap = plt.get_cmap('tab10')
    color_map = {key: cmap(idx % 10) for idx, key in enumerate(cand_keys)}

    def save_plot(name):
        # bbox_inches='tight' ensures the outside legend doesn't get cut off
        plt.savefig(os.path.join(results_dir, name), dpi=300, bbox_inches='tight')
        plt.close()

    # Generic plot generator to easily handle all per-candidate charts
    def plot_metric(metric_name, title, ylabel, yscale='linear', target_val=None, target_label=None):
        plt.figure(figsize=(9, 6))
        
        for cand, metrics in cand_data.items():
            # Filter out missing batches to draw a continuous line between successful batches
            x_valid = [b for b, v in zip(batches, metrics[metric_name]) if v is not None]
            y_valid = [v for v in metrics[metric_name] if v is not None]
            
            if x_valid:
                plt.plot(x_valid, y_valid, marker='o', linestyle='-', color=color_map[cand], label=cand)
        
        if target_val is not None:
            plt.axhline(y=target_val, color='red', linestyle='--', linewidth=2, label=target_label)
        
        plt.yscale(yscale)
        plt.title(title)
        plt.xlabel('Batch Number')
        plt.ylabel(ylabel)
        plt.xticks(x_ticks)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save_plot(f'0_{metric_name}_lineage.png')

    # --- GENERATE LINEAGE PLOTS ---
    plot_metric('fitness', 'Candidate Fitness Score Over Time', 'Fitness Score')
    plot_metric('voltage', 'Candidate Output Voltage Over Time', 'Voltage (V)', target_val=target_voltage, target_label=f'Target ({target_voltage}V)')
    plot_metric('efficiency', 'Candidate Efficiency Over Time', 'Efficiency', target_val=target_efficiency, target_label=f'Target ({target_efficiency})')
    plot_metric('volume', 'Candidate Total Volume Over Time (Log Scale)', 'Volume (cm³)', yscale='log')
    plot_metric('components', 'Candidate Component Count Over Time', 'Number of Components')

    # --- GENERATE SUMMARY PLOTS ---
    
    # Summary Fitness (Average vs Best)
    plt.figure(figsize=(9, 6))
    plt.plot(batches, avg_fitness_history, label='Batch Average Fitness', marker='o', linestyle='-', color='blue')
    plt.plot(batches, best_fitness_history, label='Batch Best Fitness', marker='*', linestyle='--', color='green', markersize=8)
    plt.title('Overall Summary: Fitness Progression')
    plt.xlabel('Batch Number')
    plt.ylabel('Fitness Score')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot('0_summary_fitness.png')

    # Valid Topologies Bar Chart
    plt.figure(figsize=(9, 6))
    plt.bar(batches, valid_counts_history, color='mediumseagreen', alpha=0.8, edgecolor='black')
    plt.title('Amount of Valid Topologies Per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Valid Circuits')
    plt.xticks(x_ticks)
    max_valid = max(valid_counts_history) if valid_counts_history else 4
    plt.yticks(range(0, max_valid + 2)) 
    plt.grid(axis='y', alpha=0.3)
    save_plot('0_valid_topologies_bar.png')


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'pipeline', 'data'))
    # Find the latest Run_XXX folder
    run_folders = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if re.match(r"Run_\d+", d)]
    if run_folders:
        latest_run = max(run_folders, key=lambda x: int(x.split('_')[-1]))
        plot_run_results(latest_run)
        print("✅ Finished plotting latest run.")
    else:
        print("No run folders found.")