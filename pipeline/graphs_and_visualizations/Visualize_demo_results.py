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
    print(f"\n📊 Generating/Updating plots in: {results_dir}")

    # Master list of all batches to keep the x-axis consistent
    batches = [extract_batch_number(os.path.basename(folder)) for folder in batch_folders]
    
    avg_fitness_history, best_fitness_history = [], []
    all_voltages_history, all_volumes_history, all_components_history = [], [], []
    valid_counts_history = [] 
    
    target_voltage = None

    # --- DATA EXTRACTION ---
    for folder in batch_folders:
        reward_file = os.path.join(folder, 'reward_results.json')
        valid_file = os.path.join(folder, 'validation_results.json')

        # 1. Read Reward Data
        if os.path.exists(reward_file):
            try:
                with open(reward_file, 'r') as f:
                    reward_data = json.load(f)
                
                if target_voltage is None:
                    target_voltage = reward_data.get('active_constraints', {}).get('vout_target', 5.0)

                circuits = reward_data.get('circuits', {})
                b_fit, b_volt, b_vol, b_comp = [], [], [], []

                for circ_id, circ_data in circuits.items():
                    b_fit.append(circ_data['fitness_score'])
                    raw_metrics = circ_data.get('raw_metrics', {})
                    b_volt.append(raw_metrics.get('simulation_output_voltage', 0))
                    b_vol.append(raw_metrics.get('total_volume_cm3', 0))
                    b_comp.append(raw_metrics.get('total_components', 0))

                avg_fitness_history.append(sum(b_fit) / len(b_fit) if b_fit else None)
                best_fitness_history.append(max(b_fit) if b_fit else None)
                all_voltages_history.append(b_volt)
                all_volumes_history.append(b_vol)
                all_components_history.append(b_comp)

            except json.JSONDecodeError:
                avg_fitness_history.append(None)
                best_fitness_history.append(None)
                all_voltages_history.append([])
                all_volumes_history.append([])
                all_components_history.append([])
        else:
            avg_fitness_history.append(None)
            best_fitness_history.append(None)
            all_voltages_history.append([])
            all_volumes_history.append([])
            all_components_history.append([])

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

    x_ticks = batches

    # --- PLOTTING SECTION ---
    # Using a helper to avoid repetitive code
    def save_plot(name):
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, name), dpi=300)
        plt.close()

    # 1. Fitness
    plt.figure(figsize=(8, 6))
    plt.plot(batches, avg_fitness_history, label='Average Fitness', marker='o', linestyle='-', color='blue')
    plt.plot(batches, best_fitness_history, label='Best Fitness', marker='*', linestyle='--', color='green', markersize=8)
    plt.title('Fitness Score Over Batches')
    plt.xlabel('Batch Number')
    plt.ylabel('Fitness Score')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot('01_fitness_scores.png')

    # 2. Voltage
    plt.figure(figsize=(8, 6))
    for i, batch_voltages in enumerate(all_voltages_history):
        if batch_voltages:
            plt.scatter([batches[i]] * len(batch_voltages), batch_voltages, color='dodgerblue', alpha=0.7)
    if target_voltage is not None:
        plt.axhline(y=target_voltage, color='red', linestyle='--', label=f'Target ({target_voltage}V)')
        plt.legend()
    plt.title('Output Voltage of Candidate Circuits')
    plt.xlabel('Batch Number')
    plt.ylabel('Voltage (V)')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    save_plot('02_output_voltage.png')

    # 3. Volume
    plt.figure(figsize=(8, 6))
    for i, batch_volumes in enumerate(all_volumes_history):
        if batch_volumes:
            plt.scatter([batches[i]] * len(batch_volumes), batch_volumes, color='darkorange', alpha=0.7)
    plt.yscale('log') 
    plt.title('Total Volume Over Batches (Log Scale)')
    plt.xlabel('Batch Number')
    plt.ylabel('Volume (cm³)')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    save_plot('03_total_volume.png')

    # 4. Components
    plt.figure(figsize=(8, 6))
    for i, batch_components in enumerate(all_components_history):
        if batch_components:
            plt.scatter([batches[i]] * len(batch_components), batch_components, color='purple', alpha=0.7)
    plt.title('Total Components Per Circuit Over Batches')
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Components')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    save_plot('04_total_components.png')

    # 5. Valid Topologies
    plt.figure(figsize=(8, 6))
    plt.bar(batches, valid_counts_history, color='mediumseagreen', alpha=0.8, edgecolor='black')
    plt.title('Amount of Valid Topologies Per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Valid Circuits')
    plt.xticks(x_ticks)
    max_valid = max(valid_counts_history) if valid_counts_history else 4
    plt.yticks(range(0, max_valid + 2)) 
    plt.grid(axis='y', alpha=0.3)
    save_plot('05_valid_topologies.png')

# Allow it to still be run directly from the command line on the most recent run
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