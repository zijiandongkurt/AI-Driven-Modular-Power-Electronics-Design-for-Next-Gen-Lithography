import os
import json
import glob
import matplotlib.pyplot as plt
import re

# --- Configuration ---
# Dynamic pathing relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

def extract_batch_number(folder_name):
    """Extracts the integer batch number from the folder name."""
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else -1

def main():
    # Find all batch folders and sort them numerically
    batch_folders = glob.glob(os.path.join(DATA_DIR, 'batch_*'))
    batch_folders.sort(key=lambda x: extract_batch_number(os.path.basename(x)))

    if not batch_folders:
        print(f"No batch folders found in {DATA_DIR}. Please check your path.")
        return

    # Create the results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"📁 Saving plots to: {RESULTS_DIR}\n")

    # Master list of all batches to keep the x-axis consistent
    batches = [extract_batch_number(os.path.basename(folder)) for folder in batch_folders]
    
    # Lists to store our aggregated data
    avg_fitness_history = []
    best_fitness_history = []
    all_voltages_history = []
    all_volumes_history = []
    all_components_history = []
    valid_counts_history = [] 
    
    target_voltage = None

    # --- DATA EXTRACTION ---
    for folder in batch_folders:
        batch_num = extract_batch_number(os.path.basename(folder))
        
        reward_file = os.path.join(folder, 'reward_results.json')
        valid_file = os.path.join(folder, 'validation_results.json')

        # 1. Read Reward Data (Fitness, Voltage, Volume, Components)
        if os.path.exists(reward_file):
            try:
                with open(reward_file, 'r') as f:
                    reward_data = json.load(f)
                
                # Grab target voltage if we haven't yet
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
                print(f"Warning: Could not parse {reward_file}.")
                avg_fitness_history.append(None)
                best_fitness_history.append(None)
                all_voltages_history.append([])
                all_volumes_history.append([])
                all_components_history.append([])
        else:
            print(f"Notice: {reward_file} missing. Leaving gap in graphs.")
            avg_fitness_history.append(None)
            best_fitness_history.append(None)
            all_voltages_history.append([])
            all_volumes_history.append([])
            all_components_history.append([])

        # 2. Read Validation Data (Valid Topologies Count)
        if os.path.exists(valid_file):
            try:
                with open(valid_file, 'r') as vf:
                    valid_data = json.load(vf)
                valid_count = sum(1 for circ in valid_data.values() if circ.get('passed') is True)
                valid_counts_history.append(valid_count)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {valid_file}.")
                valid_counts_history.append(0)
        else:
            print(f"Notice: {valid_file} missing. Recording 0 valid circuits.")
            valid_counts_history.append(0)

    # Make sure we use integers for the X-axis
    x_ticks = batches

    # --- PLOTTING SECTION ---
    
    # Plot 1: Fitness
    plt.figure(figsize=(8, 6))
    plt.plot(batches, avg_fitness_history, label='Average Fitness', marker='o', linestyle='-', color='blue')
    plt.plot(batches, best_fitness_history, label='Best Fitness', marker='*', linestyle='--', color='green', markersize=8)
    plt.title('Fitness Score Over Batches')
    plt.xlabel('Batch Number')
    plt.ylabel('Fitness Score')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '01_fitness_scores.png'), dpi=300)
    plt.close()
    print("✅ Saved 01_fitness_scores.png")

    # Plot 2: Output Voltage
    plt.figure(figsize=(8, 6))
    for i, batch_voltages in enumerate(all_voltages_history):
        if batch_voltages: # Only plot if we have data
            plt.scatter([batches[i]] * len(batch_voltages), batch_voltages, color='dodgerblue', alpha=0.7)
    if target_voltage is not None:
        plt.axhline(y=target_voltage, color='red', linestyle='--', label=f'Target ({target_voltage}V)')
        plt.legend()
    plt.title('Output Voltage of Candidate Circuits')
    plt.xlabel('Batch Number')
    plt.ylabel('Voltage (V)')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '02_output_voltage.png'), dpi=300)
    plt.close()
    print("✅ Saved 02_output_voltage.png")

    # Plot 3: Total Volume
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
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '03_total_volume.png'), dpi=300)
    plt.close()
    print("✅ Saved 03_total_volume.png")

    # Plot 4: Components
    plt.figure(figsize=(8, 6))
    for i, batch_components in enumerate(all_components_history):
        if batch_components:
            plt.scatter([batches[i]] * len(batch_components), batch_components, color='purple', alpha=0.7)
    plt.title('Total Components Per Circuit Over Batches')
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Components')
    plt.xticks(x_ticks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '04_total_components.png'), dpi=300)
    plt.close()
    print("✅ Saved 04_total_components.png")

    # Plot 5: Valid Topologies
    plt.figure(figsize=(8, 6))
    plt.bar(batches, valid_counts_history, color='mediumseagreen', alpha=0.8, edgecolor='black')
    plt.title('Amount of Valid Topologies Per Batch')
    plt.xlabel('Batch Number')
    plt.ylabel('Valid Circuits')
    plt.xticks(x_ticks)
    max_valid = max(valid_counts_history) if valid_counts_history else 4
    plt.yticks(range(0, max_valid + 2)) 
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '05_valid_topologies.png'), dpi=300)
    plt.close()
    print("✅ Saved 05_valid_topologies.png")

    print(f"\n🎉 All charts generated successfully in {RESULTS_DIR}!")

if __name__ == '__main__':
    main()