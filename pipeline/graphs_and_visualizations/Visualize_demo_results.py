import os
import json
import glob
import matplotlib.pyplot as plt
import re

# --- Configuration ---
# Path points to the data directory based on image_f3da75.png
DATA_DIR = os.path.join('pipeline', 'data')

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

    # Lists to store our aggregated data
    batches = []
    avg_fitness_history = []
    best_fitness_history = []
    all_voltages_history = []
    all_volumes_history = []
    all_components_history = []
    
    target_voltage = None

    for folder in batch_folders:
        batch_num = extract_batch_number(os.path.basename(folder))
        
        # As per image_f3da75.png, the files are inside the "LLM_output" directory
        reward_file = os.path.join(folder, 'LLM_output', 'reward_results.json')

        try:
            with open(reward_file, 'r') as f:
                reward_data = json.load(f)
            
            # Extract target voltage from the first batch's constraints
            if target_voltage is None:
                target_voltage = reward_data.get('active_constraints', {}).get('vout_target', 5.0)

            circuits = reward_data.get('circuits', {})
            
            batch_fitness = []
            batch_voltages = []
            batch_volumes = []
            batch_components = []

            for circ_id, circ_data in circuits.items():
                batch_fitness.append(circ_data['fitness_score'])
                
                raw_metrics = circ_data.get('raw_metrics', {})
                batch_voltages.append(raw_metrics.get('simulation_output_voltage', 0))
                
                # Handle potential excessively large volumes or inf
                vol = raw_metrics.get('total_volume_cm3', 0)
                batch_volumes.append(vol)
                
                batch_components.append(raw_metrics.get('total_components', 0))
            
            # Only append to main history if we found circuit data
            if batch_fitness:
                batches.append(batch_num)
                avg_fitness_history.append(sum(batch_fitness) / len(batch_fitness))
                best_fitness_history.append(max(batch_fitness)) # Using max since higher is usually better, change to min if lower score is better
                all_voltages_history.append(batch_voltages)
                all_volumes_history.append(batch_volumes)
                all_components_history.append(batch_components)

        except FileNotFoundError:
            print(f"Warning: reward_results.json not found for batch {batch_num}. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON in batch {batch_num}. Skipping.")
        except KeyError as e:
            print(f"Warning: Missing expected data key {e} in batch {batch_num}. Skipping.")

    if not batches:
        print("No valid data could be extracted to plot.")
        return

    # -----------------------------------------------------------------
    # PLOTTING SECTION
    # -----------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolutionary Pipeline Metrics Over Time', fontsize=16)

    # Plot 1: Average and Best Fitness
    axs[0, 0].plot(batches, avg_fitness_history, label='Average Fitness', marker='o', linestyle='-', color='blue')
    axs[0, 0].plot(batches, best_fitness_history, label='Best Fitness', marker='*', linestyle='--', color='green', markersize=8)
    axs[0, 0].set_title('Fitness Score Over Batches')
    axs[0, 0].set_xlabel('Batch Number')
    axs[0, 0].set_ylabel('Fitness Score')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # Plot 2: Output Voltage (Scatter)
    for i, batch_voltages in enumerate(all_voltages_history):
        x_coords = [batches[i]] * len(batch_voltages)
        axs[0, 1].scatter(x_coords, batch_voltages, color='dodgerblue', alpha=0.7)
    
    if target_voltage is not None:
        axs[0, 1].axhline(y=target_voltage, color='red', linestyle='--', label=f'Target ({target_voltage}V)')
        axs[0, 1].legend()
        
    axs[0, 1].set_title('Output Voltage of Candidate Circuits')
    axs[0, 1].set_xlabel('Batch Number')
    axs[0, 1].set_ylabel('Voltage (V)')
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Total Volume (Scatter)
    for i, batch_volumes in enumerate(all_volumes_history):
        x_coords = [batches[i]] * len(batch_volumes)
        axs[1, 0].scatter(x_coords, batch_volumes, color='darkorange', alpha=0.7)
    
    # Optional: Use a log scale for volume if some values are astronomically high (like the 10000.0 in top3)
    axs[1, 0].set_yscale('log') 
    axs[1, 0].set_title('Total Volume Over Batches (Log Scale)')
    axs[1, 0].set_xlabel('Batch Number')
    axs[1, 0].set_ylabel('Volume (cm³)')
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Number of Components (Scatter)
    for i, batch_components in enumerate(all_components_history):
        x_coords = [batches[i]] * len(batch_components)
        axs[1, 1].scatter(x_coords, batch_components, color='purple', alpha=0.7)
        
    axs[1, 1].set_title('Total Components Per Circuit Over Batches')
    axs[1, 1].set_xlabel('Batch Number')
    axs[1, 1].set_ylabel('Number of Components')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout so suptitle fits cleanly
    plt.show()

if __name__ == '__main__':
    main()