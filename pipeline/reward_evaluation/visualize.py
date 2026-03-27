import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_simulation(csv_file_path, constraints):
    # Load the data
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file_path}")
        return

    # Sort and group by the simulation source file
    df = df.sort_values(by=['source_file', 'time'])
    grouped = df.groupby('source_file')

    required_cols = ['time', 'V(1)', 'V(2)', 'I(L1)', 'I(Vin)']

    for source_file, group_df in grouped:
        # Check if required columns exist
        if not all(col in group_df.columns for col in required_cols):
            print(f"Skipping {source_file}: Missing required columns.")
            continue
            
        # Drop NaNs just like in the reward function
        clean_df = group_df[required_cols].dropna()
        if len(clean_df) == 0:
            print(f"Skipping {source_file}: No valid data after dropping NaNs.")
            continue

        time = clean_df['time']
        v_in = clean_df['V(1)']
        v_out = clean_df['V(2)']
        i_L1 = clean_df['I(L1)']
        i_in = clean_df['I(Vin)']

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Circuit Simulation Analysis: {source_file}', fontsize=16, fontweight='bold')

        # --- 1. Output Voltage (Vout) ---
        ax = axs[0, 0]
        ax.plot(time, v_out, label='V_out (V(2))', color='blue')
        ax.axhline(constraints['target_v_out'], color='green', linestyle='--', label='Target (24V)')
        ax.axhline(constraints['target_v_out'] + constraints['v_out_tol'], color='red', linestyle=':', label='Upper Bound')
        ax.axhline(constraints['target_v_out'] - constraints['v_out_tol'], color='red', linestyle=':', label='Lower Bound')
        ax.set_title('Output Voltage vs Time')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
        ax.grid(True)

        # --- 2. Input Voltage (Vin) ---
        ax = axs[0, 1]
        ax.plot(time, v_in, label='V_in (V(1))', color='orange')
        ax.axhline(constraints['target_v_in'], color='green', linestyle='--', label='Target (12V)')
        ax.axhline(constraints['target_v_in'] + constraints['v_in_tol'], color='red', linestyle=':', label='Upper Bound')
        ax.axhline(constraints['target_v_in'] - constraints['v_in_tol'], color='red', linestyle=':', label='Lower Bound')
        ax.set_title('Input Voltage vs Time')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
        ax.grid(True)

        # --- 3. Inductor Current (IL1) ---
        ax = axs[1, 0]
        ax.plot(time, i_L1, label='I_L1', color='purple')
        
        # Calculate text for peak-to-peak ripple to display on plot
        ripple = i_L1.max() - i_L1.min()
        ax.text(0.05, 0.95, f'P2P Ripple: {ripple:.3f}A\nMax Allowed: {constraints["max_il1_ripple"]}A', 
                transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title('Inductor Current vs Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')
        ax.legend()
        ax.grid(True)

        # --- 4. Input Current (I_Vin) ---
        ax = axs[1, 1]
        ax.plot(time, i_in, label='I_Vin', color='brown')
        ax.axhline(constraints['max_ivin_peak'], color='red', linestyle=':', label='+10A Limit')
        ax.axhline(-constraints['max_ivin_peak'], color='red', linestyle=':', label='-10A Limit')
        ax.set_title('Input Current vs Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    my_constraints = {
        'target_v_in': 12.0,
        'v_in_tol': 0.5,    
        'v_out_tol': 0.1,   
        'target_v_out': 24.0,  
        'max_il1_ripple': 1.0, 
        'max_ivin_peak': 10.0
    }

    # Dynamically get the directory of the script and look for the CSV there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'batch_001_out.csv')
    
    # If it's not in the script directory, fallback to current working directory
    if not os.path.exists(csv_file_path):
        csv_file_path = 'batch_001_out.csv'

    visualize_simulation(csv_file_path, my_constraints)