import pandas as pd
import numpy as np
import os

def calculate_reward(group_df, constraints, weights):
    """
    Calculates the fitness score including the l5 inefficiency penalty.
    Returns both the base_fitness (hardware constraints only) and final_fitness (with efficiency).
    """
    required_cols = ['V(2)', 'I(L1)', 'V(1)', 'I(Vin)', 'I(Rload)']
    
    # Pre-check for required columns and enough data
    if not all(col in group_df.columns for col in required_cols):
        return 0.0

    clean_df = group_df[required_cols].dropna()
    if len(clean_df) < 10: 
        return 0.0

    v_in = clean_df['V(1)']
    v_out = clean_df['V(2)']
    i_in = clean_df['I(Vin)']
    i_out = clean_df['I(Rload)']
    i_L1 = clean_df['I(L1)']

    # Extract needed parameters
    # take the mean at the end of the simulation, look at the steady state value
    # take the last 100 - 150 samples to avoid transient
    v_in_mean = v_in.mean()
    il1_p2p = i_L1.max() - i_L1.min()
    ivin_abs_peak = i_in.abs().max()

    # Calculate Efficiency
    instant_p_in = -v_in * i_in
    instant_p_out = v_out * i_out
    
    avg_p_in = instant_p_in.mean()
    avg_p_out = instant_p_out.mean()
    
    efficiency = 0.0
    epsilon = 1e-9 
    if avg_p_in > epsilon:
        efficiency = np.clip(avg_p_out / avg_p_in, 0.0, 1.0)

    # Calculate Hardware Penalties (l1 - l4)
    penalty_v_out = ((v_out - constraints['target_v_out']) ** 2).mean()
    
    penalty_v_in = 0.0
    if abs(v_in_mean - constraints['target_v_in']) > constraints['v_in_tol']:
        penalty_v_in = (abs(v_in_mean - constraints['target_v_in']) - constraints['v_in_tol']) ** 2

    # might be incorrect if multiple inductors 
    penalty_il1_ripple = 0.0
    if il1_p2p > constraints['max_il1_ripple']:
        penalty_il1_ripple = (il1_p2p - constraints['max_il1_ripple']) ** 2

    penalty_ivin_peak = 0.0
    if ivin_abs_peak > constraints['max_ivin_peak']:
        penalty_ivin_peak = (ivin_abs_peak - constraints['max_ivin_peak']) ** 2

    l1 = weights.get('v_out', 1.0) * penalty_v_out
    l2 = weights.get('v_in', 1.0) * penalty_v_in
    l3 = weights.get('il1_ripple', 1.0) * penalty_il1_ripple
    l4 = weights.get('ivin_peak', 1.0) * penalty_ivin_peak
    l5 = weights.get('efficiency', 0.0) * (1.0 - efficiency)
    
    # Final fitness combines hardware constraint losses with efficiency loss
    fitness = float(1.0 / (1.0 + l1 + l2 + l3 + l4 + l5))
    
    return fitness

def process_csv_and_calculate_fitness(csv_file_path, constraints, weights):
    """
    Reads a CSV file, groups data by 'source_file', and calculates
    a fitness score using dynamic constraints and dynamic penalty weights.
    Returns a list of [source_file, final_fitness_score].
    """
    fitness_results = []
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return [["Error", "File not found"]]
    except Exception as e:
        return [["Error", f"Could not read CSV: {e}"]]

    if 'source_file' not in df.columns:
        return [["Error", "Missing required 'source_file' column"]]

    # Ensure temporal sequence is maintained
    df = df.sort_values(by=['source_file', 'time'])
    grouped = df.groupby('source_file')

    # Calculate for each unique source_file run
    for source_file_name, group_df in grouped:
        # Unpack both, but only append the final_fitness to the list
        base_fitness, final_fitness = calculate_reward(group_df, constraints, weights)
        fitness_results.append([source_file_name, final_fitness])

    return fitness_results


# --- Example Execution Setup ---

my_constraints = {
    'target_v_in': 12.0,   
    'v_in_tol': 0.5,       
    'target_v_out': 24.0,  
    'max_il1_ripple': 1.0, 
    'max_ivin_peak': 10.0  
}

# Standard weights, including the efficiency weight parameter
my_weights = {
    'v_out': 1.0,
    'v_in': 1.0,
    'il1_ripple': 1.0,
    'ivin_peak': 1.0,
    'efficiency': 0.5  # Adjust this dynamically in your external fine-tuning code
}

# Dynamically get the folder where this specific script is saved
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join that folder path with the file name
csv_file_path = os.path.join(script_dir, 'batch_001_out.csv')

# Execute
results = process_csv_and_calculate_fitness(csv_file_path, my_constraints, my_weights)

print(results)