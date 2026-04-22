import pandas as pd
import numpy as np
import os

class RewardFunction:
    def __init__(self):
        # Define a maximum penalty cap to prevent network gradients from exploding
        self.MAX_PENALTY = 10000.0 

    def calculate_loss(self, row, constraints, weights):
        """
        Calculates the total loss as a weighted sum of penalties.
        A lower loss means a better circuit.
        """
        # --- 1. Safely extract metrics, converting NaNs to 0.0 ---
        # Pandas sometimes reads empty CSV cells as NaN instead of missing keys
        v_out = row.get('voltage_out_mean_V', 0.0)
        v_out = 0.0 if pd.isna(v_out) else v_out
        
        efficiency = row.get('efficiency', 0.0)
        efficiency = 0.0 if pd.isna(efficiency) else efficiency
        
        total_volume = row.get('total_volume_cm3', self.MAX_PENALTY)
        total_volume = self.MAX_PENALTY if pd.isna(total_volume) else total_volume
        
        count_mosfets = row.get('count_mosfets', 0)
        count_diodes = row.get('count_diodes', 0)
        count_inductors = row.get('count_inductors', 0)
        count_capacitors = row.get('count_capacitors', 0)

        # --- 2. Hardware Penalty: Voltage accuracy ---
        target_v_out = constraints.get('vout_target', 5.0)
        penalty_v_out = (v_out - target_v_out) ** 2
        
        # --- 3. Efficiency Penalty ---
        safe_efficiency = max(0.0, min(1.0, float(efficiency)))
        penalty_efficiency = 1.0 - safe_efficiency

        # --- 4. Volume Penalty (THE INFINITY FIX) ---
        # If the colleague's script output 'inf' because it melted, cap it at MAX_PENALTY
        if np.isinf(total_volume):
            penalty_volume = self.MAX_PENALTY
        else:
            # Also cap it just in case it's a valid but astronomically high number
            penalty_volume = min(float(total_volume), self.MAX_PENALTY)

        # --- 5. Custom Component Penalty ---
        comp_weights = weights.get('components', {})
        penalty_components = (
            comp_weights.get('mosfet', 3.0) * count_mosfets +
            comp_weights.get('diode', 1.5) * count_diodes +
            comp_weights.get('inductor', 2.5) * count_inductors +
            comp_weights.get('capacitor', 1.0) * count_capacitors
        )

        # --- 6. Apply top-level weights to all penalties ---
        l1 = weights.get('v_out', 1.0) * penalty_v_out
        l2 = weights.get('efficiency', 5.0) * penalty_efficiency
        l3 = weights.get('volume', 0.1) * penalty_volume
        l4 = weights.get('cost', 1.0) * penalty_components
        
        total_loss = l1 + l2 + l3 + l4
        
        # Final safety net: If loss somehow still became NaN or Inf, cap it
        if np.isinf(total_loss) or pd.isna(total_loss):
            return self.MAX_PENALTY
            
        return total_loss

    def calculate_reward(self, row, constraints, weights):
        loss = self.calculate_loss(row, constraints, weights)
        return -loss

    def process_csv_and_calculate_reward(self, csv_file_path, constraints, weights):
        """
        Reads a CSV file, processes data by 'source_file', and calculates
        the final reward using the weighted sum loss.
        """
        results = []
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            return [["Error", "File not found"]]
        except Exception as e:
            return [["Error", f"Could not read CSV: {e}"]]

        if 'source_file' not in df.columns:
            return [["Error", "Missing required 'source_file' column"]]

        # Group by source_file to handle potential .step parameter sweeps.
        grouped = df.groupby('source_file').mean(numeric_only=True)

        # Calculate for each unique source_file run
        for source_file_name, row in grouped.iterrows():
            # Get the reward (negative loss)
            final_reward = self.calculate_reward(row, constraints, weights)
            
            # Optional: You can also append the raw loss if you want to track it
            # raw_loss = self.calculate_loss(row, constraints, weights)
            
            results.append([source_file_name, final_reward])

        return results


# --- Example Execution Setup ---

my_constraints = {
    "vin_min": 12,
    "vin_max": 100,
    "vout_target": 5,
    "efficiency_target": 0.90,
    "power_in": 100,
}

my_weights = {
    'v_out': 10.0,          # Increased weight: Voltage tracking is usually the most critical
    'efficiency': 20.0,     # High weight to heavily incentivize high efficiency
    'volume': 2.0,          # Penalty per cm3
    'cost': 1.0,            # Global multiplier for the custom component sum
    'components': {         # Custom tweakable component penalties
        'mosfet': 5.0,
        'diode': 2.0,
        'inductor': 4.0,
        'capacitor': 1.0
    }
}

# Dynamically get the folder where this specific script is saved
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join that folder path with the file name
csv_file_path = os.path.join(script_dir, 'batch_001_out.csv')

# Execute
reward_function = RewardFunction()
results = reward_function.process_csv_and_calculate_reward(csv_file_path, my_constraints, my_weights)

print(results)