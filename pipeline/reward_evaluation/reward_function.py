import pandas as pd
import numpy as np
import os
import json

class RewardFunction:
    def __init__(self):
        # Define a maximum penalty cap to prevent network gradients from exploding
        self.MAX_PENALTY = 10000.0 

    def calculate_loss(self, row, constraints, weights):
        """
        Calculates the total loss as a weighted sum of penalties.
        Returns a tuple: (total_loss, details_dict)
        """
        # --- 1. Safely extract metrics, converting NaNs to 0.0 ---
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
        target_efficiency = constraints.get('efficiency_target', 0.90)
        safe_efficiency = max(0.0, min(1.0, float(efficiency)))
        penalty_efficiency = max(0.0, target_efficiency - safe_efficiency)

        # --- 4. Volume Penalty (THE INFINITY FIX) ---
        if np.isinf(total_volume):
            penalty_volume = self.MAX_PENALTY
        else:
            penalty_volume = min(float(total_volume), self.MAX_PENALTY)

        # --- 5. Custom Component Penalty ---
        comp_weights = weights.get('components', {})
        penalty_components = (
            comp_weights.get('mosfet', 1.0) * count_mosfets +
            comp_weights.get('diode', 1.0) * count_diodes +
            comp_weights.get('inductor', 1.0) * count_inductors +
            comp_weights.get('capacitor', 1.0) * count_capacitors
        )

        # --- 6. Apply top-level weights to all penalties ---
        loss_v_out = weights.get('v_out', 1.0) * penalty_v_out
        loss_efficiency = weights.get('efficiency', 1.0) * penalty_efficiency
        loss_volume = weights.get('volume', 1.0) * penalty_volume
        loss_components = weights.get('component_cost', 1.0) * penalty_components
        
        total_loss = loss_v_out + loss_efficiency + loss_volume + loss_components
        
        # Bundle the requested details for optional JSON output
        # (Removed the redundant constraints from this inner dictionary)
        details = {
            "loss_breakdown": {
                "voltage_tracking_loss": float(loss_v_out),
                "efficiency_loss": float(loss_efficiency),
                "volume_loss": float(loss_volume),
                "component_cost_loss": float(loss_components)
            },
            "raw_metrics": {
                "simulation_output_voltage": float(v_out),
                "efficiency": float(safe_efficiency),
                "total_volume_cm3": float(penalty_volume), 
                "total_components": int(count_mosfets + count_diodes + count_inductors + count_capacitors)
            }
        }

        # Final safety net: If loss somehow still became NaN or Inf, cap it
        if np.isinf(total_loss) or pd.isna(total_loss):
            return self.MAX_PENALTY, details
            
        return total_loss, details

    def calculate_reward(self, row, constraints, weights):
        loss, details = self.calculate_loss(row, constraints, weights)
        return -loss, details

    def process_csv_to_json(self, csv_file_path, output_json_path, constraints, weights, include_detailed_metrics=False):
        """
        Reads a CSV file, processes data by 'source_file', calculates
        the final reward using a SINGLE constraint set for all, outputs JSON, 
        and saves it to a specified file.
        
        Returns:
            tuple: (json_output_string, path_of_saved_file)
        """
        
        # --- NEW STRUCTURE: Set up the global dictionary format ---
        final_output = {
            "active_constraints": constraints,
            "circuits": {}
        }
        
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            return json.dumps({"Error": {"message": "File not found"}}, indent=4), None
        except Exception as e:
            return json.dumps({"Error": {"message": f"Could not read CSV: {e}"}}, indent=4), None

        if 'source_file' not in df.columns:
            return json.dumps({"Error": {"message": "Missing required 'source_file' column"}}, indent=4), None

        # Define how each column should be aggregated across the voltage sweep
        aggregation_rules = {
            'total_volume_cm3': 'max',            
            'voltage_out_ripple_V': 'max',        
            'switch_voltage_peak_V': 'max',       
            'switch_current_peak_A': 'max',
            'inductor_current_peak_A': 'max',
            'efficiency': 'mean',                 
            'voltage_out_mean_V': 'mean',         
            'count_mosfets': 'first',
            'count_diodes': 'first',
            'count_inductors': 'first',
            'count_capacitors': 'first'
        }

        # Filter out rules for columns that don't exist in the CSV
        valid_rules = {col: rule for col, rule in aggregation_rules.items() if col in df.columns}

        # Group by the circuit and apply ONLY the valid specific rules
        grouped = df.groupby('source_file').agg(valid_rules)

        # Calculate the reward using this properly aggregated data
        for source_file_name, row in grouped.iterrows():
            
            # Use the single global constraint dictionary passed into the function
            final_reward, details = self.calculate_reward(row, constraints, weights)
            
            circuit_data = {
                "fitness_score": float(final_reward)
            }
            
            # Inject the extra variables if the hyperparameter is toggled
            if include_detailed_metrics:
                circuit_data.update(details)

            # --- Inject into the new nested 'circuits' branch ---
            final_output["circuits"][str(source_file_name)] = circuit_data

        # Dump the entire nested dictionary to JSON
        json_string = json.dumps(final_output, indent=4)

        # Write the JSON string to the specified file path
        try:
            with open(output_json_path, 'w') as json_file:
                json_file.write(json_string)
        except Exception as e:
            return json.dumps({"Error": {"message": f"Could not save JSON file: {e}"}}, indent=4), None

        return json_string, output_json_path


# --- Example Execution Setup ---

if __name__ == "__main__":
    
    # A single constraint dictionary applied to EVERY topology in the batch
    my_constraints = {
        "vin_min": 12, 
        "vin_max": 12, 
        "vout_target": 5.0, 
        "efficiency_target": 0.90, 
        "power_in": 100
    }

    my_weights = {
        'v_out': 10.0,          
        'efficiency': 20.0,     
        'volume': 2.0,          
        'component_cost': 1.0,            
        'components': {         
            'mosfet': 1.0,
            'diode': 1.0,
            'inductor': 1.0,
            'capacitor': 1.0
        }
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'batch_001_out.csv')
    
    # Specify where you want the JSON saved
    output_json_path = os.path.join(script_dir, 'batch_001_results.json')

    reward_function = RewardFunction()
    
    # Unpack the returned tuple
    json_output, saved_file_path = reward_function.process_csv_to_json(
        csv_file_path, 
        output_json_path, 
        my_constraints, 
        my_weights, 
        include_detailed_metrics=False 
    )

    print("--- JSON DATA ---")
    print(json_output)
    print("\n--- FILE STATUS ---")
    if saved_file_path:
        print(f"Successfully saved to: {saved_file_path}")
    else:
        print("Failed to save JSON file.")
