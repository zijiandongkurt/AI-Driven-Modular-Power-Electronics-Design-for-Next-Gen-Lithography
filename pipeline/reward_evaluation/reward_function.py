import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

class RewardFunction:
    def __init__(self):
        # Define a maximum penalty cap to prevent network gradients from exploding
        self.MAX_PENALTY = 10000.0 

    def calculate_loss(self, row, constraints, weights):
        """Calculate the total loss as a weighted sum of penalties.

        Args:
            row (dict | pd.Series): Simulation metrics for one circuit.
            constraints (dict): Target values (e.g. ``vout_target``,
                ``efficiency_target``).
            weights (dict): Per-term loss weights including a nested
                ``"components"`` sub-dict.

        Returns:
            tuple: ``(total_loss, details_dict)`` where ``total_loss`` is a
                float and ``details_dict`` contains ``"loss_breakdown"`` and
                ``"raw_metrics"`` sub-dicts.
        """
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

        target_v_out = constraints.get('vout_target', 5.0)
        penalty_v_out = (v_out - target_v_out) ** 2
        
        target_efficiency = constraints.get('efficiency_target', 0.90)
        safe_efficiency = max(0.0, min(1.0, float(efficiency)))
        penalty_efficiency = max(0.0, target_efficiency - safe_efficiency)

        if np.isinf(total_volume):
            penalty_volume = self.MAX_PENALTY
        else:
            penalty_volume = min(float(total_volume), self.MAX_PENALTY)

        comp_weights = weights.get('components', {})
        penalty_components = (
            comp_weights.get('mosfet', 1.0) * count_mosfets +
            comp_weights.get('diode', 1.0) * count_diodes +
            comp_weights.get('inductor', 1.0) * count_inductors +
            comp_weights.get('capacitor', 1.0) * count_capacitors
        )

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
        """Read a CSV, calculate rewards for all circuits, and write JSON.

        Processes data grouped by ``source_file``, applying a single constraint
        set to all circuits.

        Args:
            csv_file_path: Path to simulation_results.csv.
            output_json_path: Destination path for reward_results.json.
            constraints (dict): Target values applied to every circuit.
            weights (dict): Loss weight dict.
            include_detailed_metrics (bool): Whether to include
                ``"loss_breakdown"`` and ``"raw_metrics"`` in output.

        Returns:
            tuple: ``(json_output_string, path_of_saved_file)`` where
                ``path_of_saved_file`` is None on error.
        """
        
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


    def process_batch(self, batchID, constraints, weights, include_detailed_metrics=True):
        """Standard pipeline entry point. Reads and writes from data/<batchID>/."""
        data_dir         = Path(__file__).parent.parent / "data"
        csv_file_path    = data_dir / batchID / "simulation_results.csv"
        output_json_path = data_dir / batchID / "reward_results.json"

        return self.process_csv_to_json(
            csv_file_path    = csv_file_path,
            output_json_path = output_json_path,
            constraints      = constraints,
            weights          = weights,
            include_detailed_metrics=include_detailed_metrics,
        )


# if __name__ == "__main__":
    
#     # A single constraint dictionary applied to EVERY topology in the batch
#     my_constraints = {
#         "vin_min": 12,
#         "vin_max": 100,
#         "vout_target": 5,
#         "efficiency_target": 0.90,
#         "power_in": 100,
#     }


#     my_weights = {
#         'v_out': 10.0,          
#         'efficiency': 20.0,     
#         'volume': 2.0,          
#         'component_cost': 1.0,            
#         'components': {         
#             'mosfet': 1.0,
#             'diode': 1.0,
#             'inductor': 1.0,
#             'capacitor': 1.0
#         }
#     }

#     data_dir         = Path(__file__).parent.parent / "data"
#     csv_file_path    = data_dir / "batch_2" / "simulation_results.csv"
#     output_json_path = data_dir / "batch_2" / "reward_results.json"

#     reward_function = RewardFunction()
    
#     # Unpack the returned tuple
#     json_output, saved_file_path = reward_function.process_csv_to_json(
#         csv_file_path, 
#         output_json_path, 
#         my_constraints, 
#         my_weights, 
#         include_detailed_metrics=True
#     )

#     print("--- JSON DATA ---")
#     print(json_output)
#     print("\n--- FILE STATUS ---")
#     if saved_file_path:
#         print(f"Successfully saved to: {saved_file_path}")
#     else:
#         print("Failed to save JSON file.")