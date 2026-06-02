import pandas as pd
import numpy as np
import json
from pathlib import Path


# ── Per-component loss caps ───────────────────────────────────────────────
# These define the "worst realistic case" for each loss term.
# Anything above the cap is treated as equally bad — prevents one runaway
# term (e.g. volume=20000) from drowning the gradient signal from others.

LOSS_CAP = {
    "voltage_tracking_loss": 100.0,   # (5V - 0V)^2 * weight=10 = 250 worst case; cap lower
    "efficiency_loss":        20.0,   # max penalty_efficiency=1.0 * weight=20
    "volume_loss":           100.0,   # cap raw volume contribution
    "component_cost_loss":    10.0,   # max ~10 components * weight=1
}

# Validation check weights for partial reward on invalid topologies.
# Weights sum to 1.0 — represents fraction of "structural correctness".
VALIDATION_CHECK_WEIGHTS = {
    # Critical — netlist is meaningless without these
    "netlist_syntax":        0.20,
    "ground_reference":      0.10,
    "power_supply":          0.10,
    "naming_conventions":    0.10,
    "connected_graph":       0.10,
    # Important structural checks
    "model_declarations":    0.08,
    "simulation_parameters": 0.08,
    "gate_drive_present":    0.06,
    "gate_per_mosfet":       0.06,
    "floating_nodes":        0.04,
    "current_path":          0.04,
    # Safety/sanity checks — low weight, mostly expected to pass
    "invalid_components":    0.01,
    "duplicate_refs":        0.01,
    "component_values":      0.01,
    "mosfet_bulk":           0.01,
    # Informational only — no gradient signal
    "cl_loops":              0.00,
    "short_same_node":       0.00,
    "short_zero_resistor":   0.00,
    "short_zero_voltage":    0.00,
    "singular_voltage_cap":  0.00,
    "short_input_to_gnd":    0.00,
    "kvl_voltage_loop":      0.00,
    "singular_cap_loop":     0.00,
}


class RewardFunctionNorm:
    def __init__(self):
        self.MAX_PENALTY = 10000.0
        LOSS_CAP["volume_loss"] = self.MAX_PENALTY

    # ── Validation reward (invalid topologies) ───────────────────────────

    def validation_reward(self, checks: dict) -> float:
        """Convert a validation checks dict into a scalar reward in [-1.0, -0.5].

        A completely invalid netlist (all checks failed) returns -1.0.
        A netlist that passed all structural checks but failed simulation
        directives returns close to -0.5.

        This range is intentionally separated from the simulation reward range [0.5, 1.0]
        to create a strict distinction (a gap from -0.5 to 0.5) so that any 
        valid/simulated topology vastly outranks an invalid one.

        Args:
            checks: dict of check_name -> bool from validation_results.json

        Returns:
            float in [-1.0, -0.5]
        """
        score = 0.0
        for check, weight in VALIDATION_CHECK_WEIGHTS.items():
            if checks.get(check, False):
                score += weight
        # score in [0, 1] → remap to [-1.0, -0.5]
        return -1.0 + (score * 0.5)

    # ── Simulation reward (valid topologies) ─────────────────────────────

    def calculate_loss(self, row, constraints, weights):
        """Compute weighted loss from simulation metrics.

        Each loss term is capped and normalized to [0, 1] before weighting,
        so no single term can dominate the gradient signal.

        Returns:
            tuple: (total_loss, details_dict)
                   total_loss is in [0, 1] after normalization.
        """
        # --- Extract metrics safely ---
        v_out = row.get('voltage_out_mean_V', 0.0)
        v_out = 0.0 if pd.isna(v_out) else float(v_out)

        efficiency = row.get('efficiency', 0.0)
        efficiency = 0.0 if pd.isna(efficiency) else float(efficiency)

        total_volume = row.get('total_volume_cm3', self.MAX_PENALTY)
        total_volume = self.MAX_PENALTY if pd.isna(total_volume) else float(total_volume)

        count_mosfets    = int(row.get('count_mosfets', 0))
        count_diodes     = int(row.get('count_diodes', 0))
        count_inductors  = int(row.get('count_inductors', 0))
        count_capacitors = int(row.get('count_capacitors', 0))

        # --- Raw penalty terms ---
        target_v_out = constraints.get('vout_target', 5.0)
        penalty_v_out = (v_out - target_v_out) ** 2

        target_efficiency = constraints.get('efficiency_target', 0.90)
        safe_efficiency = max(0.0, min(1.0, efficiency))
        penalty_efficiency = max(0.0, target_efficiency - safe_efficiency)

        if np.isinf(total_volume):
            penalty_volume = self.MAX_PENALTY
        else:
            penalty_volume = min(total_volume, self.MAX_PENALTY)

        comp_weights = weights.get('components', {})
        penalty_components = (
            comp_weights.get('mosfet', 1.0)    * count_mosfets +
            comp_weights.get('diode', 1.0)     * count_diodes +
            comp_weights.get('inductor', 1.0)  * count_inductors +
            comp_weights.get('capacitor', 1.0) * count_capacitors
        )

        # --- Normalize each term to [0, 1] using per-component caps ---
        norm_v_out      = min(penalty_v_out      / LOSS_CAP["voltage_tracking_loss"], 1.0)
        norm_efficiency = min(penalty_efficiency / LOSS_CAP["efficiency_loss"],       1.0)
        norm_volume     = min(penalty_volume     / LOSS_CAP["volume_loss"],           1.0)
        norm_components = min(penalty_components / LOSS_CAP["component_cost_loss"],   1.0)

        # --- Apply top-level weights ---
        loss_v_out      = weights.get('v_out', 1.0)           * norm_v_out
        loss_efficiency = weights.get('efficiency', 1.0)      * norm_efficiency
        loss_volume     = weights.get('volume', 1.0)          * norm_volume
        loss_components = weights.get('component_cost', 1.0)  * norm_components
        
        # --- calculate total weight ---
        total_w = (
            weights.get('v_out', 1.0) +
            weights.get('efficiency', 1.0) +
            weights.get('volume', 1.0) +
            weights.get('component_cost', 1.0)
        )
        # sum up normalized losses and divide by total weight to keep in [0, 1]
        total_loss = (loss_v_out + loss_efficiency + loss_volume + loss_components) / max(total_w, 1e-8)

        # Clamp final loss to [0, 1]
        total_loss = float(np.clip(total_loss, 0.0, 1.0))

        details = {
            "loss_breakdown": {
                "voltage_tracking_loss": float(loss_v_out),
                "efficiency_loss":       float(loss_efficiency),
                "volume_loss":           float(loss_volume),
                "component_cost_loss":   float(loss_components),
            },
            "raw_metrics": {
                "simulation_output_voltage": v_out,
                "efficiency":                safe_efficiency,
                "total_volume_cm3":          float(penalty_volume),
                "total_components":          count_mosfets + count_diodes + count_inductors + count_capacitors,
            },
        }

        if np.isinf(total_loss) or np.isnan(total_loss):
            return 1.0, details  # worst valid reward

        return total_loss, details

    def calculate_reward(self, row, constraints, weights):
        """Convert loss into a reward in [0.5, 1.0]."""
        loss, details = self.calculate_loss(row, constraints, weights)
        
        # Base reward mapping loss [0, 1] -> reward [1.0, 0.5]
        reward = 1.0 - (0.5 * loss)
        
        # --- NEW: CONTINUOUS FATAL VOLTAGE PENALTY ---
        v_out = details["raw_metrics"]["simulation_output_voltage"]
        target = constraints.get('vout_target', 5.0)
        
        v_error = abs(v_out - target)
        safe_margin = target * 0.50  # Start penalizing after 50% error (e.g., outside 4V-6V)
        
        if v_error > safe_margin:
            # Calculate how far past the safe margin we are
            excess_error = v_error - safe_margin
            
            # Exponential decay: The worse the voltage, the closer this gets to 0.0
            # A factor of 0.15 means the reward drops sharply, but maintains a smooth curve.
            decay_factor = np.exp(-0.15 * excess_error)
            
            # Squeeze the available reward space down toward the 0.50 floor
            reward = 0.50 + ((reward - 0.50) * decay_factor)
            
            details["fatal_penalty_applied"] = True
            details["voltage_decay_factor"] = float(decay_factor)

        return float(reward), details

    # ── Batch normalization for GRPO ─────────────────────────────────────

    def normalize_group_rewards(self, rewards: dict) -> dict:
        """Normalize a group of rewards for GRPO using z-score, clamped to [-1, 1].

        Call this on the full batch rewards dict before passing to the trainer.
        Handles the NaN/zero-std edge case when all candidates score identically.

        Args:
            rewards: {candidate_name: float} raw rewards

        Returns:
            {candidate_name: float} normalized rewards in [-1, 1]
        """
        names  = list(rewards.keys())
        values = np.array([rewards[n] for n in names], dtype=np.float32)

        mean = values.mean()
        std  = max(values.std(), 1e-8)
        normalized = np.clip((values - mean) / std, -4.0, 4.0)

        return {name: float(normalized[i]) for i, name in enumerate(names)}

    # ── CSV → JSON pipeline step ──────────────────────────────────────────

    def process_csv_to_json(
        self,
        csv_file_path,
        output_json_path,
        constraints,
        weights,
        include_detailed_metrics=False,
        validation_json_path=None,
    ):
        """Read simulation CSV, compute rewards, merge invalid candidates, write JSON.

        Invalid candidates (those that failed validation and were never simulated)
        are assigned a validation-based reward in [-1.0, -0.5] so GRPO still
        receives a gradient signal for the full group.

        Args:
            csv_file_path:         Path to simulation_results.csv
            output_json_path:      Path to write reward_results.json
            constraints:           Constraint dict
            weights:               Loss weight dict
            include_detailed_metrics: Whether to include loss breakdown in output
            validation_json_path:  Optional path to validation_results.json.
                                   When provided, failed candidates are included
                                   with their validation-based reward.

        Returns:
            tuple: (json_string, output_path)
        """
        final_output = {
            "active_constraints": constraints,
            "circuits": {}
        }

        # --- Load validation results if provided ---
        validation_results = {}
        if validation_json_path is not None:
            try:
                with open(validation_json_path, "r") as f:
                    validation_results = json.load(f)
            except Exception as e:
                print(f"Warning: could not load validation results: {e}")

        # --- Add invalid candidates first (they were never simulated) ---
        for cand_name, val_result in validation_results.items():
            if not val_result.get("passed", True):
                reward = self.validation_reward(val_result["checks"])
                final_output["circuits"][cand_name] = {
                    "fitness_score": reward,
                    "source": "validation_penalty",
                }

        # --- Load and process simulation CSV ---
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            # No simulation results at all — only validation penalties
            if not final_output["circuits"]:
                return json.dumps({"Error": {"message": "No CSV and no validation results"}}, indent=4), None
            json_string = json.dumps(final_output, indent=4)
            with open(output_json_path, "w") as f:
                f.write(json_string)
            return json_string, output_json_path
        except Exception as e:
            return json.dumps({"Error": {"message": f"Could not read CSV: {e}"}}, indent=4), None

        if "source_file" not in df.columns:
            return json.dumps({"Error": {"message": "Missing 'source_file' column"}}, indent=4), None

        aggregation_rules = {
            "total_volume_cm3":          "max",
            "voltage_out_ripple_V":      "max",
            "switch_voltage_peak_V":     "max",
            "switch_current_peak_A":     "max",
            "inductor_current_peak_A":   "max",
            "efficiency":                "mean",
            "voltage_out_mean_V":        "mean",
            "count_mosfets":             "first",
            "count_diodes":              "first",
            "count_inductors":           "first",
            "count_capacitors":          "first",
        }
        valid_rules = {col: rule for col, rule in aggregation_rules.items() if col in df.columns}
        grouped = df.groupby("source_file").agg(valid_rules)

        for source_file_name, row in grouped.iterrows():
            reward, details = self.calculate_reward(row, constraints, weights)
            circuit_data = {"fitness_score": float(reward), "source": "simulation"}
            if include_detailed_metrics:
                circuit_data.update(details)
            final_output["circuits"][str(source_file_name)] = circuit_data

        # --- Normalize all rewards across the group for GRPO ---
        raw_rewards = {
            name: data["fitness_score"]
            for name, data in final_output["circuits"].items()
        }
        normalized = self.normalize_group_rewards(raw_rewards)
        for name, norm_reward in normalized.items():
            final_output["circuits"][name]["grpo_reward"] = norm_reward

        # --- Write output ---
        json_string = json.dumps(final_output, indent=4)
        try:
            with open(output_json_path, "w") as f:
                f.write(json_string)
        except Exception as e:
            return json.dumps({"Error": {"message": f"Could not save JSON: {e}"}}, indent=4), None

        return json_string, output_json_path

    # ── Pipeline entry point ──────────────────────────────────────────────

    def process_batch(self, batchID, constraints, weights, include_detailed_metrics=True):
        """Standard pipeline entry point. Reads and writes from data/<batchID>/."""
        data_dir              = Path(__file__).parent.parent / "data"
        csv_file_path         = data_dir / batchID / "simulation_results.csv"
        output_json_path      = data_dir / batchID / "reward_results.json"
        validation_json_path  = data_dir / batchID / "validation_results.json"

        return self.process_csv_to_json(
            csv_file_path          = csv_file_path,
            output_json_path       = output_json_path,
            constraints            = constraints,
            weights                = weights,
            include_detailed_metrics = include_detailed_metrics,
            validation_json_path   = validation_json_path,
        )