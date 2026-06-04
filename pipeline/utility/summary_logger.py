import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict
from pipeline.utility.netlist_database import NetlistDatabase

class SummaryLogger:
    """
    Handles logging and formatting of the MCTS/RL Loop execution summary.
    Generates run_summary.txt and outputs the current best_topology.net to disk.
    """
    def __init__(self, run_folder_path: Path, n_batches: int, sim_params: dict, weights: dict, constraint_idx: int = None):
        self.run_folder_path = Path(run_folder_path)
        self.results_dir = self.run_folder_path / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_batches = n_batches
        self.sim_params = sim_params
        self.weights = weights
        self.constraint_idx = constraint_idx
        
        self.best_fitness_history = []
        self.batch_times = []

    def log_batch(self, batch_idx: int, batch_duration: float, db: NetlistDatabase, best_cand_id: str):
        self.batch_times.append(batch_duration)
        avg_time_batch = sum(self.batch_times) / len(self.batch_times)
        
        total_generated = db.total_valid + db.total_invalid
        avg_time_netlist = sum(self.batch_times) / total_generated if total_generated > 0 else 0
        valid_pct = (db.total_valid / total_generated * 100) if total_generated > 0 else 0
        
        best_record = db.records[best_cand_id]
        best_fitness = best_record["fitness"]
        self.best_fitness_history.append((best_cand_id, best_fitness))
        
        # 1. Overwrite the Best Netlist File
        best_netlist_path = self.results_dir / "best_topology.net"
        with open(best_netlist_path, "w", encoding="utf-8") as f:
            f.write(f"* OVERALL BEST CANDIDATE: {best_cand_id}\n")
            f.write(f"* FITNESS SCORE: {best_fitness:.4f}\n*\n")
            f.write(best_record["netlist_text"])
            
        # 2. Overwrite the Summary Log
        summary_path = self.results_dir / "run_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== MCTS INFERENCE SUMMARY: {self.run_folder_path.name} ===\n")
            f.write(f"Batches Completed: {batch_idx} / {self.n_batches}\n")
            f.write(f"Total Time Elapsed: {sum(self.batch_times):.2f} sec ({sum(self.batch_times)/60:.2f} min)\n")
            f.write(f"Average Time per Batch: {avg_time_batch:.2f} sec\n")
            f.write(f"Average Time per Netlist: {avg_time_netlist:.2f} sec\n")
            
            n_generated_batch = self.sim_params.get("sampled_states", 0) * self.sim_params.get("cands_per_prompt", 0)
            f.write(f"Netlists Generated per Batch: {n_generated_batch}\n\n")
            
            f.write("--- SIMULATION PARAMETERS ---\n")
            f.write(f"Sampled States per Batch: {self.sim_params.get('sampled_states')}\n")
            f.write(f"Candidates per Prompt: {self.sim_params.get('cands_per_prompt')}\n")
            f.write(f"MCTS Temperature: {self.sim_params.get('mcts_temp')}\n")
            f.write(f"LLM Max Tokens: {self.sim_params.get('max_tokens')}\n\n")
            
            f.write("--- TOPOLOGY YIELD ---\n")
            f.write(f"Total Candidates Generated: {total_generated}\n")
            f.write(f"Valid Topologies: {db.total_valid}\n")
            f.write(f"Invalid Topologies: {db.total_invalid}\n")
            f.write(f"Validity Rate: {valid_pct:.1f}%\n\n")

            unique_pct = (db.total_unique / total_generated * 100) if total_generated > 0 else 0
            f.write(f"Unique Topologies (Globally): {db.total_unique}\n")
            f.write(f"Exact Duplicates (Globally):  {db.total_duplicates}\n")
            f.write(f"Global Uniqueness Rate: {unique_pct:.1f}%\n\n")
            
            f.write("--- FITNESS PROGRESSION ---\n")
            f.write(f"Overall Best Fitness: {best_fitness:.4f} ({best_cand_id})\n")
            for b_idx, (b_id, b_fit) in enumerate(self.best_fitness_history, 1):
                f.write(f"  Batch {b_idx} Best DB Score: {b_fit:.4f} ({b_id})\n")
                
            f.write("\n=========================================\n")
            
            # Extract Metrics Before Netlist
            raw_metrics = best_record.get("metrics", {}).get("raw_metrics", {})
            f.write("--- BEST NETLIST METRICS ---\n")
            if raw_metrics:
                f.write(f"Output Voltage: {raw_metrics.get('simulation_output_voltage', 0):.4f} V\n")
                f.write(f"Efficiency:     {raw_metrics.get('efficiency', 0):.4f}\n")
                f.write(f"Total Volume:   {raw_metrics.get('total_volume_cm3', 0):.4f} cm^3\n")
            else:
                f.write("No raw metrics available. (Topology likely failed validation).\n")
            
            f.write("\n--- COMPONENT BREAKDOWN ---\n")
            
            counts = {"MOSFETs (M)": 0, "Diodes (D)": 0, "Inductors (L)": 0, "Capacitors (C)": 0, "Resistors (R)": 0, "Voltage Sources (V)": 0}
            
            # 1. Look up M, D, L, C components securely via CSV mapping
            match = re.search(r'_b(\d+)_', best_cand_id)
            found_csv = False
            
            if match:
                cand_batch = f"batch_{match.group(1)}"
                csv_path = self.run_folder_path / cand_batch / "simulation_results.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        row = df[df['source_file'] == best_cand_id]
                        if not row.empty:
                            counts["MOSFETs (M)"] = int(row['count_mosfets'].iloc[0])
                            counts["Diodes (D)"] = int(row['count_diodes'].iloc[0])
                            counts["Inductors (L)"] = int(row['count_inductors'].iloc[0])
                            counts["Capacitors (C)"] = int(row['count_capacitors'].iloc[0])
                            found_csv = True
                    except Exception as e:
                        print(f"Error reading CSV for component counts: {e}")
            
            # 2. Extract R and V logic from text since they are missing from CSV, fallback heavily if CSV not found 
            for line in best_record["netlist_text"].split("\n"):
                line = line.strip().upper()
                if line.startswith("R"):
                    counts["Resistors (R)"] += 1
                elif line.startswith("V"):
                    counts["Voltage Sources (V)"] += 1
                # Standard fallback mechanism
                if not found_csv:
                    if line.startswith("M"):
                        counts["MOSFETs (M)"] += 1
                    elif line.startswith("D"):
                        counts["Diodes (D)"] += 1
                    elif line.startswith("L"):
                        counts["Inductors (L)"] += 1
                    elif line.startswith("C"):
                        counts["Capacitors (C)"] += 1
            
            total_components = sum(counts.values())
            for comp, count in counts.items():
                f.write(f"{comp}: {count}\n")
                
            f.write(f"----------------------\n")
            f.write(f"Total Components: {total_components}\n")
            
            # Print Netlist Last
            f.write("\nOVERALL BEST SPICE NETLIST:\n")
            f.write("=========================================\n")
            f.write(best_record["netlist_text"])
            f.write("\n\n")

    def log_batch_training(
        self,
        batch_idx: int,
        batch_duration: float,
        history: List[Dict],
        batch_id: str,
    ) -> None:
        """Training-loop variant of log_batch.

        Works directly from the running history list and the per-batch
        reward/validation JSON files instead of requiring a NetlistDatabase.
        Overwrites results/run_summary.txt and results/best_topology.net after
        every batch so progress is visible mid-run.
        """
        if not history:
            return

        self.batch_times.append(batch_duration)
        avg_time_batch = sum(self.batch_times) / len(self.batch_times)
        avg_time_netlist = sum(self.batch_times) / max(len(history), 1)

        # Best candidate across all history so far
        best_entry = max(history, key=lambda x: x["fitness"])
        best_cand_id = best_entry["netlist_id"]
        best_fitness = best_entry["fitness"]
        self.best_fitness_history.append((best_cand_id, best_fitness))

        data_root = Path("pipeline") / "data"

        # Valid / invalid count for the current batch
        val_path = data_root / batch_id / "validation_results.json"
        # Valid / invalid count for the ENTIRE run (calculated from history)
        n_valid = sum(1 for cand in history if cand.get("is_valid", False))
        n_invalid = len(history) - n_valid
        if val_path.exists():
            with val_path.open(encoding="utf-8") as f:
                val_data = json.load(f)
            for v in val_data.values():
                if v.get("passed", False):
                    n_valid += 1
                else:
                    n_invalid += 1

        # Raw metrics and netlist text for the global best candidate
        best_batch_dir = data_root / best_entry.get("batch_id", batch_id)
        reward_path = best_batch_dir / "reward_results.json"
        raw_metrics: Dict = {}
        if reward_path.exists():
            with reward_path.open(encoding="utf-8") as f:
                rdata = json.load(f)
            raw_metrics = rdata.get("circuits", {}).get(best_cand_id, {}).get("raw_metrics", {})

        net_file = best_batch_dir / "LLM_output" / f"{best_cand_id}.net"
        netlist_text = net_file.read_text(encoding="utf-8") if net_file.exists() else ""

        # Overwrite best_topology.net
        best_netlist_path = self.results_dir / "best_topology.net"
        with best_netlist_path.open("w", encoding="utf-8") as f:
            f.write(f"* OVERALL BEST CANDIDATE: {best_cand_id}\n")
            f.write(f"* FITNESS SCORE: {best_fitness:.4f}\n*\n")
            f.write(netlist_text)

        # Overwrite run_summary.txt
        summary_path = self.results_dir / "run_summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(f"=== TRAINING SUMMARY: {self.run_folder_path.name} ===\n")
            if self.constraint_idx is not None:
                f.write(f"Constraint Index: {self.constraint_idx}\n")
            f.write(f"Batches Completed: {batch_idx} / {self.n_batches}\n")
            total_sec = sum(self.batch_times)
            f.write(f"Total Time Elapsed: {total_sec:.2f} sec ({total_sec / 60:.2f} min)\n")
            f.write(f"Average Time per Batch: {avg_time_batch:.2f} sec\n")
            f.write(f"Average Time per Netlist: {avg_time_netlist:.2f} sec\n\n")

            f.write("--- TOPOLOGY YIELD (entire run) ---\n")
            f.write(f"Valid:   {n_valid}\n")
            f.write(f"Invalid: {n_invalid}\n")
            f.write(f"Total candidates in history: {len(history)}\n")
            
            unique_hashes = set(cand.get("topo_hash", cand["netlist_id"]) for cand in history)
            n_unique = len(unique_hashes)
            n_total = len(history)
            unique_pct = (n_unique / n_total * 100) if n_total > 0 else 0
            
            f.write(f"Unique Topologies: {n_unique}\n")
            f.write(f"Exact Duplicates:  {n_total - n_unique}\n")
            f.write(f"Uniqueness Rate:   {unique_pct:.1f}%\n\n")

            f.write("--- FITNESS PROGRESSION ---\n")
            f.write(f"Overall Best Fitness: {best_fitness:.4f} ({best_cand_id})\n")
            for b_idx, (b_id, b_fit) in enumerate(self.best_fitness_history, 1):
                f.write(f"  Batch {b_idx} Best: {b_fit:.4f} ({b_id})\n")

            f.write("\n--- BEST NETLIST METRICS ---\n")
            if raw_metrics:
                f.write(f"Output Voltage: {raw_metrics.get('simulation_output_voltage', 0):.4f} V\n")
                f.write(f"Efficiency:     {raw_metrics.get('efficiency', 0):.4f}\n")
                f.write(f"Total Volume:   {raw_metrics.get('total_volume_cm3', 0):.4f} cm^3\n")
            else:
                f.write("No raw metrics available.\n")

            if netlist_text:
                f.write("\n=========================================\n")
                f.write("OVERALL BEST SPICE NETLIST:\n")
                f.write("=========================================\n")
                f.write(netlist_text)
                f.write("\n\n")