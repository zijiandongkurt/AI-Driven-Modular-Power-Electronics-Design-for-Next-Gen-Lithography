import os
import sys
import json
import time
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.netlist_validation.validator import validator
from pipeline.simulation.local.ltspice_runner import LTSpiceSimulator
from pipeline.simulation.local.raw_extractor import RawExtractor
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm

# Hardcoded weights from your benchmark_config.json
WEIGHTS = {
    "v_out": 20.0,
    "efficiency": 10.0,
    "volume": 2.0,
    "component_cost": 0.05,
    "components": {
        "mosfet": 1.0,
        "diode": 1.0,
        "inductor": 1.0,
        "capacitor": 1.0
    }
}

def get_all_batches(data_dir: Path):
    """Handles both Depth 1 (Bench) and Depth 2 (Zycos) structures."""
    batches = []
    top_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    for top_dir in top_folders:
        if top_dir.name.startswith("Bench_"):
            # Depth 1 Structure: Bench_XXX / batch_1
            batch_folders = sorted([d for d in top_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                                   key=lambda x: int(x.name.split('_')[1]))
            for b in batch_folders:
                batches.append((top_dir, b))
                
        elif top_dir.name.startswith("zycos_"):
            # Depth 2 Structure: zycos_XXX / Run_001 / batch_1
            run_folders = sorted([d for d in top_dir.iterdir() if d.is_dir() and d.name.startswith("Run_")])
            for run_dir in run_folders:
                batch_folders = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")], 
                                       key=lambda x: int(x.name.split('_')[1]))
                for b in batch_folders:
                    batches.append((run_dir, b)) 
                    
    return batches

def format_eta(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def main():
    data_dir = PROJECT_ROOT / "pipeline" / "data"
    
    if not data_dir.exists():
        print(f"❌ Error: Could not find new data directory at {data_dir}")
        return

    all_batches = get_all_batches(data_dir)
    total_batches = len(all_batches)
    
    if total_batches == 0:
        print("❌ No batches found to resimulate.")
        return

    print(f"\n{'='*70}")
    print(f" 🚀 STARTING MASS RESIMULATION")
    print(f" Total Batches to Process: {total_batches}")
    print(f" Target Engine: Local LTSpice")
    print(f"{'='*70}\n")

    val = validator()
    reward_fn = RewardFunctionNorm()
    
    rolling_times = deque(maxlen=10)
    start_time = time.time()

    for idx, (run_dir, batch_dir) in enumerate(all_batches, 1):
        batch_start_time = time.time()
        
        # Batch ID needs to include the zycos folder if applicable for the validator path
        if "zycos_" in str(run_dir):
            zycos_parent = run_dir.parent.name
            batch_id = f"{zycos_parent}/{run_dir.name}/{batch_dir.name}"
        else:
            batch_id = f"{run_dir.name}/{batch_dir.name}"
        
        constraint_file = run_dir / "active_constraint.json"
        if not constraint_file.exists():
            print(f" ⚠️ Skipping {batch_id} - No active_constraint.json found.")
            continue
            
        with open(constraint_file, "r", encoding="utf-8") as f:
            constraint_data = json.load(f)
        active_constraints = constraint_data.get("active_constraints", {})

        print(f"\n--- [Batch {idx}/{total_batches}] {batch_id} ---")
        
        try:
            print(" 1. Validating...")
            validation_results = val.validate(batch_id)
            
            llm_output_dir = batch_dir / "LLM_output"
            valid_net_paths = [
                llm_output_dir / net_name 
                for net_name, data in validation_results.items() 
                if data[0] == True 
            ]
            
            csv_path = batch_dir / "simulation_results.csv"
            
            if valid_net_paths:
                print(f" 2. Simulating {len(valid_net_paths)} valid netlists...")
                simulator = LTSpiceSimulator(output_dir=batch_dir)
                netlist_map = simulator.simulate(valid_net_paths)
                
                print(" 3. Extracting Raw Data...")
                extractor = RawExtractor(output_dir=batch_dir)
                extractor.extract(netlist_map, results_path=csv_path)
            else:
                print(" 2. Simulating... (Skipped: 0 valid netlists)")
                
            print(" 4. Calculating Rewards...")
            reward_fn.process_batch(
                batchID=batch_id, 
                constraints=active_constraints, 
                weights=WEIGHTS
            )

        except Exception as e:
            print(f" ❌ Error processing {batch_id}: {e}")

        batch_duration = time.time() - batch_start_time
        rolling_times.append(batch_duration)
        
        avg_time_per_batch = sum(rolling_times) / len(rolling_times)
        batches_remaining = total_batches - idx
        eta_seconds = batches_remaining * avg_time_per_batch
        
        projected_end_time = datetime.now() + timedelta(seconds=eta_seconds)
        
        print(f" ✅ Batch Complete in {batch_duration:.1f}s")
        print(f" ⏱️ Rolling Avg: {avg_time_per_batch:.1f}s/batch | ETA: {format_eta(eta_seconds)}")
        print(f" 🏁 Projected End Time: {projected_end_time.strftime('%H:%M:%S')}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" 🎉 MASS RESIMULATION COMPLETE!")
    print(f" Processed {total_batches} batches in {format_eta(total_time)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()