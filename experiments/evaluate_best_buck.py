import sys
import os
import json
from pathlib import Path

# ==============================================================================
# 1. SETUP PIPELINE IMPORTS
# ==============================================================================
# This script is located in /experiments, so we append the parent directory (root)
# to sys.path to allow importing from the /pipeline folder.
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

# Import the classes from your pipeline
from pipeline.simulation.local.ltspice_runner import LTSpiceSimulator
from pipeline.simulation.local.raw_extractor import RawExtractor
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm

# ==============================================================================
# 2. THE BEST NETLIST
# ==============================================================================
BEST_NETLIST = """* 10V to 5V Buck Converter (Corrected)
Vin in 0 10
M1 in gate sw sw NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out 100u
C1 out 0 470u
Rload out 0 0.5 
Vgate gate sw PULSE(0 12 0 1n 1n 5u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 10n 1m
.save V(*) I(*)
.end
"""

def main():
    # ==========================================================================
    # 3. DIRECTORY SETUP
    # ==========================================================================
    results_dir = CURRENT_DIR / "simulation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    netlist_path = results_dir / "best_buck.net"
    csv_path = results_dir / "simulation_metrics.csv"
    json_path = results_dir / "reward_metrics.json"

    print(f"📁 Setup complete. Results will be saved to: {results_dir}")

    # Write the netlist to disk
    with open(netlist_path, "w") as f:
        f.write(BEST_NETLIST)

    # ==========================================================================
    # 4. RUN SIMULATION
    # ==========================================================================
    print("\n🚀 Step 1: Running LTSpice Simulation...")
    simulator = LTSpiceSimulator(output_dir=results_dir)
    # The simulate method returns the netlist_map needed for the extractor
    netlist_map = simulator.simulate([netlist_path])

    # ==========================================================================
    # 5. EXTRACT METRICS
    # ==========================================================================
    print("\n📊 Step 2: Extracting metrics from .raw files...")
    extractor = RawExtractor(output_dir=results_dir)
    # The extract method reads the .raw files, creates the CSV, and cleans up .raw files
    rows = extractor.extract(netlist_map, results_path=csv_path)

    if not rows:
        print("❌ ERROR: No metrics were extracted. The simulation may have failed.")
        sys.exit(1)

    # ==========================================================================
    # 6. CALCULATE REWARD
    # ==========================================================================
    print("\n🧠 Step 3: Calculating Normalized Reward...")
    row = rows[0] # We only simulated one netlist
    
    # Define our constraints and weights to match our 10V -> 5V design
    constraints = {
        "vin_min": 10.0,
        "vin_max": 14.0,
        "vout_target": 5.0,
        "efficiency_target": 0.80,
        "power_in": 15.0
    }
    
    weights = {
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
    
    reward_calc = RewardFunctionNorm()
    reward, details = reward_calc.calculate_reward(row, constraints, weights)

    # ==========================================================================
    # 7. SAVE & DISPLAY RESULTS
    # ==========================================================================
    final_output = {
        "circuit_name": "best_buck",
        "fitness_score": reward,
        "constraints_used": constraints,
        "weights_used": weights,
        "details": details
    }
    
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ Full pipeline complete! All artifacts saved in: {results_dir}")
    print("="*60)
    print(f"🏆 Final Calculated Reward: {reward:.4f} / 1.0000")
    print("="*60)
    print("Extracted Raw Metrics:")
    for key, val in details["raw_metrics"].items():
        print(f"  - {key}: {val:.4f}")
    
    print("\nLoss Breakdown:")
    for key, val in details["loss_breakdown"].items():
        print(f"  - {key}: {val:.4f}")

if __name__ == "__main__":
    main()