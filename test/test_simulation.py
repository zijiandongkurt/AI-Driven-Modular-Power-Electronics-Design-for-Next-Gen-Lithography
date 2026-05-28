import sys
import os
import shutil
import json
from pathlib import Path
import pytest
import pandas as pd

# Tell Python to look one directory up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.simulation.ltspice_runner import LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm 
 

@pytest.fixture
def dummy_batch_setup():
    """
    Pytest Fixture: Sets up a temporary batch folder for the simulator,
    copies the cheating netlist into it, generates a fake validation file,
    and automatically cleans it up after.
    """
    cheater_netlist = Path("pipeline/data/Run_006/database/Phase1_cons4_cand2_b1.net")
    
    if not cheater_netlist.exists():
        pytest.skip(f"Required test file not found: {cheater_netlist}")

    dummy_batch_id = "test_pytest_exploit/batch_1"
    dummy_batch_dir = Path("pipeline/data") / dummy_batch_id
    dummy_out_dir = dummy_batch_dir / "LLM_output"
    dummy_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy file into the dummy structure
    copied_netlist = dummy_out_dir / cheater_netlist.name
    shutil.copy(cheater_netlist, copied_netlist)

    # 2. Create a fake validation_results.json to trick the simulator
    val_data = {
        cheater_netlist.stem: {"passed": True}
    }
    with open(dummy_batch_dir / "validation_results.json", "w") as f:
        json.dump(val_data, f)

    yield dummy_batch_id, cheater_netlist.stem

    # --- TEARDOWN ---
    shutil.rmtree(Path("pipeline/data/test_pytest_exploit"), ignore_errors=True)


def test_boost_converter_efficiency_exploit(dummy_batch_setup):
    """
    Tests that the simulator correctly measures input power from the voltage 
    source, preventing Boost converters from scoring >100% efficiency.
    """
    batch_id, netlist_stem = dummy_batch_setup

    constraint = {
        "vin_min": 10,
        "vin_max": 14,
        "vout_target": 5,
        "efficiency_target": 0.8,
        "power_in": 15
    }

    # 1. Run Simulator (Returns a Pandas DataFrame)
    simulator = LTSpiceSimulator()
    batch_df = simulator.simulate(batch_id)
    
    # 2. Extract Data using Pandas filtering
    assert not batch_df.empty, "Simulator returned no data"
    
    # Find the specific row for our netlist
    row = batch_df[batch_df['source_file'] == netlist_stem]
    assert not row.empty, f"Missing metrics for {netlist_stem}"
    
    # Convert that specific row back into a normal Python dictionary
    raw_metrics = row.iloc[0].to_dict()
    eff = raw_metrics.get("efficiency", 0)

    # 3. Assertions
    assert raw_metrics.get("power_in_W", 0) > 0, "Input power was not measured properly."
    assert eff > 0.0, "Efficiency is 0; simulation likely failed."
    assert eff < 0.95, f"BUG ACTIVE: Efficiency is artificially high ({eff * 100:.2f}%)"


def test_boost_converter_fitness_penalty(dummy_batch_setup):
    """
    Tests that a 29V Boost Converter gets properly penalized by the reward 
    function when evaluated against a 5V Buck constraint.
    """
    batch_id, netlist_stem = dummy_batch_setup

    constraint = {
        "vin_min": 10,
        "vin_max": 14,
        "vout_target": 5,
        "efficiency_target": 0.8,
        "power_in": 15
    }
    
    # Standard weighting for the reward function
    weights = {
        "voltage_tracking": 1.0,
        "efficiency": 1.0,
        "ripple": 0.5,
        "volume": 0.5
    }

    # Run Simulation and Extract Dict
    simulator = LTSpiceSimulator()
    batch_df = simulator.simulate(batch_id)
    row = batch_df[batch_df['source_file'] == netlist_stem]
    raw_metrics = row.iloc[0].to_dict() if not row.empty else {}
    
    # Calculate Reward with weights
    reward_fn = RewardFunctionNorm()
    result = reward_fn.calculate_reward(raw_metrics, constraint, weights=weights)
    
    # Extract the float if it returns a tuple like (score, breakdown_dict)
    fitness = result[0] if isinstance(result, tuple) else result

    print(f"\n---> THE FINAL PUNISHED FITNESS SCORE IS: {fitness:.4f} <---")
    
    # Assert it gets heavily penalized for outputting ~29V
    assert fitness < 0.60, f"BUG ACTIVE: Fitness score is suspiciously high ({fitness:.4f})"