import sys
import os
import shutil
import json
from pathlib import Path
import pytest
import pandas as pd

# Tell Python to look one directory up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.simulation.local.ltspice_runner import LTSpiceSimulator
from pipeline.simulation.local.raw_extractor import RawExtractor
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm


# --- HARDCODED CHEATER NETLIST ---
# This ensures the test is completely self-contained and never skips due to missing files.
CHEATER_NETLIST = """* Generated: 2026-05-21T16:58:47
* Constraint: Phase 1: Standard 12V to 5V Logic (Buck)
* vin_min = 10
* vin_max = 14
* vout_target = 5
* efficiency_target = 0.8
* power_in = 15
* Candidate: 2 of 2
*
* 10-14V to 5V Boost Converter
Vin in 0 12
L1 in sw 33u
M1 sw gate 0 0 NMOS W=1 L=1
D1 sw out DIODE
C1 out 0 220u
Rload out 0 1.79
Vgate gate 0 PULSE(0 12 0 1n 1n 6.5u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1u 10m
.save V(*) I(*)
.end
"""

@pytest.fixture
def dummy_batch_setup():
    """
    Pytest Fixture: Sets up a temporary batch folder for the simulator,
    writes the hardcoded cheating netlist into it, generates a fake validation file,
    and automatically cleans it up after.
    """
    dummy_batch_id = "test_pytest_exploit/batch_1"
    dummy_batch_dir = Path("pipeline/data") / dummy_batch_id
    dummy_out_dir = dummy_batch_dir / "LLM_output"
    dummy_out_dir.mkdir(parents=True, exist_ok=True)
    
    netlist_stem = "Phase1_cons4_cand2_b1"
    
    # 1. Write the hardcoded string directly into a dummy .net file
    dummy_netlist_path = dummy_out_dir / f"{netlist_stem}.net"
    dummy_netlist_path.write_text(CHEATER_NETLIST)

    # 2. Create a fake validation_results.json to trick the simulator
    val_data = {
        netlist_stem: {"passed": True}
    }
    with open(dummy_batch_dir / "validation_results.json", "w") as f:
        json.dump(val_data, f)

    yield dummy_batch_id, netlist_stem

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

    # Reconstruct .net path from fixture data (fixture creates it at this location)
    net_file = Path("pipeline/data") / batch_id / "LLM_output" / f"{netlist_stem}.net"
    raw_dir  = Path("pipeline/data") / batch_id / "raw"
    results_path = Path("pipeline/data") / batch_id / "simulation_results.csv"

    # 1. Run Simulator — takes list[Path], returns netlist_map dict
    simulator   = LTSpiceSimulator(output_dir=raw_dir)
    netlist_map = simulator.simulate([net_file])

    # 2. Extract metrics via RawExtractor → CSV, returns list of row dicts
    extractor = RawExtractor(output_dir=raw_dir)
    rows = extractor.extract(netlist_map, results_path)
    batch_df = pd.DataFrame(rows)

    assert not batch_df.empty, "Simulator returned no data"

    row = batch_df[batch_df['source_file'] == netlist_stem]
    assert not row.empty, f"Missing metrics for {netlist_stem}"

    raw_metrics = row.iloc[0].to_dict()
    eff = raw_metrics.get("efficiency", 0)

    # 3. Assertions
    assert raw_metrics.get("power_in_W", 0) > 0, "Input power was not measured properly."
    assert eff > 0.0, "Efficiency is 0; simulation likely failed."
    assert eff < 0.95, f"BUG ACTIVE: Efficiency is artificially high ({eff * 100:.2f}%)"


def test_boost_converter_fitness_penalty(dummy_batch_setup):
    """
    Tests that a Boost Converter outputting >5V gets properly penalized by
    the reward function when evaluated against a 5V Buck constraint.
    """
    batch_id, netlist_stem = dummy_batch_setup

    constraint = {
        "vin_min": 10,
        "vin_max": 14,
        "vout_target": 5,
        "efficiency_target": 0.8,
        "power_in": 15
    }

    weights = {
        "v_out": 20.0,
        "efficiency": 10.0,
        "volume": 2.0,
        "component_cost": 0.05,
        "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0},
    }

    # Reconstruct .net path from fixture data (fixture creates it at this location)
    net_file = Path("pipeline/data") / batch_id / "LLM_output" / f"{netlist_stem}.net"
    raw_dir  = Path("pipeline/data") / batch_id / "raw"
    results_path = Path("pipeline/data") / batch_id / "simulation_results.csv"

    # Run Simulator + Extract metrics
    simulator   = LTSpiceSimulator(output_dir=raw_dir)
    netlist_map = simulator.simulate([net_file])
    extractor   = RawExtractor(output_dir=raw_dir)
    rows        = extractor.extract(netlist_map, results_path)
    batch_df    = pd.DataFrame(rows)

    row = batch_df[batch_df['source_file'] == netlist_stem]
    raw_metrics = row.iloc[0].to_dict() if not row.empty else {}

    # Calculate Reward
    reward_fn = RewardFunctionNorm()
    result  = reward_fn.calculate_reward(raw_metrics, constraint, weights=weights)
    fitness = result[0] if isinstance(result, tuple) else result

    print(f"\n---> THE FINAL PUNISHED FITNESS SCORE IS: {fitness:.4f} <---")

    # Boost outputting ~29V against a 5V target must be heavily penalized
    assert fitness < 0.60, f"BUG ACTIVE: Fitness score is suspiciously high ({fitness:.4f})"