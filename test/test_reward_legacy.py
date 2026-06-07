"""
Unit tests for reward_function.py — the legacy (non-normalized) reward function.

Key difference from RewardFunctionNorm:
  - calculate_reward returns (-loss, details), so rewards are <= 0
  - No validation_reward(), normalize_group_rewards(), or fatal voltage penalty
  - No per-term capping or normalization — raw weighted sum
"""
import sys
import json
import pytest
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.reward_evaluation.reward_function import RewardFunction


#  Fixtures 

@pytest.fixture
def rf():
    return RewardFunction()


GOOD_METRICS = {
    "voltage_out_mean_V": 5.0,
    "efficiency": 0.92,
    "total_volume_cm3": 30.0,
    "count_mosfets": 1,
    "count_diodes": 1,
    "count_inductors": 1,
    "count_capacitors": 1,
}

CONSTRAINTS = {
    "vout_target": 5.0,
    "efficiency_target": 0.90,
}

WEIGHTS = {
    "v_out": 20.0,
    "efficiency": 10.0,
    "volume": 2.0,
    "component_cost": 0.05,
    "components": {
        "mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0
    },
}


#  calculate_loss 

class TestLegacyCalculateLoss:
    def test_perfect_circuit_near_zero_loss(self, rf):
        loss, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        # Volume and component penalties still apply, but voltage/efficiency are minimal
        assert loss >= 0.0

    def test_zero_voltage_increases_loss(self, rf):
        bad = {**GOOD_METRICS, "voltage_out_mean_V": 0.0}
        loss_bad, _  = rf.calculate_loss(bad, CONSTRAINTS, WEIGHTS)
        loss_good, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert loss_bad > loss_good

    def test_zero_efficiency_increases_loss(self, rf):
        bad = {**GOOD_METRICS, "efficiency": 0.0}
        loss_bad, _  = rf.calculate_loss(bad, CONSTRAINTS, WEIGHTS)
        loss_good, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert loss_bad > loss_good

    def test_infinite_volume_capped_at_max_penalty(self, rf):
        inf_metrics = {**GOOD_METRICS, "total_volume_cm3": float("inf")}
        loss, details = rf.calculate_loss(inf_metrics, CONSTRAINTS, WEIGHTS)
        assert loss < float("inf"), "Infinite volume must be capped"

    def test_nan_volume_capped_at_max_penalty(self, rf):
        import math
        nan_metrics = {**GOOD_METRICS, "total_volume_cm3": float("nan")}
        loss, _ = rf.calculate_loss(nan_metrics, CONSTRAINTS, WEIGHTS)
        assert not math.isnan(loss), "NaN volume must be handled"

    def test_loss_details_has_breakdown(self, rf):
        _, details = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert "loss_breakdown" in details
        assert "voltage_tracking_loss" in details["loss_breakdown"]
        assert "efficiency_loss" in details["loss_breakdown"]

    def test_more_components_increase_loss(self, rf):
        few  = {**GOOD_METRICS, "count_mosfets": 1, "count_diodes": 1}
        many = {**GOOD_METRICS, "count_mosfets": 5, "count_diodes": 5, "count_capacitors": 5}
        loss_few,  _ = rf.calculate_loss(few,  CONSTRAINTS, WEIGHTS)
        loss_many, _ = rf.calculate_loss(many, CONSTRAINTS, WEIGHTS)
        assert loss_many > loss_few


#  calculate_reward 

class TestLegacyCalculateReward:
    def test_reward_is_negative_loss(self, rf):
        loss, _   = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        reward, _ = rf.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert reward == pytest.approx(-loss)

    def test_reward_is_non_positive(self, rf):
        reward, _ = rf.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert reward <= 0.0

    def test_better_circuit_gives_higher_reward(self, rf):
        good = {**GOOD_METRICS, "voltage_out_mean_V": 5.0, "efficiency": 0.92}
        bad  = {**GOOD_METRICS, "voltage_out_mean_V": 0.0, "efficiency": 0.0}
        reward_good, _ = rf.calculate_reward(good, CONSTRAINTS, WEIGHTS)
        reward_bad,  _ = rf.calculate_reward(bad,  CONSTRAINTS, WEIGHTS)
        assert reward_good > reward_bad

    def test_contract_differs_from_norm_version(self, rf):
        # Legacy returns (-loss) which is <= 0.
        # RewardFunctionNorm returns a value in [0.5, 1.0].
        # This test documents that the two are NOT interchangeable.
        from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
        rf_norm = RewardFunctionNorm()
        legacy_reward, _ = rf.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        norm_reward,   _ = rf_norm.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert legacy_reward <= 0.0
        assert 0.5 <= norm_reward <= 1.0


#  process_csv_to_json 

class TestLegacyProcessCsvToJson:
    def test_missing_csv_returns_error(self, rf, tmp_path):
        result, out_path = rf.process_csv_to_json(
            csv_file_path=tmp_path / "nonexistent.csv",
            output_json_path=tmp_path / "out.json",
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
        )
        data = json.loads(result)
        assert "Error" in data
        assert out_path is None

    def test_valid_csv_produces_negative_fitness(self, rf, tmp_path):
        df = pd.DataFrame([{
            "source_file": "cand1",
            "voltage_out_mean_V": 5.0,
            "efficiency": 0.92,
            "total_volume_cm3": 30.0,
            "count_mosfets": 1, "count_diodes": 1,
            "count_inductors": 1, "count_capacitors": 1,
        }])
        csv_path = tmp_path / "simulation_results.csv"
        df.to_csv(csv_path, index=False)

        result, out_path = rf.process_csv_to_json(
            csv_file_path=csv_path,
            output_json_path=tmp_path / "out.json",
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
        )
        data = json.loads(result)
        fitness = data["circuits"]["cand1"]["fitness_score"]
        assert fitness <= 0.0, "Legacy reward must be non-positive"

    def test_no_grpo_normalization_in_output(self, rf, tmp_path):
        df = pd.DataFrame([{
            "source_file": "cand1",
            "voltage_out_mean_V": 5.0,
            "efficiency": 0.92,
            "total_volume_cm3": 30.0,
            "count_mosfets": 1, "count_diodes": 1,
            "count_inductors": 1, "count_capacitors": 1,
        }])
        csv_path = tmp_path / "simulation_results.csv"
        df.to_csv(csv_path, index=False)

        result, _ = rf.process_csv_to_json(
            csv_file_path=csv_path,
            output_json_path=tmp_path / "out.json",
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
        )
        data = json.loads(result)
        # Legacy function does not add grpo_reward — that's a RewardFunctionNorm feature
        assert "grpo_reward" not in data["circuits"]["cand1"]

    def test_output_file_written(self, rf, tmp_path):
        df = pd.DataFrame([{
            "source_file": "cand1",
            "voltage_out_mean_V": 5.0,
            "efficiency": 0.92,
            "total_volume_cm3": 30.0,
            "count_mosfets": 1, "count_diodes": 1,
            "count_inductors": 1, "count_capacitors": 1,
        }])
        csv_path = tmp_path / "simulation_results.csv"
        out_path = tmp_path / "out.json"
        df.to_csv(csv_path, index=False)

        _, saved = rf.process_csv_to_json(
            csv_file_path=csv_path,
            output_json_path=out_path,
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
        )
        assert saved is not None
        assert out_path.exists()

    def test_missing_source_file_column_returns_error(self, rf, tmp_path):
        df = pd.DataFrame([{"voltage_out_mean_V": 5.0}])  # no source_file column
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        result, out_path = rf.process_csv_to_json(
            csv_file_path=csv_path,
            output_json_path=tmp_path / "out.json",
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
        )
        data = json.loads(result)
        assert "Error" in data
        assert out_path is None