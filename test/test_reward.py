import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm, VALIDATION_CHECK_WEIGHTS


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def rf():
    return RewardFunctionNorm()


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


# ── validation_reward ─────────────────────────────────────────────────────────

class TestValidationReward:
    def test_all_pass_gives_minus_half(self, rf):
        all_pass = {k: True for k in VALIDATION_CHECK_WEIGHTS}
        reward = rf.validation_reward(all_pass)
        assert pytest.approx(reward, abs=1e-6) == -0.5

    def test_all_fail_gives_minus_one(self, rf):
        all_fail = {k: False for k in VALIDATION_CHECK_WEIGHTS}
        reward = rf.validation_reward(all_fail)
        assert pytest.approx(reward, abs=1e-6) == -1.0

    def test_range_is_valid(self, rf):
        partial = {k: (i % 2 == 0) for i, k in enumerate(VALIDATION_CHECK_WEIGHTS)}
        reward = rf.validation_reward(partial)
        assert -1.0 <= reward <= -0.5

    def test_empty_checks_gives_minus_one(self, rf):
        assert rf.validation_reward({}) == -1.0

    def test_strictly_below_simulation_range(self, rf):
        # Simulation rewards are in [0.5, 1.0] — validation must stay below 0.5
        all_pass = {k: True for k in VALIDATION_CHECK_WEIGHTS}
        assert rf.validation_reward(all_pass) < 0.5


# ── calculate_loss ────────────────────────────────────────────────────────────

class TestCalculateLoss:
    def test_perfect_circuit_has_low_loss(self, rf):
        loss, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert loss < 0.1, f"Perfect circuit has unexpectedly high loss: {loss}"

    def test_loss_is_in_unit_interval(self, rf):
        loss, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert 0.0 <= loss <= 1.0

    def test_zero_voltage_penalized(self, rf):
        bad = {**GOOD_METRICS, "voltage_out_mean_V": 0.0}
        loss_bad, _ = rf.calculate_loss(bad, CONSTRAINTS, WEIGHTS)
        loss_good, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert loss_bad > loss_good

    def test_zero_efficiency_penalized(self, rf):
        bad = {**GOOD_METRICS, "efficiency": 0.0}
        loss_bad, _ = rf.calculate_loss(bad, CONSTRAINTS, WEIGHTS)
        loss_good, _ = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert loss_bad > loss_good

    def test_nan_metrics_handled_gracefully(self, rf):
        nan_metrics = {k: float("nan") for k in GOOD_METRICS}
        loss, _ = rf.calculate_loss(nan_metrics, CONSTRAINTS, WEIGHTS)
        assert 0.0 <= loss <= 1.0

    def test_loss_details_has_breakdown(self, rf):
        _, details = rf.calculate_loss(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert "loss_breakdown" in details
        assert "voltage_tracking_loss" in details["loss_breakdown"]

    def test_higher_component_count_increases_loss(self, rf):
        few = {**GOOD_METRICS, "count_mosfets": 1, "count_diodes": 1}
        many = {**GOOD_METRICS, "count_mosfets": 5, "count_diodes": 5, "count_capacitors": 5}
        loss_few, _ = rf.calculate_loss(few, CONSTRAINTS, WEIGHTS)
        loss_many, _ = rf.calculate_loss(many, CONSTRAINTS, WEIGHTS)
        assert loss_many >= loss_few


# ── calculate_reward ──────────────────────────────────────────────────────────

class TestCalculateReward:
    def test_reward_in_valid_range(self, rf):
        reward, _ = rf.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert 0.5 <= reward <= 1.0

    def test_perfect_metrics_give_high_reward(self, rf):
        reward, _ = rf.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert reward > 0.90, f"Expected reward > 0.90, got {reward}"

    def test_wrong_voltage_triggers_fatal_penalty(self, rf):
        bad_voltage = {**GOOD_METRICS, "voltage_out_mean_V": 30.0}  # Way off target of 5V
        reward_bad, details = rf.calculate_reward(bad_voltage, CONSTRAINTS, WEIGHTS)
        reward_good, _ = rf.calculate_reward(GOOD_METRICS, CONSTRAINTS, WEIGHTS)
        assert reward_bad < reward_good
        assert details.get("fatal_penalty_applied") is True

    def test_slight_voltage_miss_no_fatal_penalty(self, rf):
        # 5.5V is within 50% of 5V target (safe_margin = 2.5V)
        close = {**GOOD_METRICS, "voltage_out_mean_V": 5.5}
        _, details = rf.calculate_reward(close, CONSTRAINTS, WEIGHTS)
        assert "fatal_penalty_applied" not in details

    def test_reward_never_below_floor(self, rf):
        worst = {
            "voltage_out_mean_V": 100.0,
            "efficiency": 0.0,
            "total_volume_cm3": 99999.0,
            "count_mosfets": 20, "count_diodes": 20,
            "count_inductors": 20, "count_capacitors": 20,
        }
        reward, _ = rf.calculate_reward(worst, CONSTRAINTS, WEIGHTS)
        assert reward >= 0.0  # must not go negative


# ── normalize_group_rewards ───────────────────────────────────────────────────

class TestNormalizeGroupRewards:
    def test_output_keys_match_input(self, rf):
        rewards = {"a": 0.7, "b": 0.8, "c": 0.6}
        normalized = rf.normalize_group_rewards(rewards)
        assert set(normalized.keys()) == set(rewards.keys())

    def test_normalized_mean_near_zero(self, rf):
        rewards = {"a": 0.6, "b": 0.7, "c": 0.8, "d": 0.9}
        normalized = rf.normalize_group_rewards(rewards)
        mean = np.mean(list(normalized.values()))
        assert abs(mean) < 0.1

    def test_uniform_rewards_dont_crash(self, rf):
        rewards = {"a": 0.75, "b": 0.75, "c": 0.75}
        normalized = rf.normalize_group_rewards(rewards)
        # With zero std, all should be clipped to 0
        for v in normalized.values():
            assert v == 0.0

    def test_values_clamped_to_range(self, rf):
        rewards = {"good": 1.0, "bad": -1.0, "extreme": 100.0}
        normalized = rf.normalize_group_rewards(rewards)
        for v in normalized.values():
            assert -4.0 <= v <= 4.0


# ── process_csv_to_json ───────────────────────────────────────────────────────

class TestProcessCsvToJson:
    def test_missing_csv_with_no_validation_returns_error(self, rf, tmp_path):
        result, out_path = rf.process_csv_to_json(
            csv_file_path=tmp_path / "nonexistent.csv",
            output_json_path=tmp_path / "out.json",
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
        )
        data = json.loads(result)
        assert "Error" in data
        assert out_path is None

    def test_invalid_candidates_get_validation_penalty(self, rf, tmp_path):
        # Create a validation results file with one failing candidate
        val_results = {
            "cand1": {
                "passed": False,
                "checks": {k: False for k in VALIDATION_CHECK_WEIGHTS}
            }
        }
        val_path = tmp_path / "validation_results.json"
        val_path.write_text(json.dumps(val_results))

        result, out_path = rf.process_csv_to_json(
            csv_file_path=tmp_path / "nonexistent.csv",
            output_json_path=tmp_path / "out.json",
            constraints=CONSTRAINTS,
            weights=WEIGHTS,
            validation_json_path=val_path,
        )
        data = json.loads(result)
        assert "cand1" in data["circuits"]
        assert data["circuits"]["cand1"]["fitness_score"] == pytest.approx(-1.0, abs=0.01)
        assert out_path is not None  # still writes output when at least one candidate exists

    def test_grpo_rewards_written_to_output(self, rf, tmp_path):
        import pandas as pd
        # Write a minimal CSV
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
        assert "grpo_reward" in data["circuits"]["cand1"]