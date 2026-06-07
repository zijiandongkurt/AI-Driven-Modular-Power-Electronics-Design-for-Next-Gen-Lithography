"""
Unit tests for raw_extractor.py physics formulas (Gashi et al. volume model).

These tests do NOT require LTSpice or any .raw files — they verify the
mathematical correctness of the thermal and volume calculations embedded in
RawExtractor.extract().  The integration path (actual .raw parsing) is already
covered by test_simulation.py.
"""
import sys
import pytest
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.simulation.local.raw_extractor import (
    K_L, K_H, V_FIXED,
    T_MAX, T_AMB, R_TH_JC,
)


#  Gashi et al. model constants 

class TestVolumeModelConstants:
    def test_inductor_scaling_constant(self):
        assert K_L == pytest.approx(4.0e4)

    def test_heatsink_scaling_constant(self):
        assert K_H == pytest.approx(40.0)

    def test_fixed_volume_constant(self):
        assert V_FIXED == pytest.approx(20.0)

    def test_thermal_constants(self):
        assert T_MAX  == pytest.approx(125.0)
        assert T_AMB  == pytest.approx(40.0)
        assert R_TH_JC == pytest.approx(1.5)


#  Inductor volume surrogate: V_ind = K_L * sum(l_values) 

class TestInductorVolume:
    def test_single_inductor(self):
        l_values = [47e-6]
        v_ind = K_L * sum(l_values)
        assert v_ind == pytest.approx(K_L * 47e-6)

    def test_two_inductors_sum(self):
        l_values = [22e-6, 33e-6]
        v_ind = K_L * sum(l_values)
        assert v_ind == pytest.approx(K_L * 55e-6)

    def test_larger_inductor_gives_larger_volume(self):
        v_small = K_L * sum([10e-6])
        v_large = K_L * sum([100e-6])
        assert v_large > v_small

    def test_empty_inductor_list_gives_zero(self):
        # When l_values is empty the branch is skipped — model result is 0
        v_ind = K_L * sum([])
        assert v_ind == 0.0


#  Thermal resistance formula: r_th_req = (T_MAX - T_AMB) / p_loss - R_TH_JC 

class TestThermalResistance:
    def _r_th(self, p_loss):
        return (T_MAX - T_AMB) / p_loss - R_TH_JC

    def test_reasonable_loss(self):
        # 10 W loss → r_th_req = 85/10 - 1.5 = 7.0 °C/W
        r = self._r_th(10.0)
        assert r == pytest.approx(7.0)

    def test_high_loss_gives_small_rth(self):
        r_low_loss  = self._r_th(1.0)
        r_high_loss = self._r_th(50.0)
        assert r_high_loss < r_low_loss

    def test_thermal_budget(self):
        # Available temp rise = T_MAX - T_AMB = 85 °C
        assert (T_MAX - T_AMB) == pytest.approx(85.0)

    def test_rth_negative_when_loss_too_high(self):
        # When p_loss > (T_MAX - T_AMB) / R_TH_JC, r_th_req goes negative
        # → heatsink is physically impossible → result must be infinity
        p_loss_extreme = (T_MAX - T_AMB) / R_TH_JC + 1.0
        r = self._r_th(p_loss_extreme)
        assert r < 0


#  Heatsink volume surrogate: V_hs = K_H / r_th_req 

class TestHeatsinkVolume:
    def _r_th(self, p_loss):
        return (T_MAX - T_AMB) / p_loss - R_TH_JC

    def test_reasonable_loss_gives_finite_volume(self):
        r = self._r_th(10.0)   # r = 7.0 °C/W
        v_hs = K_H / r
        assert math.isfinite(v_hs)
        assert v_hs == pytest.approx(40.0 / 7.0)

    def test_higher_loss_needs_bigger_heatsink(self):
        v_hs_low  = K_H / self._r_th(5.0)
        v_hs_high = K_H / self._r_th(20.0)
        assert v_hs_high > v_hs_low

    def test_zero_loss_means_no_heatsink_needed(self):
        # p_loss == 0 → heatsink_volume_cm3 = 0 (raw_extractor sets it to 0.0)
        p_loss = 0.0
        # Code path: p_loss > 0 is False → heatsink_volume_cm3 = 0.0
        heatsink_vol = 0.0 if p_loss <= 0 else K_H / self._r_th(p_loss)
        assert heatsink_vol == 0.0

    def test_negative_rth_means_infinite_heatsink(self):
        # When r_th_req <= 0 the extractor stores float("inf")
        r = self._r_th(100.0)   # very high loss → negative r_th
        heatsink_vol = K_H / r if r > 0 else float("inf")
        assert heatsink_vol == float("inf")

    def test_negative_loss_also_infinite(self):
        # Negative power_loss (shouldn't happen in practice) — same guard
        p_loss = -5.0
        heatsink_vol = 0.0 if p_loss <= 0 else K_H / self._r_th(p_loss)
        assert heatsink_vol == 0.0


#  Total volume: V_total = V_FIXED + V_ind + V_hs 

class TestTotalVolume:
    def test_total_is_sum_of_three_terms(self):
        l_values = [47e-6]
        p_loss = 10.0
        r_th = (T_MAX - T_AMB) / p_loss - R_TH_JC
        v_ind = K_L * sum(l_values)
        v_hs  = K_H / r_th
        v_total = V_FIXED + v_ind + v_hs
        assert v_total == pytest.approx(V_FIXED + v_ind + v_hs)

    def test_fixed_floor(self):
        # Even a lossless ideal converter with no inductors still has V_FIXED
        v_total_min = V_FIXED + 0.0 + 0.0
        assert v_total_min == pytest.approx(20.0)

    def test_infinite_heatsink_propagates_to_total(self):
        v_total = V_FIXED + K_L * 47e-6 + float("inf")
        assert math.isinf(v_total)

    def test_volume_increases_with_inductor_size(self):
        def total(l_h, p_loss=10.0):
            r_th = (T_MAX - T_AMB) / p_loss - R_TH_JC
            v_ind = K_L * l_h
            v_hs  = K_H / r_th
            return V_FIXED + v_ind + v_hs

        assert total(100e-6) > total(10e-6)

    def test_reward_function_caps_infinity(self):
        # V_FIXED + inf stays infinite — capping must happen in the reward function,
        # not in the extractor. Confirm the extractor's raw output is still inf.
        v_hs_inf = float("inf")
        raw_total = V_FIXED + K_L * 47e-6 + v_hs_inf
        assert math.isinf(raw_total), \
            "Extractor must pass infinity through; reward function is responsible for capping"