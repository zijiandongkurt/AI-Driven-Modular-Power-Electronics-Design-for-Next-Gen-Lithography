from PySpice.Spice.NgSpice.Server import SpiceServer
from pathlib import Path
import re
import subprocess
import json
import pandas as pd
import numpy as np

# Thermal constants
T_MAX    = 125.0  # Max junction temperature (degC)
T_AMB    =  40.0  # Ambient temperature (degC)
R_TH_JC  =   1.5  # Junction-to-case + case-to-sink thermal resistance (degC/W)

# Volume model constants (Gashi et al.)
K_L     = 4.0e4  # Inductance-to-volume fitting constant (cm3/H)
K_H     = 40.0   # Heatsink fitting constant (cm3*degC/W)
V_FIXED = 20.0   # Fixed converter volume — PCB, capacitors, connectors (cm3)


def _to_float(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip().lower()
    # meg must be checked before m to avoid partial match
    SUFFIXES = [
        ("meg", 1e6), ("g", 1e9),
        ("f", 1e-15), ("p", 1e-12), ("n", 1e-9), ("u", 1e-6),
        ("m", 1e-3),  ("k", 1e3),
    ]
    for suffix, mult in SUFFIXES:
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


COMPONENT_WEIGHTS = {
    "M": 3.0,   # MOSFETs
    "D": 1.5,   # Diodes
    "L": 2.5,   # Inductors
    "C": 1.0,   # Capacitors
    "R": 0.2,   # Resistors
}


def _parse_netlist_metadata(netlist_text):
    """Extract component counts, inductor values, and switching frequency
    directly from raw netlist text (replaces SpiceEditor).

    PARAMS:
    netlist_text <str> : Full text of the .net file

    RETURNS:
    meta <dict> : counts, l_values, switching_freq_Hz
    """
    counts = {k: 0 for k in COMPONENT_WEIGHTS}
    l_values = []
    f_sw = float("nan")

    for line in netlist_text.splitlines():
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue

        ref = line.split()[0].upper()
        prefix = ref[0]

        if prefix in counts:
            counts[prefix] += 1

        # Inductor value: L<ref> <n+> <n-> <value>
        if prefix == "L":
            parts = line.split()
            if len(parts) >= 4:
                try:
                    l_values.append(_to_float(parts[3]))
                except ValueError:
                    pass

        # Switching frequency from PULSE period (7th parameter)
        if prefix == "V" and "PULSE" in line.upper():
            pulse_match = re.search(r'PULSE\s*\(([^)]+)\)', line, re.IGNORECASE)
            if pulse_match:
                parts = pulse_match.group(1).split()
                if len(parts) >= 7:
                    try:
                        period = _to_float(parts[6])
                        if period > 0 and np.isnan(f_sw):
                            f_sw = 1.0 / period
                    except ValueError:
                        pass

    return {"counts": counts, "l_values": l_values, "switching_freq_Hz": f_sw}


class NGSpiceSimulator():
    def __init__(self, ngspice_command="ngspice"):
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR.parent / "data"
        self.ngspice_command = ngspice_command
        self._server = SpiceServer(spice_command=ngspice_command)

    def _verify_ngspice(self):
        """Check that the ngspice binary is reachable before attempting any simulation."""
        try:
            result = subprocess.run(
                [self.ngspice_command, "--version"],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def simulate(self, batchID):
        """Simulates all netlists in data/<batchID>/llm_output/ that passed validation.
        Reads validation_results.json to determine which netlists to simulate.

        PARAMS:
        batchID <string> : The ID of a Batch

        RETURNS:
        results <DataFrame> : Refined scalar metrics dataframe, one row per simulation run
        """
        if not self._verify_ngspice():
            raise EnvironmentError(
                f"ngspice binary not found at '{self.ngspice_command}'.\n"
                f"Install it with:  conda install -c conda-forge ngspice\n"
                f"Or if compiled into user space, pass the full path:\n"
                f"    NGSpiceSimulator(ngspice_command='~/.local/bin/ngspice')"
            )

        batch_dir        = self.DATA_DIR / batchID
        llm_output_dir   = batch_dir / "LLM_output"
        val_results_path = batch_dir / "validation_results.json"

        assert llm_output_dir.exists(),   f"llm_output folder not found: {llm_output_dir.resolve()}"
        assert val_results_path.exists(), f"validation_results.json not found: {val_results_path.resolve()}"

        val_results = json.loads(val_results_path.read_text())
        valid_stems = {stem for stem, data in val_results.items() if data.get("passed", False)}

        if not valid_stems:
            print(f"WARNING: No valid netlists found in validation_results.json for batch '{batchID}'")
            return pd.DataFrame()

        ok, failed = 0, 0
        all_rows = []

        for netpath in sorted(llm_output_dir.glob("*.net")):
            if netpath.stem not in valid_stems:
                print(f"SKIPPING (failed validation): {netpath.name}")
                continue

            netlist_text = netpath.read_text()
            meta = _parse_netlist_metadata(netlist_text)

            print(f"Simulating: {netpath.name}")
            try:
                raw_file = self._server(spice_input=netlist_text)
                rows = self._extract_metrics(raw_file, stem=netpath.stem, meta=meta)
                all_rows.extend(rows)
                ok += 1
                print(f"  OK — {len(rows)} step(s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1

        print(f"Batch '{batchID}' done — {ok}/{ok + failed} successful")
        if failed:
            print(f"WARNING: {failed} simulation(s) failed in batch '{batchID}'")

        results = pd.DataFrame(all_rows)

        csv_path = self.DATA_DIR / batchID / "simulation_results.csv"
        results.to_csv(csv_path, index=False)
        print(f"Saved {len(all_rows)} run(s) -> {csv_path}")

        return results

    def _extract_metrics(self, raw_file, stem, meta):
        """Extract scalar performance metrics from a PySpice RawFile object.

        PySpice gives us raw_file.variables — a dict keyed by signal name
        (e.g. 'v(out)', 'i(vin)'), each with a .data numpy array.

        PARAMS:
        raw_file : PySpice RawFile returned by SpiceServer
        stem     <str>  : netlist filename stem (for tagging rows)
        meta     <dict> : pre-parsed netlist metadata (counts, l_values, f_sw)

        RETURNS:
        rows <list[dict]> : one dict per simulation step
        """
        # PySpice doesn't expose steps the same way PyLTSpice does.
        # For a plain .tran run there is exactly one dataset — treat as step 0.
        vars_ = {k.lower(): v.data for k, v in raw_file.variables.items()}

        if "time" not in vars_:
            print(f"  WARNING: no time vector in {stem}, skipping")
            return []

        t_all = vars_["time"]

        # Steady-state slice: last 20 % of simulation time
        t_max  = t_all.max()
        mask   = t_all >= t_max * 0.8
        t      = t_all[mask]
        t_span = t[-1] - t[0]

        if t_span == 0:
            print(f"  WARNING: zero time span in {stem}, skipping")
            return []

        # Helper: slice a signal to the steady-state window
        def ss(key):
            return vars_[key][mask] if key in vars_ else None

        counts   = meta.get("counts", {})
        l_values = meta.get("l_values", [])

        row = {"source_file": stem, "step": 0}

        # ── Output voltage ──────────────────────────────────────────────────
        v_out = ss("v(out)")
        if v_out is not None:
            row["voltage_out_mean_V"]   = np.trapezoid(v_out, t) / t_span
            row["voltage_out_ripple_V"] = v_out.max() - v_out.min()

        # ── Input voltage ───────────────────────────────────────────────────
        v_in = ss("v(in)")
        if v_in is not None:
            row["voltage_in_mean_V"]   = np.trapezoid(v_in, t) / t_span
            row["voltage_in_ripple_V"] = v_in.max() - v_in.min()

        # ── Conversion ratio ────────────────────────────────────────────────
        if "voltage_out_mean_V" in row and "voltage_in_mean_V" in row \
                and row["voltage_in_mean_V"] != 0:
            row["conversion_ratio"] = row["voltage_out_mean_V"] / row["voltage_in_mean_V"]

        # ── Inductor current (worst-case across all inductors) ──────────────
        i_l_keys = [k for k in vars_ if re.match(r'i\(l\d*\)', k)]
        if i_l_keys:
            all_mean, all_rms, all_peak, all_ripple, all_min = [], [], [], [], []
            for key in i_l_keys:
                vals = vars_[key][mask]
                all_mean.append(np.trapezoid(vals, t) / t_span)
                all_rms.append(float(np.sqrt(np.trapezoid(vals**2, t) / t_span)))
                all_peak.append(vals.max())
                all_ripple.append(vals.max() - vals.min())
                all_min.append(vals.min())
            row["inductor_current_mean_A"]   = max(all_mean)
            row["inductor_current_rms_A"]    = max(all_rms)
            row["inductor_current_peak_A"]   = max(all_peak)
            row["inductor_current_ripple_A"] = max(all_ripple)
            row["is_ccm"]                    = int(min(all_min) > 1e-6)

        # ── Switching frequency ─────────────────────────────────────────────
        f_sw = meta.get("switching_freq_Hz", float("nan"))
        if np.isnan(f_sw) and i_l_keys:
            i_l_vals = vars_[i_l_keys[0]][mask]
            peaks = np.where(
                (i_l_vals[1:-1] > i_l_vals[:-2]) &
                (i_l_vals[1:-1] > i_l_vals[2:])
            )[0] + 1
            if len(peaks) >= 2:
                f_sw = float(1.0 / np.diff(t[peaks]).mean())
        row["switching_freq_Hz"] = f_sw

        # ── Output power ────────────────────────────────────────────────────
        # ngspice names load current i(rload)
        i_load = ss("i(rload)")
        if i_load is not None and "voltage_out_mean_V" in row:
            row["load_current_mean_A"] = abs(np.trapezoid(i_load, t) / t_span)
            row["power_out_W"]         = row["voltage_out_mean_V"] * row["load_current_mean_A"]

        # ── Input power & efficiency ────────────────────────────────────────
        # ngspice reports source current as i(vin)
        i_vin = ss("i(vin)")
        if i_vin is not None and v_in is not None:
            p_in_inst = v_in * np.abs(i_vin)
            p_in = np.trapezoid(p_in_inst, t) / t_span
            if p_in > 0:
                row["power_in_W"] = p_in
                if "power_out_W" in row:
                    row["efficiency"] = row["power_out_W"] / row["power_in_W"]

        # ── Losses & thermal ────────────────────────────────────────────────
        if "power_in_W" in row and "power_out_W" in row:
            p_loss              = row["power_in_W"] - row["power_out_W"]
            row["power_loss_W"] = p_loss
            if p_loss > 0:
                r_th_req = (T_MAX - T_AMB) / p_loss - R_TH_JC
                row["heatsink_thermal_resistance_CW"] = r_th_req
                row["heatsink_volume_cm3"]            = K_H / r_th_req if r_th_req > 0 else float("inf")
            else:
                row["heatsink_thermal_resistance_CW"] = float("inf")
                row["heatsink_volume_cm3"]            = 0.0

        # ── Switch voltage stress ───────────────────────────────────────────
        v_sw_keys = [k for k in vars_ if k == "v(sw)" or re.match(r'v\(sw\d+\)', k)]
        if v_sw_keys:
            row["switch_voltage_peak_V"] = max(vars_[k][mask].max() for k in v_sw_keys)

        # ── Switch current stress ───────────────────────────────────────────
        # ngspice drain current is i(m1:d) — the :d suffix selects the drain terminal
        id_keys = [k for k in vars_ if re.match(r'i\(m\d+:d\)', k)]
        if id_keys:
            sw_peaks, sw_rms = [], []
            for key in id_keys:
                vals = vars_[key][mask]
                sw_peaks.append(np.max(np.abs(vals)))
                sw_rms.append(float(np.sqrt(np.trapezoid(vals**2, t) / t_span)))
            row["switch_current_peak_A"] = max(sw_peaks)
            row["switch_current_rms_A"]  = max(sw_rms)

        # ── Inductor volume surrogate ───────────────────────────────────────
        if l_values:
            row["inductor_volume_cm3"] = K_L * sum(l_values)

        # ── Total converter volume ──────────────────────────────────────────
        if "inductor_volume_cm3" in row and "heatsink_volume_cm3" in row:
            row["total_volume_cm3"] = V_FIXED + row["inductor_volume_cm3"] + row["heatsink_volume_cm3"]

        # ── Component counts ────────────────────────────────────────────────
        row["count_mosfets"]    = counts.get("M", 0)
        row["count_diodes"]     = counts.get("D", 0)
        row["count_inductors"]  = counts.get("L", 0)
        row["count_capacitors"] = counts.get("C", 0)

        return [row]


# # testing
# simo = NGSpiceSimulator()
# _ = simo.simulate("batch_2")