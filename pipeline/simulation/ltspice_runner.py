from PyLTSpice import LTspice, SimRunner, SpiceEditor, RawRead
from pathlib import Path
import re
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


class TopologySimulator():
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR.parent / "data"

    def _onSimulationComplete(self, raw_file, log_file):
        """Event informing completion of a simulation of a particular netlist"""
        print(f"SIMULATION COMPLETE: {raw_file}, {log_file}")
        raw_path = Path(raw_file)

        for suffix in [".net", ".db", ".op.raw"]:
            unwanted = raw_path.with_suffix(suffix) if not suffix.startswith(".op") \
                    else raw_path.with_name(raw_path.stem + suffix)
            if unwanted.exists():
                unwanted.unlink()

    def simulate(self, batchID):
        """Simulates all netlists in data/<batchID>/llm_output/ that passed validation.
        Reads validation_results.json to determine which netlists to simulate.

        PARAMS:
        batchID <string> : The ID of a Batch

        RETURNS:
        results <DataFrame> : Refined scalar metrics dataframe, one row per simulation run
        """
        import json

        batch_dir       = self.DATA_DIR / batchID
        llm_output_dir  = batch_dir / "llm_output"
        val_results_path = batch_dir / "validation_results.json"

        assert llm_output_dir.exists(),   f"llm_output folder not found: {llm_output_dir.resolve()}"
        assert val_results_path.exists(), f"validation_results.json not found: {val_results_path.resolve()}"

        # Load validation results and collect only passing netlist stems
        val_results  = json.loads(val_results_path.read_text())
        valid_stems  = {stem for stem, data in val_results.items() if data.get("passed", False)}

        if not valid_stems:
            print(f"WARNING: No valid netlists found in validation_results.json for batch '{batchID}'")
            return pd.DataFrame()

        # Init temp output path for .raw files (stays local to simulation/)
        output_path = self.BASE_DIR / "output" / batchID
        output_path.mkdir(parents=True, exist_ok=True)

        # Init runner
        runner = SimRunner(output_folder=output_path, simulator=LTspice, parallel_sims=4)

        # Extract netlist metadata via SpiceEditor and queue valid ones for simulation
        netlist_map = {}
        for netpath in llm_output_dir.glob("*.net"):
            if netpath.stem not in valid_stems:
                print(f"SKIPPING (failed validation): {netpath.name}")
                continue

            net      = SpiceEditor(netpath)
            l_refs   = net.get_components("L")
            l_values = [_to_float(net.get_component_value(l)) for l in l_refs]
            counts   = {k: len(net.get_components(k)) for k in COMPONENT_WEIGHTS}

            # Extract switching frequency from gate voltage PULSE period
            f_sw = float("nan")
            for vref in net.get_components("V"):
                val = net.get_component_value(vref)
                if val and "PULSE" in val.upper():
                    parts = val.upper().replace("PULSE(","").replace(")","").split()
                    if len(parts) >= 7:
                        period = _to_float(parts[6])
                        if period and period > 0:
                            f_sw = 1.0 / period
                            break

            netlist_map[netpath.stem] = {
                "counts":            counts,
                "l_values":          l_values,
                "switching_freq_Hz": f_sw,
            }


            runner.run(net,
                       run_filename=netpath.name,
                       callback=self._onSimulationComplete)

        # Wait for all parallel sims to finish
        runner.wait_completion()
        print(f"Batch '{batchID}' done — {runner.okSim}/{runner.runno} successful")

        if runner.okSim < runner.runno:
            print(f"WARNING: {runner.runno - runner.okSim} simulation(s) failed in batch '{batchID}'")

        # Extract and return refined metrics
        results = self.__getResultsOf(batchID=batchID, netlist_map=netlist_map)
        return results

    def __getResultsOf(self, batchID, netlist_map):
        """Extracts scalar performance metrics from .raw simulation outputs and
        combines them with netlist-derived metrics into a single refined DataFrame.

        Each row represents one simulation run. No raw time-series data is retained.

        PARAMS:
        batchID     <string> : The ID of a Batch
        netlist_map <dict>   : Maps netlist stem -> pre-extracted metadata from SpiceEditor

        RETURNS:
        combined <DataFrame> : Refined scalar metrics, one row per simulation run
        """

        output_dir = self.BASE_DIR / "output" / batchID
        assert output_dir.exists(), f"No results found for batch: {batchID}"

        raw_files = list(output_dir.glob("*.raw"))
        assert raw_files, f"No .raw files found in: {output_dir}"

        all_rows = []

        for raw_path in raw_files:
            try:
                raw   = RawRead(raw_path)
                steps = raw.get_steps()
            except Exception as e:
                print(f"WARNING: Could not read {raw_path.name} — {e}")
                continue

            meta     = netlist_map.get(raw_path.stem, {})
            counts   = meta.get("counts", {})
            l_values = meta.get("l_values", [])

            for step in steps:
                try:
                    df = raw.to_dataframe(step=step)
                except Exception as e:
                    print(f"WARNING: Could not parse step {step} of {raw_path.name} — {e}")
                    continue

                df.columns = [c.lower() for c in df.columns]

                # time is the index in PyLTSpice dataframes — reset it to a column
                df = df.reset_index()
                if "time" not in df.columns or df.empty:
                    print(f"WARNING: No time column in {raw_path.name} step {step}, skipping")
                    continue

                # ── Steady-state slice (last 20% of simulation time) ──────────────
                t_max = df["time"].max()
                df_ss = df[df["time"] >= t_max * 0.8].copy()

                t      = df_ss["time"].values
                t_span = t[-1] - t[0]

                if t_span == 0:
                    print(f"WARNING: Zero time span in {raw_path.name} step {step}, skipping")
                    continue

                row = {
                    "source_file": raw_path.stem,
                    "step":        step,
                }

                # ── Output voltage ────────────────────────────────────────────────
                if "v(out)" in df_ss:
                    v_out_vals                  = df_ss["v(out)"].values
                    row["voltage_out_mean_V"]   = np.trapezoid(v_out_vals, t) / t_span
                    row["voltage_out_ripple_V"] = v_out_vals.max() - v_out_vals.min()

                # ── Input voltage ─────────────────────────────────────────────────
                if "v(in)" in df_ss:
                    v_in_vals                   = df_ss["v(in)"].values
                    row["voltage_in_mean_V"]    = np.trapezoid(v_in_vals, t) / t_span
                    row["voltage_in_ripple_V"]  = v_in_vals.max() - v_in_vals.min()

                # ── Conversion ratio ──────────────────────────────────────────────
                if "voltage_out_mean_V" in row and "voltage_in_mean_V" in row and row["voltage_in_mean_V"] != 0:
                    row["conversion_ratio"] = row["voltage_out_mean_V"] / row["voltage_in_mean_V"]

                # ── Inductor current (worst-case across all inductors) ────────────
                i_l_cols = [c for c in df_ss.columns if c.startswith("i(l")]
                if i_l_cols:
                    all_mean, all_rms, all_peak, all_ripple, all_min = [], [], [], [], []
                    for col in i_l_cols:
                        vals = df_ss[col].values
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

                # ── Switching frequency ───────────────────────────────────────────
                f_sw = meta.get("switching_freq_Hz", float("nan"))
                if np.isnan(f_sw) and i_l_cols:
                    i_l_vals = df_ss[i_l_cols[0]].values
                    peaks = np.where(
                        (i_l_vals[1:-1] > i_l_vals[:-2]) &
                        (i_l_vals[1:-1] > i_l_vals[2:])
                    )[0] + 1
                    if len(peaks) >= 2:
                        f_sw = float(1.0 / np.diff(t[peaks]).mean())
                row["switching_freq_Hz"] = f_sw

                # ── Output power ──────────────────────────────────────────────────
                i_load_col = next(
                    (c for c in df_ss.columns if "rload" in c or "r_load" in c), None
                )
                if i_load_col and "voltage_out_mean_V" in row:
                    i_load_vals                = df_ss[i_load_col].values
                    row["load_current_mean_A"] = abs(np.trapezoid(i_load_vals, t) / t_span)
                    row["power_out_W"]         = row["voltage_out_mean_V"] * row["load_current_mean_A"]

                # ── Input power & efficiency ──────────────────────────────────────
                # Compute P_in = V(in) * sum(Id(Mx)) — MOSFET drain currents are always
                # saved and represent the switched input current reliably
                if "v(in)" in df_ss.columns:
                    v_in_vals  = df_ss["v(in)"].values
                    id_cols    = [c for c in df_ss.columns if c.startswith("id(m")]
                    if id_cols:
                        i_sw_total = sum(df_ss[c].values for c in id_cols)
                        p_in_inst  = v_in_vals * np.abs(i_sw_total)
                        p_in       = np.trapezoid(p_in_inst, t) / t_span
                        if p_in > 0:
                            row["power_in_W"] = p_in
                            if "power_out_W" in row:
                                row["efficiency"] = row["power_out_W"] / row["power_in_W"]

                # ── Losses & thermal ──────────────────────────────────────────────
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

                # ── Switch voltage stress — max across all sw nodes ───────────────
                v_ds_cols = [c for c in df_ss.columns if c == "v(sw)" or re.match(r'v\(sw\d+\)', c) or c == "v(ds)"]
                if v_ds_cols:
                    row["switch_voltage_peak_V"] = max(df_ss[c].abs().max() for c in v_ds_cols)

                # ── Switch current stress ─────────────────────────────────────────
                i_sw_cols = [c for c in df_ss.columns if c.startswith("id(m")]
                if i_sw_cols:
                    sw_peaks, sw_rms = [], []
                    for col in i_sw_cols:
                        vals = df_ss[col].values
                        sw_peaks.append(np.max(np.abs(vals)))
                        sw_rms.append(float(np.sqrt(np.trapezoid(vals**2, t) / t_span)))
                    row["switch_current_peak_A"] = max(sw_peaks)
                    row["switch_current_rms_A"]  = max(sw_rms)

                # ── Inductor volume surrogate ─────────────────────────────────────
                if l_values:
                    row["inductor_volume_cm3"] = K_L * sum(l_values)

                # ── Total converter volume ────────────────────────────────────────
                if "inductor_volume_cm3" in row and "heatsink_volume_cm3" in row:
                    row["total_volume_cm3"] = V_FIXED + row["inductor_volume_cm3"] + row["heatsink_volume_cm3"]

                # ── Component counts ──────────────────────────────────────────────
                row["count_mosfets"]    = counts.get("M", 0)
                row["count_diodes"]     = counts.get("D", 0)
                row["count_inductors"]  = counts.get("L", 0)
                row["count_capacitors"] = counts.get("C", 0)

                all_rows.append(row)

        combined = pd.DataFrame(all_rows)

        csv_path = self.DATA_DIR / batchID / "simulation_results.csv"
        combined.to_csv(csv_path, index=False)
        print(f"Saved {len(all_rows)} run(s) -> {csv_path}")

        return combined


# testing
simo = TopologySimulator()
_ = simo.simulate("batch_2")