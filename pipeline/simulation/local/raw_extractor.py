from PyLTSpice import RawRead
from pathlib import Path
import re
import numpy as np

# Thermal constants
T_MAX   = 125.0
T_AMB   =  40.0
R_TH_JC =   1.5

# Volume model constants (Gashi et al.)
K_L     = 4.0e4
K_H     = 40.0
V_FIXED = 20.0

COMPONENT_WEIGHTS = {
    "M": 3.0,
    "D": 1.5,
    "L": 2.5,
    "C": 1.0,
    "R": 0.2,
}


class RawExtractor:
    """Extracts scalar performance metrics from .raw simulation outputs.

    Takes a directory of .raw files and a netlist_map (pre-extracted metadata
    from SpiceEditor), and produces a simulation_results.json file.

    After extraction, all .raw files are deleted to keep disk usage clean.
    """

    def __init__(self, output_dir: Path):
        """Initialize the extractor.

        Args:
            output_dir (Path): Directory containing .raw files for one batch.
        """
        self.output_dir = Path(output_dir)

    def extract(self, netlist_map: dict, results_path: Path) -> list:
        """Extract metrics from all .raw files in output_dir.

        Args:
            netlist_map (dict): Maps netlist stem to metadata from SpiceEditor
                with keys ``counts``, ``l_values``, and ``switching_freq_Hz``.
            results_path (Path): Where to write simulation_results.csv.

        Returns:
            list: List of dicts, one per simulation step, each containing
                scalar performance metrics.
        """
        assert self.output_dir.exists(), f"Output dir not found: {self.output_dir}"

        raw_files = list(self.output_dir.glob("*.raw"))
        assert raw_files, f"No .raw files found in: {self.output_dir}"

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
                df = df.reset_index()

                if "time" not in df.columns or df.empty:
                    print(f"WARNING: No time column in {raw_path.name} step {step}, skipping")
                    continue

                t_max = df["time"].max()
                df_ss = df[df["time"] >= t_max * 0.8].copy()
                t      = df_ss["time"].values
                t_span = t[-1] - t[0]

                if t_span == 0:
                    print(f"WARNING: Zero time span in {raw_path.name} step {step}, skipping")
                    continue

                row = {"source_file": raw_path.stem, "step": step}

                #  Output voltage 
                if "v(out)" in df_ss:
                    v_out_vals                  = df_ss["v(out)"].values
                    row["voltage_out_mean_V"]   = np.trapezoid(v_out_vals, t) / t_span
                    row["voltage_out_ripple_V"] = v_out_vals.max() - v_out_vals.min()

                #  Input voltage 
                if "v(in)" in df_ss:
                    v_in_vals                  = df_ss["v(in)"].values
                    row["voltage_in_mean_V"]   = np.trapezoid(v_in_vals, t) / t_span
                    row["voltage_in_ripple_V"] = v_in_vals.max() - v_in_vals.min()

                #  Conversion ratio 
                if "voltage_out_mean_V" in row and "voltage_in_mean_V" in row and row["voltage_in_mean_V"] != 0:
                    row["conversion_ratio"] = row["voltage_out_mean_V"] / row["voltage_in_mean_V"]

                #  Inductor current 
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

                #  Switching frequency 
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

                #  Output power 
                i_load_col = next((c for c in df_ss.columns if "rload" in c or "r_load" in c), None)
                if i_load_col and "voltage_out_mean_V" in row:
                    i_load_vals                = df_ss[i_load_col].values
                    row["load_current_mean_A"] = abs(np.trapezoid(i_load_vals, t) / t_span)
                    row["power_out_W"]         = row["voltage_out_mean_V"] * row["load_current_mean_A"]

                #  Input power & efficiency 
                if "v(in)" in df_ss.columns:
                    v_in_vals = df_ss["v(in)"].values
                    i_in_cols = [c for c in df_ss.columns if c.startswith("i(v") and "in" in c]
                    if i_in_cols:
                        i_in_vals = np.abs(df_ss[i_in_cols[0]].values)
                        p_in      = np.trapezoid(v_in_vals * i_in_vals, t) / t_span
                        if p_in > 0:
                            row["power_in_W"] = p_in
                            if "power_out_W" in row:
                                row["efficiency"] = row["power_out_W"] / row["power_in_W"]

                #  Losses & thermal 
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

                #  Switch voltage stress 
                v_ds_cols = [c for c in df_ss.columns if c == "v(sw)" or re.match(r'v\(sw\d+\)', c) or c == "v(ds)"]
                if v_ds_cols:
                    row["switch_voltage_peak_V"] = max(df_ss[c].abs().max() for c in v_ds_cols)

                #  Switch current stress 
                i_sw_cols = [c for c in df_ss.columns if c.startswith("id(m")]
                if i_sw_cols:
                    sw_peaks, sw_rms = [], []
                    for col in i_sw_cols:
                        vals = df_ss[col].values
                        sw_peaks.append(np.max(np.abs(vals)))
                        sw_rms.append(float(np.sqrt(np.trapezoid(vals**2, t) / t_span)))
                    row["switch_current_peak_A"] = max(sw_peaks)
                    row["switch_current_rms_A"]  = max(sw_rms)

                #  Inductor volume surrogate 
                if l_values:
                    row["inductor_volume_cm3"] = K_L * sum(l_values)

                #  Total converter volume 
                if "inductor_volume_cm3" in row and "heatsink_volume_cm3" in row:
                    row["total_volume_cm3"] = V_FIXED + row["inductor_volume_cm3"] + row["heatsink_volume_cm3"]

                #  Component counts 
                row["count_mosfets"]    = counts.get("M", 0)
                row["count_diodes"]     = counts.get("D", 0)
                row["count_inductors"]  = counts.get("L", 0)
                row["count_capacitors"] = counts.get("C", 0)

                all_rows.append(row)

        # Write results CSV
        import pandas as pd
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(results_path, index=False, lineterminator='\n')
        print(f"Saved {len(all_rows)} result(s) -> {results_path}")

        # Clean up .raw files
        for raw_path in raw_files:
            try:
                raw_path.unlink()
            except Exception as e:
                print(f"WARNING: Could not delete {raw_path.name} — {e}")
        print(f"Cleaned up {len(raw_files)} .raw file(s) from {self.output_dir}")

        return all_rows