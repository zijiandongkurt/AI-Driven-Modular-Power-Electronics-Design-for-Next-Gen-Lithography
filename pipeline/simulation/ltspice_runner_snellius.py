from PyLTSpice import SpiceEditor, RawRead
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import shutil
import re
import json
import pandas as pd
import numpy as np

# Thermal constants
T_MAX    = 125.0
T_AMB    =  40.0
R_TH_JC  =   1.5

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


def _to_float(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip().lower()
    SUFFIXES = [
        ("meg", 1e6), ("g", 1e9),
        ("f", 1e-15), ("p", 1e-12), ("n", 1e-9), ("u", 1e-6),
        ("m", 1e-3),  ("k", 1e3),
    ]
    for suffix, mult in SUFFIXES:
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


class LTSpiceSimulator():
    def __init__(self):
        self.BASE_DIR   = Path(__file__).parent
        self.DATA_DIR   = self.BASE_DIR.parent / "data"

        # Container paths — adjust if your setup differs
        self.HOME_DIR        = Path.home()
        self.LTSPICE_FILES   = self.HOME_DIR / "ltspice-files"   # bind-mounted to /sim in container
        self.RUN_SCRIPT      = self.HOME_DIR / "run_ltspice_snellius.sh"
        self.OVERLAY_DIR     = self.HOME_DIR / "overlays"        # one overlay image per parallel slot
        self.OVERLAY_SIZE_MB = 2048                              # 2 GB per overlay — fits Wine prefix copy
        self.PARALLEL_SIMS   = 4                                 # re-enabled: each slot gets its own overlay

    # -------------------------------------------------------------------------
    # Overlay management (Bram's fix)
    # -------------------------------------------------------------------------

    def _init_overlays(self):
        """Create one reusable overlay image per parallel slot if not already present.

        Uses --no-mount hostfs,tmp so /tmp inside the container is fully isolated
        from the host and from other parallel containers. Each slot gets its own
        overlay so Wine prefix copies never collide.

        RETURNS:
        overlays <list[Path]> : Paths to overlay images, indexed by slot number.
        """
        self.OVERLAY_DIR.mkdir(exist_ok=True)
        overlays = []
        for i in range(self.PARALLEL_SIMS):
            img = self.OVERLAY_DIR / f"overlay_slot{i}.img"
            if not img.exists():
                print(f"Creating overlay slot {i}: {img}")
                subprocess.run(
                    ["apptainer", "overlay", "create", "--size", str(self.OVERLAY_SIZE_MB), str(img)],
                    check=True,
                )
            overlays.append(img)
        return overlays

    def _cleanup_overlays(self, overlays):
        """Remove overlay images after a batch completes to reclaim disk space.

        This prevents accumulated overlays from exhausting scratch disk across
        multiple batches. Overlays are recreated cheaply at the next batch start.
        """
        for img in overlays:
            try:
                img.unlink()
                print(f"Removed overlay: {img.name}")
            except FileNotFoundError:
                pass

    # -------------------------------------------------------------------------
    # Simulation lifecycle
    # -------------------------------------------------------------------------

    def _onSimulationComplete(self, net_stem):
        print(f"SIMULATION COMPLETE: {net_stem}")

    def simulate(self, batchID):
        """Simulates all netlists in data/<batchID>/llm_output/ that passed validation.
        Launches each simulation via the Apptainer container (run_ltspice_snellius.sh),
        now with per-slot overlay isolation so parallel runs no longer collide on
        the shared Wine prefix.

        PARAMS:
        batchID <string> : The ID of a Batch

        RETURNS:
        results <DataFrame> : Refined scalar metrics dataframe, one row per simulation run
        """
        batch_dir        = self.DATA_DIR / batchID
        llm_output_dir   = batch_dir / "LLM_output"
        val_results_path = batch_dir / "validation_results.json"

        assert llm_output_dir.exists(),   f"llm_output folder not found: {llm_output_dir.resolve()}"
        assert val_results_path.exists(), f"validation_results.json not found: {val_results_path.resolve()}"
        assert self.RUN_SCRIPT.exists(),  f"Run script not found: {self.RUN_SCRIPT}"

        # Load validation results — only simulate passing netlists
        val_results = json.loads(val_results_path.read_text())
        valid_stems = {stem for stem, data in val_results.items() if data.get("passed", False)}

        if not valid_stems:
            print(f"WARNING: No valid netlists found for batch '{batchID}'")
            return pd.DataFrame()

        # Prepare directories
        self.LTSPICE_FILES.mkdir(exist_ok=True)
        output_path = self.BASE_DIR / "output" / batchID
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect valid netlists and extract metadata via SpiceEditor
        netlist_map = {}
        net_files_to_run = []

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
                    parts = val.upper().replace("PULSE(", "").replace(")", "").split()
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

            # Copy .net file into ltspice-files/ so the container can access it via /sim
            dest = self.LTSPICE_FILES / netpath.name
            shutil.copy2(netpath, dest)
            net_files_to_run.append(netpath.name)

        # Initialise one overlay image per parallel slot (Bram's fix)
        overlays = self._init_overlays()

        # Assign slot indices to filenames so each thread gets a dedicated overlay
        # and a unique Xvfb display number (:90, :91, ...) to avoid display collisions
        slot_assignments = {
            filename: idx % self.PARALLEL_SIMS
            for idx, filename in enumerate(net_files_to_run)
        }

        def _run_one(filename):
            """Run a single netlist through the container and move .raw output to output_path.

            Passes the per-slot overlay and a unique display number to the shell script
            so that concurrent containers are fully isolated from each other.
            """
            slot           = slot_assignments[filename]
            overlay        = overlays[slot]
            xvfb_display   = f":{90 + slot}"          # :90, :91, :92, :93
            container_path = f"Z:\\\\sim\\\\{filename}"

            result = subprocess.run(
                [str(self.RUN_SCRIPT), container_path, str(overlay), xvfb_display],
                text=True,
            )
            if result.returncode != 0:
                print(f"ERROR [{filename}]: {result.stderr.strip()}")
                return False

            # Move .raw from ltspice-files/ to output/<batchID>/
            raw_src = self.LTSPICE_FILES / filename.replace(".net", ".raw")
            if raw_src.exists():
                shutil.move(str(raw_src), str(output_path / raw_src.name))
                self._onSimulationComplete(Path(filename).stem)
                return True
            else:
                print(f"WARNING: No .raw output found for {filename}")
                return False

        # Run simulations in parallel — one thread per slot, each isolated via its overlay
        ok, total = 0, len(net_files_to_run)
        try:
            with ThreadPoolExecutor(max_workers=self.PARALLEL_SIMS) as executor:
                futures = {executor.submit(_run_one, f): f for f in net_files_to_run}
                for future in as_completed(futures):
                    if future.result():
                        ok += 1
        finally:
            # Always clean up overlays, even if simulations failed
            self._cleanup_overlays(overlays)

        print(f"Batch '{batchID}' done — {ok}/{total} successful")
        if ok < total:
            print(f"WARNING: {total - ok} simulation(s) failed in batch '{batchID}'")

        return self.__getResultsOf(batchID=batchID, netlist_map=netlist_map)

    def __getResultsOf(self, batchID, netlist_map):
        """Extracts scalar performance metrics from .raw simulation outputs."""
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

                if "v(out)" in df_ss:
                    v_out_vals                  = df_ss["v(out)"].values
                    row["voltage_out_mean_V"]   = np.trapezoid(v_out_vals, t) / t_span
                    row["voltage_out_ripple_V"] = v_out_vals.max() - v_out_vals.min()

                if "v(in)" in df_ss:
                    v_in_vals                  = df_ss["v(in)"].values
                    row["voltage_in_mean_V"]   = np.trapezoid(v_in_vals, t) / t_span
                    row["voltage_in_ripple_V"] = v_in_vals.max() - v_in_vals.min()

                if "voltage_out_mean_V" in row and "voltage_in_mean_V" in row and row["voltage_in_mean_V"] != 0:
                    row["conversion_ratio"] = row["voltage_out_mean_V"] / row["voltage_in_mean_V"]

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

                i_load_col = next((c for c in df_ss.columns if "rload" in c or "r_load" in c), None)
                if i_load_col and "voltage_out_mean_V" in row:
                    i_load_vals                = df_ss[i_load_col].values
                    row["load_current_mean_A"] = abs(np.trapezoid(i_load_vals, t) / t_span)
                    row["power_out_W"]         = row["voltage_out_mean_V"] * row["load_current_mean_A"]

                if "v(in)" in df_ss.columns:
                    v_in_vals = df_ss["v(in)"].values
                    id_cols   = [c for c in df_ss.columns if c.startswith("id(m")]
                    if id_cols:
                        i_sw_total = sum(df_ss[c].values for c in id_cols)
                        p_in_inst  = v_in_vals * np.abs(i_sw_total)
                        p_in       = np.trapezoid(p_in_inst, t) / t_span
                        if p_in > 0:
                            row["power_in_W"] = p_in
                            if "power_out_W" in row:
                                row["efficiency"] = row["power_out_W"] / row["power_in_W"]

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

                v_ds_cols = [c for c in df_ss.columns if c == "v(sw)" or re.match(r'v\(sw\d+\)', c) or c == "v(ds)"]
                if v_ds_cols:
                    row["switch_voltage_peak_V"] = max(df_ss[c].abs().max() for c in v_ds_cols)

                i_sw_cols = [c for c in df_ss.columns if c.startswith("id(m")]
                if i_sw_cols:
                    sw_peaks, sw_rms = [], []
                    for col in i_sw_cols:
                        vals = df_ss[col].values
                        sw_peaks.append(np.max(np.abs(vals)))
                        sw_rms.append(float(np.sqrt(np.trapezoid(vals**2, t) / t_span)))
                    row["switch_current_peak_A"] = max(sw_peaks)
                    row["switch_current_rms_A"]  = max(sw_rms)

                if l_values:
                    row["inductor_volume_cm3"] = K_L * sum(l_values)

                if "inductor_volume_cm3" in row and "heatsink_volume_cm3" in row:
                    row["total_volume_cm3"] = V_FIXED + row["inductor_volume_cm3"] + row["heatsink_volume_cm3"]

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