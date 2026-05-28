from PyLTSpice import LTspice, SimRunner, SpiceEditor
from pathlib import Path
import json


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
    try:
        return float(s)
    except ValueError:
        print(f"[!] Warning: Could not convert '{s}' to float. Defaulting to 0.0")
        return 0.0


COMPONENT_WEIGHTS = {
    "M": 3.0,
    "D": 1.5,
    "L": 2.5,
    "C": 1.0,
    "R": 0.2,
}


class LTSpiceSimulator:
    """Runs LTspice simulations on valid netlists for a given batch.

    Produces .raw files in output/<batchID>/ and returns netlist_map
    for downstream processing by RawExtractor.
    """

    def __init__(self, output_dir: Path):
        """
        PARAMS:
        output_dir <Path> : Where .raw files will be written
        """
        self.output_dir = Path(output_dir)

    def _onSimulationComplete(self, raw_file, log_file):
        """Callback fired when a simulation completes."""
        print(f"SIMULATION COMPLETE: {raw_file}")
        raw_path = Path(raw_file)
        for suffix in [".net", ".db", ".op.raw"]:
            unwanted = raw_path.with_suffix(suffix) if not suffix.startswith(".op") \
                else raw_path.with_name(raw_path.stem + suffix)
            if unwanted.exists():
                unwanted.unlink()

    def simulate(self, net_paths: list) -> dict:
        """Run LTspice on a list of .net files.

        PARAMS:
        net_paths <list[Path]> : Local paths to .net files to simulate

        RETURNS:
        netlist_map <dict> : Maps netlist stem -> metadata (counts, l_values, switching_freq_Hz)
                             Passed to RawExtractor.extract()
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        runner = SimRunner(
            output_folder=self.output_dir,
            simulator=LTspice,
            parallel_sims=4,
        )

        netlist_map = {}

        for netpath in net_paths:
            netpath = Path(netpath)
            net     = SpiceEditor(netpath)

            # Extract metadata
            l_refs   = net.get_components("L")
            l_values = [_to_float(net.get_component_value(l)) for l in l_refs]
            counts   = {k: len(net.get_components(k)) for k in COMPONENT_WEIGHTS}

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

            runner.run(net,
                       run_filename=netpath.name,
                       callback=self._onSimulationComplete)

        runner.wait_completion()
        print(f"Simulations done — {runner.okSim}/{runner.runno} successful")

        if runner.okSim < runner.runno:
            print(f"WARNING: {runner.runno - runner.okSim} simulation(s) failed")

        return netlist_map