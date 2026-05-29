"""
simulation_server.py
────────────────────
Runs on Snellius. Replaces ltspice_runner_snellius.py.

Creates a job file for the PC client, waits for simulation_results.csv
to be uploaded back, then returns the results to the training loop.

Usage (same interface as original LTSpiceSimulator):
    from simulation_server import SimulationServer
    server = SimulationServer()
    results = server.simulate(batchID)
"""

import json
import time
import pandas as pd
from pathlib import Path

POLL_INTERVAL_S = 3
JOB_TIMEOUT_S   = 3600   # 1 hour max wait per batch


class SimulationServer:
    """Drop-in replacement for LTSpiceSimulator on Snellius."""

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.REPO_DIR = self.BASE_DIR.parent.parent.parent
        self.DATA_DIR = self.REPO_DIR / "pipeline" / "data"
        self.JOBS_DIR = self.BASE_DIR / "jobs"
        self.JOBS_DIR.mkdir(exist_ok=True)

    def simulate(self, batchID: str) -> pd.DataFrame:
        """Submit a batch to the PC client and wait for results."""
        batch_dir        = self.DATA_DIR / batchID
        llm_output_dir   = batch_dir / "LLM_output"
        val_results_path = batch_dir / "validation_results.json"
        results_path     = batch_dir / "simulation_results.csv"

        assert llm_output_dir.exists(),   f"LLM_output not found: {llm_output_dir}"
        assert val_results_path.exists(), f"validation_results.json not found: {val_results_path}"

        val_results = json.loads(val_results_path.read_text())
        valid_stems = {stem for stem, data in val_results.items() if data.get("passed", False)}

        if not valid_stems:
            print(f"WARNING: No valid netlists found for batch '{batchID}'")
            return pd.DataFrame()

        net_files = [
            netpath.name
            for netpath in llm_output_dir.glob("*.net")
            if netpath.stem in valid_stems
        ]

        if not net_files:
            print(f"WARNING: No .net files found for valid stems in batch '{batchID}'")
            return pd.DataFrame()

        print(f"[{batchID}] Submitting {len(net_files)} netlists to simulation client...")

        job = {
            "batch_id":    batchID,
            "net_files":   net_files,
            "net_dir":     str(llm_output_dir),
            "results_dir": str(batch_dir),
            "status":      "pending",
        }
        job_filename = batchID.replace("/", "_") + ".json"
        job_file     = self.JOBS_DIR / job_filename
        job_file.write_text(json.dumps(job, indent=2))
        print(f"[{batchID}] Job written: {job_file}")

        # Wait for results — track wall-clock time
        print(f"[{batchID}] Waiting for simulation_results.csv...")
        elapsed = 0
        t_start = time.time()

        while elapsed < JOB_TIMEOUT_S:
            current_job = json.loads(job_file.read_text())
            if current_job.get("status") == "done" and results_path.exists():
                sim_time = time.time() - t_start
                print(f"[{batchID}] Results received in {sim_time:.1f}s")
                break

            time.sleep(POLL_INTERVAL_S)
            elapsed += POLL_INTERVAL_S

            if elapsed % 30 == 0:
                print(f"[{batchID}] Still waiting... ({elapsed}s elapsed)")
        else:
            raise TimeoutError(f"Batch '{batchID}' timed out after {JOB_TIMEOUT_S}s")

        job_file.unlink()
        print(f"[{batchID}] Job file removed.")

        return self._load_results(results_path)

    def _load_results(self, results_path: Path) -> pd.DataFrame:
        """Load simulation_results.csv into a DataFrame."""
        df = pd.read_csv(results_path)
        print(f"Loaded {len(df)} result(s) from {results_path}")
        return df