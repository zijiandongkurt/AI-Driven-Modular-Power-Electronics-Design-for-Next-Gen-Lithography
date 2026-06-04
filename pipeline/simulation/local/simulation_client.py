"""
simulation_client.py
────────────────────
Runs on your PC. Polls Snellius for pending simulation jobs, runs LTspice
natively via LTSpiceSimulator, extracts metrics via RawExtractor, and sends
simulation_results.csv back to Snellius.

Usage:
    .venv\Scripts\python pipeline\simulation\local\simulation_client.py

Keep this running in the background during training on Snellius.
All paths are derived relative to the repo root via config.json.
"""

import json
import time
import shutil
import paramiko
from pathlib import Path

from ltspice_runner import LTSpiceSimulator
from raw_extractor import RawExtractor

# ── Load config ───────────────────────────────────────────────────────────────
# simulation/local/ → simulation/ → pipeline/ → repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG    = json.loads((REPO_ROOT / "simulation_config.json").read_text())
_profile  = CONFIG.get("active_profile")
if _profile and _profile in CONFIG.get("profiles", {}):
    CONFIG.update(CONFIG["profiles"][_profile])

SNELLIUS_HOST   = CONFIG["snellius_host"]
SNELLIUS_USER   = CONFIG["snellius_user"]
SSH_KEY_PATH    = Path(CONFIG["ssh_key_path"]).expanduser()
SNELLIUS_REPO   = f"/home/{SNELLIUS_USER}/{CONFIG['snellius_repo']}"
SNELLIUS_JOBS   = f"{SNELLIUS_REPO}/pipeline/simulation/snellius/jobs"
LOCAL_WORK_DIR  = REPO_ROOT / "pipeline" / "simulation" / "output"
POLL_INTERVAL_S = CONFIG.get("poll_interval_s", 3)

# ── SSH/SFTP helpers ──────────────────────────────────────────────────────────

def _connect():
    """Open SSH + SFTP connection to Snellius."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SNELLIUS_HOST, username=SNELLIUS_USER, key_filename=str(SSH_KEY_PATH))
    return ssh, ssh.open_sftp()


def _list_pending_jobs(sftp) -> list:
    try:
        return [f for f in sftp.listdir(SNELLIUS_JOBS) if f.endswith(".json")]
    except FileNotFoundError:
        return []


def _read_job(sftp, job_file: str) -> dict:
    with sftp.open(f"{SNELLIUS_JOBS}/{job_file}", "r") as f:
        return json.load(f)


def _download_nets(sftp, job: dict, local_dir: Path) -> list:
    """Download all .net files for this job to local_dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for filename in job["net_files"]:
        remote = f"{job['net_dir']}/{filename}"
        local  = local_dir / filename
        sftp.get(remote, str(local))
        print(f"  Downloaded: {filename}")
        local_paths.append(local)
    return local_paths


def _upload_results(sftp, local_csv: Path, remote_results_dir: str):
    """Upload simulation_results.csv to Snellius data/<batchID>/."""
    remote_path = f"{remote_results_dir}/simulation_results.csv"
    sftp.put(str(local_csv), remote_path)
    print(f"  Uploaded results -> {remote_path}")


def _mark_job_done(sftp, job_file: str, job: dict):
    """Set job status to done on Snellius."""
    job["status"] = "done"
    with sftp.open(f"{SNELLIUS_JOBS}/{job_file}", "w") as f:
        json.dump(job, f, indent=2)
    print(f"  Job marked done: {job_file}")


def _cleanup_local(local_dir: Path):
    if local_dir.exists():
        shutil.rmtree(local_dir)
        print(f"  Cleaned up: {local_dir}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    print(f"Simulation client started. Polling every {POLL_INTERVAL_S}s...")
    print(f"  Repo:    {REPO_ROOT}")
    print(f"  Host:    {SNELLIUS_HOST}")
    print(f"  Jobs:    {SNELLIUS_JOBS}\n")

    while True:
        try:
            ssh, sftp = _connect()

            for job_file in _list_pending_jobs(sftp):
                job = _read_job(sftp, job_file)
                if job.get("status") != "pending":
                    continue

                batch_id = job["batch_id"]
                print(f"\n[{batch_id}] Job received: {len(job['net_files'])} netlists")

                local_net_dir = LOCAL_WORK_DIR / batch_id / "nets"
                local_raw_dir = LOCAL_WORK_DIR / batch_id / "raw"
                local_results = LOCAL_WORK_DIR / batch_id / "simulation_results.csv"

                # 1. Download .net files from Snellius
                print(f"[{batch_id}] Downloading netlists...")
                net_paths = _download_nets(sftp, job, local_net_dir)

                # 2. Run LTspice natively → .raw files in local_raw_dir
                print(f"[{batch_id}] Running simulations...")
                simulator   = LTSpiceSimulator(output_dir=local_raw_dir)
                netlist_map = simulator.simulate(net_paths)

                # 3. Extract metrics → simulation_results.csv
                print(f"[{batch_id}] Extracting metrics...")
                try:
                    extractor = RawExtractor(local_raw_dir)
                    extractor.extract(netlist_map, local_results)
                except AssertionError as err:
                    print(f"[{batch_id}] Extractor: {err}")

                # If all simulations failed no CSV is produced — create an empty
                # placeholder so upload and mark-done always run.
                if not local_results.exists():
                    local_results.parent.mkdir(parents=True, exist_ok=True)
                    local_results.touch()
                    print(f"[{batch_id}] All simulations failed — uploading empty results.")

                # 4. Upload simulation_results.csv to Snellius
                print(f"[{batch_id}] Uploading results...")
                _upload_results(sftp, local_results, job["results_dir"])

                # 5. Mark job done
                _mark_job_done(sftp, job_file, job)

                # 6. Clean up local work dir
                _cleanup_local(LOCAL_WORK_DIR / batch_id)

                print(f"[{batch_id}] Complete.")

            sftp.close()
            ssh.close()

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    run()