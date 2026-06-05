import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.netlist_validation.validator import validator
from pipeline.simulation.snellius.simulation_server import SimulationServer as LTSpiceSimulator
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm
from pipeline.llm_topology_generation.prompt_input import load_constraint
from pipeline.reinforcement_algorithm.grpo_trainer import GRPOTrainer
from pipeline.reinforcement_algorithm.new_rl_updater import RLConfig
from pipeline.graphs_and_visualizations.Visualize_demo_results import plot_run_results
from pipeline.graphs_and_visualizations.plot_probabilities import plot_softmax_probabilities
from pipeline.graphs_and_visualizations.plot_cumulative_probabilities import plot_cumulative_probabilities
from pipeline.utility.summary_logger import SummaryLogger
from pipeline.utility.topology_hasher import get_topological_hash

# --- NEW: Import your net_writer ---
from pipeline.llm_topology_generation.net_writer import write_netlists

def get_next_zycos_folder(data_dir: Path) -> str:
    """Find next zycos_XXX folder."""
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    folders = [
        d.name for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"zycos_\d+", d.name)
    ]

    if not folders:
        return "zycos_001"

    numbers = [int(f.split("_")[1]) for f in folders]
    return f"zycos_{max(numbers) + 1:03d}"


def load_reward_data(batch_id: str) -> Dict:
    """Load reward_results.json for a batch."""
    path = Path("pipeline") / "data" / batch_id / "reward_results.json"
    
    # 🛡️ SAFETY NET 1: Missing File Trap
    if not path.exists():
        print(f"⚠️ Missing reward file: {path}. Treating batch as empty/failed.")
        return {"circuits": {}}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error reading reward file {path}: {e}")
        return {"circuits": {}}


def parse_group_id(netlist_id: str) -> Optional[str]:
    """Extract group id from names like xxx_g1_cand3_b2."""
    match = re.search(r"_(g\d+)_cand\d+", netlist_id)
    if match:
        return match.group(1)
    return None


def add_batch_to_history(
    history: List[Dict],
    batch_id: str,
    group_to_parent: Dict[str, Optional[Dict]],
    default_depth: int,
) -> None:
    """Add all evaluated netlists in current batch to history database."""
    reward_data = load_reward_data(batch_id)
    circuits = reward_data.get("circuits", {})

    for netlist_id, info in circuits.items():
        fitness = info.get("fitness_score", info.get("grpo_reward", None))
        if fitness is None:
            continue

        # --- Calculate Topological Hash ---
        net_path = Path("pipeline") / "data" / batch_id / "LLM_output" / f"{netlist_id}.net"
        topo_hash = "unknown"
        if net_path.exists():
            net_text = net_path.read_text(encoding="utf-8")
            try:
                topo_hash = get_topological_hash(net_text)
            except (ValueError, Exception):
                topo_hash = f"invalid_{netlist_id}"

        group_id = parse_group_id(netlist_id)
        parent = group_to_parent.get(group_id)

        if parent is None:
            depth = default_depth
            parent_id = None
        else:
            depth = int(parent["depth"]) + 1
            parent_id = parent["netlist_id"]

        history.append({
            "netlist_id": netlist_id,
            "batch_id": batch_id,
            "group_id": group_id,
            "parent_id": parent_id,
            "fitness": float(fitness),
            "depth": depth,
            "topo_hash": topo_hash
        })


def save_history(history: List[Dict], run_folder_path: Path) -> None:
    """Save historical state database."""
    path = run_folder_path / "history_db.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def generate_global_training_summary(zycos_path: Path, current_run_idx: int, total_runs: int, config: Dict = None):
    """
    Scans all completed runs in the current zycos folder and generates a live-updating
    master summary file tracking global progress, aggregate yields, and the best overall fitness.
    """
    summary_files = list(zycos_path.glob("Run_*/results/run_summary.txt"))
    if not summary_files:
        return

    total_time = 0.0
    total_valid = 0
    total_invalid = 0
    total_netlists = 0
    total_batches = 0
    
    global_best_fit = -float('inf')
    global_best_cand = "None"
    global_best_run = "None"
    global_best_constraint = "None"

    for sf in summary_files:
        text = sf.read_text(encoding="utf-8")
        run_name = sf.parent.parent.name
        
        m_time = re.search(r"Total Time Elapsed:\s*([\d.]+)", text)
        if m_time: total_time += float(m_time.group(1))
        
        m_batch = re.search(r"Batches Completed:\s*(\d+)", text)
        if m_batch: total_batches += int(m_batch.group(1))
        
        m_valid = re.search(r"Valid:\s*(\d+)", text)
        if m_valid: total_valid += int(m_valid.group(1))
        
        m_invalid = re.search(r"Invalid:\s*(\d+)", text)
        if m_invalid: total_invalid += int(m_invalid.group(1))
        
        m_hist = re.search(r"Total candidates in history:\s*(\d+)", text)
        if m_hist: total_netlists += int(m_hist.group(1))
        
        m_fit = re.search(r"Overall Best Fitness:\s*([\d.-]+)\s*\(([^)]+)\)", text)
        m_constraint = re.search(r"Constraint Index:\s*(\d+)", text)
        
        if m_fit:
            fit_val = float(m_fit.group(1))
            if fit_val > global_best_fit:
                global_best_fit = fit_val
                global_best_cand = m_fit.group(2)
                global_best_run = run_name
                global_best_constraint = m_constraint.group(1) if m_constraint else "Unknown"

    avg_time_per_run = total_time / len(summary_files) if summary_files else 0
    avg_time_per_batch = total_time / total_batches if total_batches > 0 else 0
    avg_time_per_netlist = total_time / total_netlists if total_netlists > 0 else 0
    global_validity = (total_valid / total_netlists * 100) if total_netlists > 0 else 0
    runs_remaining = total_runs - current_run_idx

    config_text = ""
    if config:
        config_text = "\n--- TRAINING HYPERPARAMETERS ---\n"
        config_text += json.dumps(config, indent=2)
        config_text += "\n"

    master_text = f"""=== GLOBAL TRAINING RUN SUMMARY: {zycos_path.name} ===
Progress: {current_run_idx} / {total_runs} Runs Completed ({runs_remaining} remaining)

--- AGGREGATE TIME METRICS ---
Total Time Elapsed: {total_time:.2f} sec ({total_time / 3600:.2f} hours)
Average Time per Run: {avg_time_per_run:.2f} sec
Average Time per Batch: {avg_time_per_batch:.2f} sec
Average Time per Netlist: {avg_time_per_netlist:.2f} sec

--- AGGREGATE TOPOLOGY YIELD ---
Total Candidates Generated: {total_netlists}
Total Valid: {total_valid}
Total Invalid: {total_invalid}
Global Validity Rate: {global_validity:.2f}%

--- GLOBAL BEST FITNESS ---
Best Overall Score: {global_best_fit:.4f}
Best Candidate: {global_best_cand}
Found in: {global_best_run} (Constraint: {global_best_constraint})
{config_text}==================================================
"""
    out_path = zycos_path / "training_run_summary.txt"
    out_path.write_text(master_text, encoding="utf-8")

def _softmax_sample_from_pool(
    pool: List[Dict],
    temperature: float,
    rng: random.Random,
) -> Dict:
    """Sample one item from a pool using temperature-scaled softmax over fitness."""
    scores = [float(item["fitness"]) for item in pool]
    max_score = max(scores)

    exp_scores = [
        math.exp((score - max_score) / max(temperature, 1e-8))
        for score in scores
    ]

    total = sum(exp_scores)
    r = rng.random() * total

    acc = 0.0
    for item, weight in zip(pool, exp_scores):
        acc += weight
        if acc >= r:
            return item

    return pool[-1]


def epsilon_greedy_topk_sample_parents(
    history: List[Dict],
    k: int,
    top_k: int,
    epsilon: float,
    temperature: float,
    seed: int = 42,
) -> List[Dict]:
    """
    Select parents using epsilon-greedy Top-K softmax.
    Includes a fallback to sample with replacement from unique topologies 
    if we run out of unique candidates.
    """
    if not history:
        raise RuntimeError("History is completely empty. Cannot sample parents.")

    rng = random.Random(seed)

    # --- Filter out Topological Duplicates ---
    unique_history = []
    seen_hashes = set()
    for item in history:
        thash = item.get("topo_hash", item["netlist_id"]) # Fallback if hash missing
        if thash not in seen_hashes:
            seen_hashes.add(thash)
            unique_history.append(item)

    sorted_history = sorted(
        unique_history,
        key=lambda item: float(item["fitness"]),
        reverse=True,
    )
    
    # top-k netlists
    top_k_pool = sorted_history[:max(1, min(top_k, len(sorted_history)))]
    long_tail_pool = sorted_history[len(top_k_pool):]

    selected: List[Dict] = []
    selected_ids = set()

    for _ in range(k):
        available_top = [
            item for item in top_k_pool
            if item["netlist_id"] not in selected_ids
        ]

        available_tail = [
            item for item in long_tail_pool
            if item["netlist_id"] not in selected_ids
        ]
        
        # --- THE FIX: Sample from Unique History with Replacement ---
        if not available_top and not available_tail:
            # We have exhausted all unique topologies!
            # Fallback: Pick randomly from the unique champions we already found.
            chosen = rng.choice(sorted_history)
            selection_mode = "fallback_duplicate_unique"
        # ----------------------------------------------------------
        else:
            use_exploration = (
                bool(available_tail)
                and rng.random() < epsilon
            )

            if use_exploration:
                chosen = rng.choice(available_tail)
                selection_mode = "epsilon_random_long_tail"
            else:
                candidate_pool = available_top if available_top else available_tail
                chosen = _softmax_sample_from_pool(
                    pool=candidate_pool,
                    temperature=temperature,
                    rng=rng,
                )
                selection_mode = "topk_softmax"

        chosen = dict(chosen)
        chosen["selection_mode"] = selection_mode

        selected.append(chosen)
        selected_ids.add(chosen["netlist_id"])

    return selected


def run_eval_pipeline(
    batch_id: str,
    val,
    simulator,
    reward_fn,
    constraint: Dict,
    weights: Dict,
) -> None:
    """Run validate -> simulate -> reward for one batch."""
    print("Validating netlists...")
    val.validate(batch_id)

    print("Running simulations...")
    simulator.simulate(batch_id)

    print("Evaluating fitness and formatting JSON...")
    reward_fn.process_batch(batch_id, constraint, weights=weights)


def run_single(
    run_idx: int,          
    local_run_idx: int,    
    zycos_name: str,
    llm,
    val,
    simulator,
    reward_fn,
    grpo,
    data_dir: Path,
    config: Dict,
) -> None:
    """Run the full training loop for a single constraint."""

    N_batch = config["n_batch"]
    SEED_PROMPTS = config["seed_prompts"]
    PARENTS_PER_BATCH = config["parents_per_batch"]
    OUTPUTS_PER_PARENT = config["outputs_per_parent"]
    SOFTMAX_TEMPERATURE = config["softmax_temperature"]
    RANDOM_SEED = config["random_seed"]
    TOP_K = config.get("top_k", 8)
    EPSILON = config.get("epsilon", 0.15)
    weights = config["weights"]

    constraint = load_constraint(config["constraint_path"], idx=run_idx)

    run_folder_name = f"Run_{local_run_idx + 1:03d}"
    run_folder_path = data_dir / zycos_name / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(
        f"=== {zycos_name} | Run {local_run_idx + 1}/{len(config.get('constraint_indices', [])) or config['n_runs']} | constraint idx={run_idx} ==="
    )
    print(f"{'=' * 60}")

    logger = SummaryLogger(
        run_folder_path=run_folder_path,
        n_batches=N_batch,
        sim_params={},
        weights=weights,
        constraint_idx=run_idx,
    )

    history: List[Dict] = []
    selected_parents: List[Dict] = []

    for batch_idx in range(1, N_batch + 1):
        batch_start = time.time()
        current_batch_id = f"{zycos_name}/{run_folder_name}/batch_{batch_idx}"
        print(f"\n--- Processing {current_batch_id} ---")

        # 🛡️ SAFETY NET 2: LLM Generation PyTorch/OOM Error Trap
        max_retries = 3
        generation_success = False
        written = None
        group_to_parent = {}
        
        for attempt in range(max_retries):
            try:
                if batch_idx == 1:
                    seed_parent_ids = [
                        f"seed_prompt_{i}"
                        for i in range(1, SEED_PROMPTS + 1)
                    ]

                    print("Generating seed groups...")
                    written = llm.generate_grouped_for_batch(
                        constraint=constraint,
                        batchID=current_batch_id,
                        parent_ids=seed_parent_ids,
                        previous_batch_id=None,
                        outputs_per_parent=OUTPUTS_PER_PARENT,
                        DEMO=False,
                    )

                    group_to_parent = {
                        f"g{i}": None
                        for i in range(1, SEED_PROMPTS + 1)
                    }
                else:
                    print("Generating children from selected parents...")
                    if hasattr(llm, "generate_grouped_for_parent_entries"):
                        written = llm.generate_grouped_for_parent_entries(
                            constraint=constraint,
                            batchID=current_batch_id,
                            parent_entries=selected_parents,
                            outputs_per_parent=OUTPUTS_PER_PARENT,
                            DEMO=True,
                        )
                    else:
                        parent_batches = {p["batch_id"] for p in selected_parents}
                        if len(parent_batches) != 1:
                            raise RuntimeError("Selected parents come from different batches.")
                        previous_batch_id = selected_parents[0]["batch_id"]

                        written = llm.generate_grouped_for_batch(
                            constraint=constraint,
                            batchID=current_batch_id,
                            parent_ids=[p["netlist_id"] for p in selected_parents],
                            previous_batch_id=previous_batch_id,
                            outputs_per_parent=OUTPUTS_PER_PARENT,
                            DEMO=True,
                        )

                    group_to_parent = {
                        f"g{i}": parent
                        for i, parent in enumerate(selected_parents, start=1)
                    }
                
                generation_success = True
                break  # Exit retry loop on success

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ CUDA OOM during LLM generation (Attempt {attempt + 1}/{max_retries}). Clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"⚠️ PyTorch error during LLM Generation: {e}")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ LLM Generation failed (Attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(5)
                
        if not generation_success:
            print(f"❌ LLM failed to generate after {max_retries} attempts. Skipping batch {batch_idx}.")
            continue # Allows the loop to skip to the next batch instead of crashing

        print(f"Generated {len(written) if written else 0} netlists.")

        # --- Extract from raw_output.txt and use net_writer to save .net files ---
        candidate_texts = []
        raw_file = Path("pipeline") / "data" / current_batch_id / "raw_output.txt"
        
        # Pull text from the file LLM_API created
        if raw_file.exists():
            text = raw_file.read_text(encoding="utf-8")
            blocks = re.split(r"={10,}\n\s*CANDIDATE \d+\s*\n={10,}\n", text)
            candidate_texts = [llm.engine._clean(b.strip()) for b in blocks[1:]]
            
        if candidate_texts:
            custom_names = []
            for g_id in group_to_parent.keys():
                for c_idx in range(1, OUTPUTS_PER_PARENT + 1):
                    custom_names.append(f"{g_id}_cand{c_idx}")
                    
            if len(custom_names) == len(candidate_texts):
                label_raw = constraint.get("_comment", f"Topology_{run_idx}")
                clean_label = "00_" + re.sub(r'[^a-zA-Z0-9]', '_', label_raw)
                clean_label = re.sub(r'_+', '_', clean_label).strip('_')
                
                print(f"Saving {len(candidate_texts)} .net files to LLM_output...")
                # 🛡️ SAFETY NET 3: Netlist Writer Exception Trap
                try:
                    write_netlists(
                        netlists=candidate_texts,
                        constraint=constraint,
                        label=clean_label,
                        batchID=current_batch_id,
                        custom_names=custom_names
                    )
                except Exception as e:
                    print(f"⚠️ Error writing netlists for {current_batch_id}: {e}")
            else:
                print(f"⚠️ Mismatch: {len(custom_names)} custom names vs {len(candidate_texts)} texts. Skipping net_writer.")
        else:
            print("⚠️ No candidate texts found in raw_output.txt to save.")
        # -----------------------------------------------------------------------------

        try:
            run_eval_pipeline(
                batch_id=current_batch_id,
                val=val,
                simulator=simulator,
                reward_fn=reward_fn,
                constraint=constraint,
                weights=weights,
            )
        except Exception as e:
            print(f"⚠️ Eval pipeline failed for batch {batch_idx}: {e}. Skipping.")
            continue

        print("Running GRPO RL update...")
        # 🛡️ SAFETY NET 4: GRPO Update PyTorch/OOM Trap
        try:
            grpo.update_from_batch(
                batch_id=current_batch_id,
                max_samples=None,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ CUDA OOM during GRPO update for batch {batch_idx}. Clearing cache and skipping update.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"⚠️ GRPO Update failed with PyTorch error: {e}")
        except Exception as e:
            print(f"⚠️ GRPO Update failed: {e}")

        add_batch_to_history(
            history=history,
            batch_id=current_batch_id,
            group_to_parent=group_to_parent,
            default_depth=1,
        )

        save_history(history, run_folder_path)

        logger.log_batch_training(
            batch_idx=batch_idx,
            batch_duration=time.time() - batch_start,
            history=history,
            batch_id=current_batch_id,
        )

        try:
            run_id = f"{zycos_name}/{run_folder_name}"
            plot_softmax_probabilities(
                run_id=run_id,
                target_batch=batch_idx + 1,
                temperature=SOFTMAX_TEMPERATURE,
                top_k=TOP_K,
                epsilon=EPSILON,
            )
            plot_cumulative_probabilities(
                run_id=run_id,
                target_batch=batch_idx + 1,
                temperature=SOFTMAX_TEMPERATURE,
                top_k=TOP_K,
                epsilon=EPSILON,
            )
        except Exception as e:
            print(f"⚠️ Probability plots failed for batch {batch_idx}: {e}")

        print(f"History size: {len(history)} evaluated netlists")

        if batch_idx < N_batch:
            try:
                selected_parents = epsilon_greedy_topk_sample_parents(
                    history=history,
                    k=PARENTS_PER_BATCH,
                    top_k=TOP_K,
                    epsilon=EPSILON,
                    temperature=SOFTMAX_TEMPERATURE,
                    seed=RANDOM_SEED + batch_idx,
                )
            except RuntimeError as e:
                print(f"⚠️ Parent selection failed: {e}. Using best available from history.")
                sorted_history = sorted(history, key=lambda x: x["fitness"], reverse=True)
                selected_parents = sorted_history[:min(PARENTS_PER_BATCH, len(sorted_history))]

            print("Selected parents for next batch:")
            for p in selected_parents:
                score = p["fitness"]
                mode = p.get("selection_mode", "unknown")
                print(
                    f"  {p['netlist_id']} | "
                    f"fitness={p['fitness']:.4f}, "
                    f"depth={p['depth']}, "
                    f"score={score:.4f}, "
                    f"mode={mode}, "
                    f"batch={p['batch_id']}"
                )

        print(f"--- Finished {current_batch_id} ---")

    print(f"=== Finished {zycos_name}/{run_folder_name} ===")

    print(f"[training_loop] Generating result plots for {run_folder_name}...")
    plot_run_results(str(run_folder_path))


def main():
    config_path = Path("training_config.json")
    assert config_path.exists(), f"training_config.json not found at {config_path.resolve()}"

    with config_path.open("r") as f:
        config = json.load(f)

    N_RUNS = config["n_runs"]

    constraint_indices = config.get("constraint_indices", None)
    if constraint_indices is None:
        rng = random.Random(config["random_seed"])
        with open(config["constraint_path"], "r", encoding="utf-8") as f:
            all_constraints = json.load(f)

        constraint_indices = rng.sample(
            range(len(all_constraints)),
            k=min(N_RUNS, len(all_constraints)),
        )
    config["n_runs"] = len(constraint_indices)
    data_dir = Path("pipeline/data")
    zycos_name = get_next_zycos_folder(data_dir)
    zycos_path = data_dir / zycos_name
    zycos_path.mkdir(parents=True, exist_ok=True)

    sft_lora_path = config.get("sft_lora_path", None)
    llm = TopologyLLM(
        max_new_tokens=config["max_tokens"],
        lora_path=sft_lora_path,
    )

    if sft_lora_path:
        print(f"Loaded SFT LoRA adapter from: {sft_lora_path}")

    val = validator()
    simulator = LTSpiceSimulator()
    reward_fn = RewardFunctionNorm()

    rl = config["rl_config"]
    grpo = GRPOTrainer(
        llm=llm,
        validator=val,
        simulator=simulator,
        reward_fn=reward_fn,
        constraint=None,
        output_dir=f"checkpoints/{zycos_name}/grpo-lora/final",
        rl_config=RLConfig(
            max_length=rl["max_length"],
            max_prompt_length=rl["max_prompt_length"],
            max_completion_length=rl["max_completion_length"],
            learning_rate=rl["learning_rate"],
            save_every=rl["save_every"],
            lora_r=rl["lora_r"],
            lora_alpha=rl["lora_alpha"],
        ),
    )

    print(f"=== Starting {zycos_name} | {len(constraint_indices)} runs ===")
    print(f"Selected constraint indices: {constraint_indices}")

    for local_run_idx, constraint_idx in enumerate(constraint_indices):
        run_single(
            run_idx=constraint_idx,
            local_run_idx=local_run_idx,
            zycos_name=zycos_name,
            llm=llm,
            val=val,
            simulator=simulator,
            reward_fn=reward_fn,
            grpo=grpo,
            data_dir=data_dir,
            config=config,
        )
        generate_global_training_summary(zycos_path, local_run_idx + 1, len(constraint_indices), config)

    print(f"\n=== {zycos_name} complete — {len(constraint_indices)} runs finished ===")

if __name__ == "__main__":
    main()