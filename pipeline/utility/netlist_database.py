import json
import numpy as np
from pathlib import Path

from pipeline.utility.topology_hasher import get_topological_hash

class NetlistDatabase:
    """Track generated netlists and their fitness scores for MCTS state sampling.

    Uses Epsilon-Greedy Top-K Softmax to balance exploitation of high-fitness
    topologies with exploration of lower-ranked candidates.
    """

    def __init__(self, temperature=0.05):
        self.records = {}
        self.temperature = temperature

        self.total_valid   = 0
        self.total_invalid = 0

        self.seen_hashes = set()
        self.total_unique = 0
        self.total_duplicates = 0

    def add_record(
        self,
        candidate_id,
        netlist_text,
        fitness,
        feedback_text,
        is_valid=False,
        metrics=None,
        batch_id=None,
    ):
        """Store a candidate netlist and update global validity and uniqueness counters.

        Args:
            candidate_id (str): Unique identifier for this netlist.
            netlist_text (str): Raw SPICE netlist content.
            fitness (float): Fitness score from the reward function.
            feedback_text (str): Human-readable feedback string for MCTS prompting.
            is_valid (bool): Whether the netlist passed all validation checks.
            metrics (dict | None): Raw simulation and reward metrics dict.
            batch_id (str | None): Batch the candidate was generated in.
        """
        topo_hash = get_topological_hash(netlist_text)
        if topo_hash not in self.seen_hashes:
            self.seen_hashes.add(topo_hash)
            self.total_unique += 1
        else:
            self.total_duplicates += 1

        self.records[candidate_id] = {
            "netlist_text": netlist_text,
            "fitness":      fitness,
            "feedback":     feedback_text,
            "is_valid":     is_valid,
            "metrics":      metrics or {},
            "batch_id":     batch_id,
            "topo_hash":    topo_hash
        }

        if is_valid:
            self.total_valid += 1
        else:
            self.total_invalid += 1

    def ingest_batch_data(self, batch_id: str, database_dir: Path):
        """Parse batch results from disk and push all candidates into the database.

        Also saves a physical copy of each netlist to the global database folder.

        Args:
            batch_id (str): Batch identifier to load (relative to
                ``pipeline/data/``).
            database_dir (Path): Directory where .net file copies are written.
        """
        data_dir    = Path("pipeline/data") / batch_id
        reward_path = data_dir / "reward_results.json"
        val_path    = data_dir / "validation_results.json"
        llm_out_dir = data_dir / "LLM_output"

        if not reward_path.exists():
            return

        with open(reward_path, "r", encoding="utf-8") as f:
            rewards = json.load(f).get("circuits", {})

        validations = {}
        if val_path.exists():
            with open(val_path, "r", encoding="utf-8") as f:
                validations = json.load(f)

        for cand_id, metrics in rewards.items():
            net_file = llm_out_dir / f"{cand_id}.net"
            if not net_file.exists():
                continue
            netlist_text = net_file.read_text(encoding="utf-8")

            # Save copy to the central database folder
            db_file = database_dir / f"{cand_id}.net"
            db_file.write_text(netlist_text, encoding="utf-8")

            fitness  = metrics.get("fitness_score", -1.0)
            is_valid = validations.get(cand_id, {}).get("passed", False)

            feedback_text = f"Fitness Score: {fitness:.4f}\n"
            if metrics.get("source") == "validation_penalty":
                failed_checks = [
                    k for k, v in validations.get(cand_id, {}).get("checks", {}).items()
                    if not v
                ]
                feedback_text += f"FAILED TESTS: {', '.join(failed_checks)}\n"
            else:
                raw = metrics.get("raw_metrics", {})
                feedback_text += (
                    f"V_out: {raw.get('simulation_output_voltage', 0):.2f}V, "
                    f"Efficiency: {raw.get('efficiency', 0):.2f}\n"
                )

            self.add_record(
                cand_id,
                netlist_text,
                fitness,
                feedback_text,
                is_valid=is_valid,
                metrics=metrics,
                batch_id=batch_id,   # ← stored so sample_states() can return it
            )

    def sample_states(self, n=2, top_k=15, epsilon=0.15):
        """Sample n states using Epsilon-Greedy Top-K Softmax.

        With probability ``(1 - epsilon)`` exploits the Top-K elite circuits
        via softmax; with probability ``epsilon`` explores the long tail
        uniformly.

        Args:
            n (int): Number of states to sample.
            top_k (int): Size of the elite pool for softmax exploitation.
            epsilon (float): Exploration probability for uniform tail sampling.

        Returns:
            list[dict]: List of record dicts, each containing the candidate
                id and all stored fields (netlist_text, fitness, etc.).
        """
        if not self.records:
            return []

        keys = list(self.records.keys())

        # 1. Sort everything by fitness first
        scored_candidates = [
            (k, self.records[k]["fitness"]) for k in keys
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        unique_scored = []
        seen_hashes = set()
        
        for k, score in scored_candidates:
            thash = self.records[k].get("topo_hash", k)
            if thash not in seen_hashes:
                seen_hashes.add(thash)
                unique_scored.append((k, score))
                
        # Use the unique list moving forward
        scored_candidates = unique_scored

        # Protect against edge case where filtering leaves us with fewer than top_k items
        actual_top_k = min(top_k, len(scored_candidates))

        # 2. Split into Elite and Long Tail
        elite_candidates = scored_candidates[:actual_top_k]
        elite_keys       = [x[0] for x in elite_candidates]
        elite_scores     = np.array([x[1] for x in elite_candidates])

        long_tail_keys = [x[0] for x in scored_candidates[actual_top_k:]]

        # 3. Softmax Probabilities
        exp_scores   = np.exp((elite_scores - np.max(elite_scores)) / self.temperature)
        elite_probs  = exp_scores / np.sum(exp_scores)

        chosen_keys = []

        for _ in range(n):
            if np.random.rand() < epsilon and len(long_tail_keys) > 0:
                chosen = np.random.choice(long_tail_keys)
                chosen_keys.append(chosen)
                long_tail_keys.remove(chosen)
            else:
                chosen = np.random.choice(elite_keys, p=elite_probs)
                chosen_keys.append(chosen)

        return [{"id": k, **self.records[k]} for k in chosen_keys]