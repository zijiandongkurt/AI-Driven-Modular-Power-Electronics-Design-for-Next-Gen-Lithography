import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from training_loop_random import (
    parse_group_id,
    get_next_zycos_folder,
    load_reward_data,
    save_history,
    epsilon_greedy_topk_sample_parents,
    _softmax_sample_from_pool,
)


#  parse_group_id 

class TestParseGroupId:
    def test_standard_name(self):
        assert parse_group_id("Phase1_cons0_g3_cand2_b1") == "g3"

    def test_first_group(self):
        assert parse_group_id("Phase2_cons1_g1_cand1_b4") == "g1"

    def test_no_match_returns_none(self):
        assert parse_group_id("Phase1_cons0_cand1_b1") is None

    def test_empty_string_returns_none(self):
        assert parse_group_id("") is None

    def test_group_number_preserved(self):
        assert parse_group_id("Phase1_g12_cand3_b2") == "g12"


#  get_next_zycos_folder 

class TestGetNextZycosFolder:
    def test_empty_directory_returns_001(self, tmp_path):
        result = get_next_zycos_folder(tmp_path)
        assert result == "zycos_001"

    def test_increments_from_existing(self, tmp_path):
        for name in ["zycos_001", "zycos_002", "zycos_003"]:
            (tmp_path / name).mkdir()
        result = get_next_zycos_folder(tmp_path)
        assert result == "zycos_004"

    def test_zero_pads_to_three_digits(self, tmp_path):
        (tmp_path / "zycos_009").mkdir()
        result = get_next_zycos_folder(tmp_path)
        assert result == "zycos_010"

    def test_skips_non_zycos_dirs(self, tmp_path):
        (tmp_path / "zycos_005").mkdir()
        (tmp_path / "other_dir").mkdir()
        result = get_next_zycos_folder(tmp_path)
        assert result == "zycos_006"

    def test_creates_directory_if_missing(self, tmp_path):
        new_dir = tmp_path / "data"
        get_next_zycos_folder(new_dir)
        assert new_dir.exists()


#  load_reward_data 

class TestLoadRewardData:
    def test_returns_empty_circuits_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_reward_data("zycos_999/batch_99")
        assert result == {"circuits": {}}

    def test_loads_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        batch_dir = tmp_path / "pipeline" / "data" / "zycos_001" / "batch_1"
        batch_dir.mkdir(parents=True)
        data = {"circuits": {"cand1": {"fitness_score": 0.85}}}
        (batch_dir / "reward_results.json").write_text(json.dumps(data))

        result = load_reward_data("zycos_001/batch_1")
        assert result["circuits"]["cand1"]["fitness_score"] == 0.85

    def test_returns_empty_circuits_on_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        batch_dir = tmp_path / "pipeline" / "data" / "zycos_001" / "batch_1"
        batch_dir.mkdir(parents=True)
        (batch_dir / "reward_results.json").write_text("{ not valid json }")

        result = load_reward_data("zycos_001/batch_1")
        assert result == {"circuits": {}}


#  save_history 

class TestSaveHistory:
    def test_roundtrip(self, tmp_path):
        history = [
            {"netlist_id": "cand1", "fitness": 0.9, "topo_hash": "abc", "depth": 1, "batch_id": "b1"},
            {"netlist_id": "cand2", "fitness": 0.7, "topo_hash": "xyz", "depth": 2, "batch_id": "b1"},
        ]
        save_history(history, tmp_path)
        saved = json.loads((tmp_path / "history_db.json").read_text())
        assert len(saved) == 2
        assert saved[0]["netlist_id"] == "cand1"
        assert saved[1]["topo_hash"] == "xyz"

    def test_empty_history_writes_empty_list(self, tmp_path):
        save_history([], tmp_path)
        saved = json.loads((tmp_path / "history_db.json").read_text())
        assert saved == []


#  _softmax_sample_from_pool 

class TestSoftmaxSampleFromPool:
    def _make_pool(self, fitnesses):
        return [{"netlist_id": f"c{i}", "fitness": f} for i, f in enumerate(fitnesses)]

    def test_returns_item_from_pool(self):
        import random
        pool = self._make_pool([0.5, 0.7, 0.9])
        chosen = _softmax_sample_from_pool(pool, temperature=1.0, rng=random.Random(0))
        assert chosen in pool

    def test_single_item_always_chosen(self):
        import random
        pool = self._make_pool([0.8])
        chosen = _softmax_sample_from_pool(pool, temperature=1.0, rng=random.Random(0))
        assert chosen == pool[0]

    def test_very_low_temperature_prefers_best(self):
        import random
        pool = self._make_pool([0.1, 0.2, 0.99])
        counts = {}
        rng = random.Random(1)
        for _ in range(200):
            chosen = _softmax_sample_from_pool(pool, temperature=1e-6, rng=rng)
            counts[chosen["netlist_id"]] = counts.get(chosen["netlist_id"], 0) + 1
        # With near-zero temperature the best item (fitness=0.99) should dominate
        assert counts.get("c2", 0) > 150


#  epsilon_greedy_topk_sample_parents 

class TestEpsilonGreedyTopkSampleParents:
    def _make_history(self, n, base_fitness=0.5):
        return [
            {"netlist_id": f"c{i}", "fitness": base_fitness + i * 0.01,
             "topo_hash": f"hash_{i}", "depth": 1, "batch_id": "b1"}
            for i in range(n)
        ]

    def test_empty_history_raises(self):
        with pytest.raises(RuntimeError, match="History is completely empty"):
            epsilon_greedy_topk_sample_parents([], k=2, top_k=5, epsilon=0.0, temperature=1.0)

    def test_returns_correct_count(self):
        history = self._make_history(10)
        selected = epsilon_greedy_topk_sample_parents(
            history, k=4, top_k=5, epsilon=0.0, temperature=1e-5, seed=42
        )
        assert len(selected) == 4

    def test_no_duplicates_in_selection(self):
        history = self._make_history(10)
        selected = epsilon_greedy_topk_sample_parents(
            history, k=4, top_k=5, epsilon=0.0, temperature=1e-5, seed=42
        )
        ids = [p["netlist_id"] for p in selected]
        assert len(ids) == len(set(ids))

    def test_greedy_mode_picks_top_fitness(self):
        history = self._make_history(10)
        selected = epsilon_greedy_topk_sample_parents(
            history, k=3, top_k=3, epsilon=0.0, temperature=1e-9, seed=42
        )
        fitnesses = [p["fitness"] for p in selected]
        all_fitnesses = sorted([h["fitness"] for h in history], reverse=True)
        for f in fitnesses:
            assert f in all_fitnesses[:3]

    def test_topological_deduplication(self):
        # Two entries share the same topo_hash — only one should be eligible
        history = [
            {"netlist_id": "c0", "fitness": 0.9, "topo_hash": "same_hash", "depth": 1, "batch_id": "b1"},
            {"netlist_id": "c1", "fitness": 0.8, "topo_hash": "same_hash", "depth": 1, "batch_id": "b1"},
            {"netlist_id": "c2", "fitness": 0.7, "topo_hash": "unique_hash", "depth": 1, "batch_id": "b1"},
        ]
        selected = epsilon_greedy_topk_sample_parents(
            history, k=2, top_k=3, epsilon=0.0, temperature=1e-9, seed=42
        )
        ids = [p["netlist_id"] for p in selected]
        assert "c1" not in ids, "Duplicate topology must be eliminated; c0 should be kept instead"
        assert "c0" in ids

    def test_exploration_reaches_long_tail(self):
        # With epsilon=1.0 (pure exploration) we should sometimes pick non-top-k items
        history = self._make_history(20)
        top_k_ids = {h["netlist_id"] for h in sorted(history, key=lambda x: x["fitness"], reverse=True)[:5]}

        collected = set()
        for seed in range(50):
            selected = epsilon_greedy_topk_sample_parents(
                history, k=1, top_k=5, epsilon=1.0, temperature=1.0, seed=seed
            )
            collected.add(selected[0]["netlist_id"])

        tail_picks = collected - top_k_ids
        assert len(tail_picks) > 0, "Pure exploration never picked from the long tail"

    def test_fallback_when_k_exceeds_unique(self):
        # Only 2 unique topologies but k=4 — should not raise, uses fallback
        history = [
            {"netlist_id": "c0", "fitness": 0.9, "topo_hash": "h1", "depth": 1, "batch_id": "b1"},
            {"netlist_id": "c1", "fitness": 0.8, "topo_hash": "h2", "depth": 1, "batch_id": "b1"},
        ]
        selected = epsilon_greedy_topk_sample_parents(
            history, k=4, top_k=9, epsilon=0.0, temperature=1e-5, seed=42
        )
        assert len(selected) == 4  # allows duplicates via fallback