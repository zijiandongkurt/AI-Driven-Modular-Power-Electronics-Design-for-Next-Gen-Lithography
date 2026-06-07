import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Go up two levels to reach the root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the functions directly from the training loop
from training_loop_random import epsilon_greedy_topk_sample_parents, add_batch_to_history


def test_epsilon_greedy_deduplication():
    """Verify the sampling algorithm strips topological duplicates before selecting parents."""
    
    fake_history = [
        {"netlist_id": "cand_1", "fitness": 0.9, "topo_hash": "hash_A", "depth": 1, "batch_id": "b1"},
        {"netlist_id": "cand_2", "fitness": 0.8, "topo_hash": "hash_A", "depth": 1, "batch_id": "b1"}, # Duplicate!
        {"netlist_id": "cand_3", "fitness": 0.7, "topo_hash": "hash_B", "depth": 1, "batch_id": "b1"},
    ]
    
    selected = epsilon_greedy_topk_sample_parents(
        history=fake_history,
        k=2,
        top_k=2,
        epsilon=0.0, 
        temperature=1e-5, 
        seed=42
    )
    
    selected_ids = [p["netlist_id"] for p in selected]
    
    assert len(selected) == 2
    assert "cand_1" in selected_ids, "Should keep the highest fitness version of hash_A"
    assert "cand_2" not in selected_ids, "Should drop the duplicate topology"
    assert "cand_3" in selected_ids, "Should pick the next unique topology"


@patch("training_loop_random.load_reward_data")
@patch("training_loop_random.get_topological_hash")
def test_add_batch_to_history(mock_hasher, mock_load_reward, monkeypatch):
    """Verify that adding a batch reads the .net file and stores the topological hash."""
    
    mock_load_reward.return_value = {
        "circuits": {
            "test_g1_cand1_b1": {"fitness_score": 0.85}
        }
    }
    mock_hasher.return_value = "mocked_hash_123"

    # Safely mock the Path object just inside the target module
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.read_text.return_value = "L1 in out 10u"
    
    # When Path is divided by a string (Path / "folder"), return our mock
    mock_path_instance.__truediv__.return_value = mock_path_instance
    
    mock_path_class = MagicMock(return_value=mock_path_instance)
    monkeypatch.setattr("training_loop_random.Path", mock_path_class)
    
    history = []
    group_to_parent = {"g1": None}
    
    add_batch_to_history(history, "batch_1", group_to_parent, default_depth=1)
    
    assert len(history) == 1
    assert history[0]["netlist_id"] == "test_g1_cand1_b1"
    assert history[0]["topo_hash"] == "mocked_hash_123", "The history dictionary must store the topological hash"
    assert history[0]["fitness"] == 0.85

@patch("training_loop_random.load_reward_data")
def test_add_batch_to_history_missing_file(mock_load_reward, monkeypatch):
    """Verify that missing .net files are handled gracefully with an 'unknown' hash fallback."""
    
    mock_load_reward.return_value = {
        "circuits": {
            "test_g1_cand1_b1": {"fitness_score": 0.85}
        }
    }
    
    # Simulate the .net file NOT existing
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = False
    mock_path_instance.__truediv__.return_value = mock_path_instance
    
    mock_path_class = MagicMock(return_value=mock_path_instance)
    monkeypatch.setattr("training_loop_random.Path", mock_path_class)
    
    history = []
    group_to_parent = {"g1": None}
    
    add_batch_to_history(history, "batch_1", group_to_parent, default_depth=1)
    
    assert len(history) == 1
    assert history[0]["topo_hash"] == "unknown", "Should fallback to 'unknown' if file is missing"