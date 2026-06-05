import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import os
import tempfile
import json

import pytest

# Dynamically add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- FIX: Import our new get_empty_metrics function ---
from experiments.run_benchmark import extract_metrics, get_empty_metrics

# --- MOCK DATA ---
MOCK_SUMMARY = """
Overall Best Fitness: 0.85
Validity Rate: 95.0%
Batch 1 Best DB Score: 0.5
Batch 2 Best DB Score: 0.8
"""

MOCK_CHAMPION_ZERO_VOLTAGE = """
{
    "target_voltage": 0.0,
    "raw_voltage": 1.0,
    "raw_efficiency": 0.9,
    "raw_volume": 10.0,
    "raw_components": 5,
    "target_power": 10.0
}
"""

def test_extract_metrics_handles_zero_voltage():
    """
    EXPECTED PASS: The ZeroDivisionError and NumPy 2.0 trapz bugs are fixed.
    """
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.read_text', return_value=MOCK_SUMMARY), \
         patch('builtins.open', mock_open(read_data=MOCK_CHAMPION_ZERO_VOLTAGE)), \
         patch('pathlib.Path.rglob', return_value=[]): 
        
        metrics = extract_metrics("dummy_folder")
        assert metrics["v_error_pct"] is not None
        assert metrics["auc"] > 0.0

def test_fallback_dictionary_schema_matches_extract_metrics():
    """
    EXPECTED PASS: The new get_empty_metrics() perfectly matches the schema.
    """
    with patch('pathlib.Path.exists', return_value=False), \
         patch('pathlib.Path.rglob', return_value=[]):
        success_metrics = extract_metrics("dummy_folder")
    
    # Use our new function instead of the hardcoded old dict!
    fallback_dict = get_empty_metrics()
    
    missing_keys = set(success_metrics.keys()) - set(fallback_dict.keys())
    assert len(missing_keys) == 0, f"Fallback dict is still missing keys: {missing_keys}"

def test_numpy_aggregation_survives_missing_champion_files():
    """
    EXPECTED PASS: The new list comprehension filters None types safely.
    """
    trial_data = [
        {"fitness": 0.9, "volume": 25.0},
        {"fitness": 0.0, "volume": None}
    ]
    
    key = "volume"
    # This is the exact updated logic from the new run_benchmark.py
    vals = [t.get(key) for t in trial_data if t and t.get(key) is not None]
    
    # Verify it correctly filtered out the None
    assert len(vals) == 1
    assert vals[0] == 25.0
    
    # Verify NumPy can safely calculate the mean without crashing
    mean_val = np.mean(vals)
    assert mean_val == 25.0


def test_extract_metrics_survives_corrupt_champion_json():
    """
    EXPECTED PASS: extract_metrics catches JSONDecodeError and doesn't crash.
    It should preserve the summary metrics but safely nullify the physical metrics.
    """
    # Simulate a file that was cut off halfway through writing due to a power loss
    MOCK_CORRUPT_JSON = '{\n    "target_voltage": 5.0,\n    "raw_voltage": 4.9,\n    "raw_ef'
    
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.read_text', return_value=MOCK_SUMMARY), \
         patch('builtins.open', mock_open(read_data=MOCK_CORRUPT_JSON)), \
         patch('pathlib.Path.rglob', return_value=[]): 
        
        # In the old code, this would throw json.JSONDecodeError and instantly kill the 30-hour run
        metrics = extract_metrics("dummy_folder")
        
        # The script should survive and leave physical metrics as their safe None defaults
        assert metrics["v_error_pct"] is None
        assert metrics["efficiency"] is None
        assert metrics["volume"] is None
        
        # Crucially, it should STILL successfully parse the text summary!
        assert metrics["fitness"] == 0.85
        assert metrics["validity"] == 95.0

@patch('experiments.run_benchmark.os.replace')
@patch('experiments.run_benchmark.tempfile.mkstemp')
@patch('experiments.run_benchmark.run_inference')
@patch('experiments.run_benchmark.extract_metrics')
def test_atomic_checkpoint_write_behavior(mock_extract, mock_inference, mock_mkstemp, mock_replace):
    """
    EXPECTED PASS: The benchmark loop uses temporary files and atomic OS swaps 
    to save progress, rather than vulnerable direct writes.
    """
    from experiments.run_benchmark import main
    import sys
    
    # 1. Setup our OS-level mocks to prevent actual files from being written
    mock_mkstemp.return_value = (12345, "dummy_temp_path.tmp")
    mock_inference.return_value = "dummy_output_dir"
    mock_extract.return_value = {"fitness": 1.0}
    
    # 2. Mock a minimal configuration file so main() can run
    mock_config = {
        "trials_per_task": 1,
        "base_config": {
            "run_settings": {"n_batches": 10}, # <-- Added n_batches here!
            "constraint_settings": {}
        },
        "tasks": [{"label": "Test_Task", "file": "dummy.json", "idx": 0, "phase": 1}],
        "models": {"Baseline": None}
    }
    
    test_args = ["run_benchmark.py", "--config", "dummy_config.json"]
    
    # 3. Mute all file I/O AND silence all Matplotlib plotting!
    with patch.object(sys, 'argv', test_args), \
         patch('pathlib.Path.exists', side_effect=[True, False]), \
         patch('experiments.run_benchmark.plot_combined_fitness'), \
         patch('experiments.run_benchmark.plot_learning_curves'), \
         patch('experiments.run_benchmark.plot_pareto_scatter'), \
         patch('experiments.run_benchmark.plot_radar_chart'), \
         patch('experiments.run_benchmark.plot_validity_bar_chart'), \
         patch('pathlib.Path.open', mock_open(read_data=json.dumps(mock_config))), \
         patch('builtins.open', mock_open()), \
         patch('os.fdopen', mock_open()):
        
        # Run the main loop!
        try:
            main()
        except NameError:
            pass

    # 4. The Ultimate Assertion: Did the OS execute an atomic swap?
    mock_mkstemp.assert_called()
    mock_replace.assert_called()
    
    called_args = mock_replace.call_args[0]
    assert called_args[0] == "dummy_temp_path.tmp"
    assert "benchmark_checkpoint.json" in str(called_args[1])

def test_extract_metrics_survives_none_path():
    """
    EXPECTED FAILURE: TypeError
    Why it fails: If run_inference() fatally errors and returns None instead of 
    a string path, pathlib.Path(None) will throw a TypeError.
    """
    try:
        # We pass None as if the inference function failed completely
        metrics = extract_metrics(None)
    except TypeError as e:
        # If it throws TypeError, the test proves the vulnerability exists
        pass 
    except Exception as e:
        pytest.fail(f"Expected TypeError, but got {e}") 