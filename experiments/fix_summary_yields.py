import re
import json
from pathlib import Path

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def fix_run_summaries(training_folder_path: str):
    base_dir = Path(training_folder_path)
    summary_files = list(base_dir.rglob("run_summary.txt"))
    
    if not summary_files:
        print(f"❌ No 'run_summary.txt' files found inside {base_dir}")
        return

    print(f"🔧 Found {len(summary_files)} run_summary.txt files. Beginning patch...")

    for summary_path in summary_files:
        # summary_path is: pipeline/data/zycos_005/Run_008/results/run_summary.txt
        # Parent's parent gets us back to the Run_008 folder
        run_dir = summary_path.parent.parent 
        
        # Find all batch folders within this specific run
        batch_folders = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
        
        if not batch_folders:
            continue
            
        total_valid = 0
        total_invalid = 0
        
        # Directly parse the ground-truth JSON files
        for batch_folder in batch_folders:
            val_path = batch_folder / "validation_results.json"
            if val_path.exists():
                try:
                    with val_path.open("r", encoding="utf-8") as f:
                        val_data = json.load(f)
                        
                    for cand_id, data in val_data.items():
                        if data.get("passed", False):
                            total_valid += 1
                        else:
                            total_invalid += 1
                except Exception as e:
                    print(f"⚠️ Error reading {val_path.name}: {e}")
        
        # --- FILE PATCHING ---
        text = summary_path.read_text(encoding="utf-8")
        
        # 1. Update the header so we know it was fixed
        text = text.replace("--- TOPOLOGY YIELD (this batch) ---", "--- TOPOLOGY YIELD (entire run) ---")
        
        # 2. Regex to surgically replace the Valid/Invalid numbers just beneath the header
        pattern = re.compile(r"(--- TOPOLOGY YIELD[^\n]*\n)Valid:\s*\d+\nInvalid:\s*\d+")
        replacement = rf"\g<1>Valid:   {total_valid}\nInvalid: {total_invalid}"
        
        new_text = pattern.sub(replacement, text)
        
        # 3. Save it back to disk
        summary_path.write_text(new_text, encoding="utf-8")
        print(f"✅ Patched {run_dir.name} -> Valid: {total_valid}, Invalid: {total_invalid}")

if __name__ == "__main__":
    # Point this to your main training folder
    TARGET_FOLDER = PROJECT_ROOT / "pipeline" / "data" / "zycos_004"
    fix_run_summaries(TARGET_FOLDER)