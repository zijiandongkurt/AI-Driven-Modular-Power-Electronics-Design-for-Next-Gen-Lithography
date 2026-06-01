import re
from pathlib import Path
from collections import defaultdict

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def normalize_netlist(text: str) -> str:
    """Removes comments and blank lines, and converts to lowercase."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip().lower()
        if not line or line.startswith('*'):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def get_duplicates_per_batch(run_folder_path: str):
    """
    Chronologically scans raw_output.txt per batch to count mode collapse.
    Returns a dictionary mapping {batch_number: duplicate_count}.
    """
    base_dir = Path(run_folder_path)
    batch_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    
    # Sort folders strictly by batch number so we process chronologically
    def extract_batch_num(folder_name):
        match = re.search(r'batch_(\d+)', folder_name)
        return int(match.group(1)) if match else -1
        
    batch_folders = sorted(batch_folders, key=lambda x: extract_batch_num(x.name))
    
    seen_netlists = set()
    duplicates_per_batch = {}
    
    for batch_folder in batch_folders:
        batch_num = extract_batch_num(batch_folder.name)
        raw_file = batch_folder / "raw_output.txt"
        
        dup_count = 0
        if raw_file.exists():
            text = raw_file.read_text(encoding="utf-8")
            blocks = re.split(r"={10,}\n\s*CANDIDATE \d+\s*\n={10,}\n", text)
            
            for block in blocks[1:]:
                block = block.strip()
                if not block:
                    continue
                    
                norm_text = normalize_netlist(block)
                # If we have seen this exact logic in ANY previous batch (or earlier in this batch)
                if norm_text in seen_netlists:
                    dup_count += 1
                else:
                    seen_netlists.add(norm_text)
                    
        duplicates_per_batch[batch_num] = dup_count
        
    return duplicates_per_batch

def evaluate_duplicates(run_folder_path: str):
    """Standalone diagnostic printout."""
    # (Your previous print logic remains intact here if you want to run it directly)
    dup_data = get_duplicates_per_batch(run_folder_path)
    print(f"Duplicate Summary for {Path(run_folder_path).name}:")
    for batch, count in dup_data.items():
        print(f"  Batch {batch}: {count} duplicates")

if __name__ == "__main__":
    TARGET_RUN = PROJECT_ROOT / "pipeline" / "data" / "zycos_006" / "Run_005"
    evaluate_duplicates(TARGET_RUN)