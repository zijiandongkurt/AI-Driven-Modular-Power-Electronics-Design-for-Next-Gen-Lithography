import os
import json
import glob
import re

def analyze_single_run(run_dir):
    """Uses history_db.json to map evolutionary success using real fitness scores."""
    history_path = os.path.join(run_dir, 'history_db.json')
    valid_files = glob.glob(os.path.join(run_dir, 'batch_*', 'validation_results.json'))
    
    if not os.path.exists(history_path):
        return None, "History file not found."

    with open(history_path, 'r') as f: history = json.load(f)
    
    # 1. Map validity for quick lookup (True=Valid, False=Invalid)
    all_validity = {}
    for vf in valid_files:
        with open(vf, 'r') as f:
            val_data = json.load(f)
            all_validity.update({k: v.get('passed', False) for k, v in val_data.items()})

    # 2. Map fitness for quick lookup
    # We map netlist_id to its fitness score. 
    # If a circuit doesn't have a fitness score in history, it is skipped.
    fitness_map = {entry['netlist_id']: entry.get('fitness') for entry in history if 'fitness' in entry}

    # 3. Analyze events
    counts = {
        "total": 0, "stagnant": 0, "better": 0,
        "valid": 0, "stagnant_valid": 0, "better_valid": 0,
        "invalid": 0, "stagnant_invalid": 0, "better_invalid": 0
    }
    
    for entry in history:
        parent_id = entry.get('parent_id')
        child_fit = entry.get('fitness')
        
        # Skip if no parent or missing fitness data (avoiding false comparisons)
        if parent_id is None or parent_id not in fitness_map or child_fit is None:
            continue
            
        parent_fit = fitness_map[parent_id]
        
        counts["total"] += 1
        
        # Parent Validity Status
        is_valid = all_validity.get(parent_id, False) # Default to False if status unknown
        
        # Logic: Betterment if child > parent, Stagnation if child <= parent
        is_stagnant = (child_fit <= parent_fit)
        
        if is_stagnant:
            counts["stagnant"] += 1
            if is_valid: counts["stagnant_valid"] += 1
            else: counts["stagnant_invalid"] += 1
        else:
            counts["better"] += 1
            if is_valid: counts["better_valid"] += 1
            else: counts["better_invalid"] += 1
            
        if is_valid: counts["valid"] += 1
        else: counts["invalid"] += 1

    def get_rate(n, d): return (n / d * 100) if d > 0 else 0
    
    report = (
        f"Evolutionary Audit: {os.path.basename(run_dir)}\n"
        f"----------------------------------------------\n"
        f"Global Stagnation: {get_rate(counts['stagnant'], counts['total']):.1f}% ({counts['stagnant']}/{counts['total']})\n"
        f"Global Betterment: {get_rate(counts['better'], counts['total']):.1f}% ({counts['better']}/{counts['total']})\n"
        f"Valid Parent Stagnation: {get_rate(counts['stagnant_valid'], counts['valid']):.1f}% ({counts['stagnant_valid']}/{counts['valid']})\n"
        f"Valid Parent Betterment: {get_rate(counts['better_valid'], counts['valid']):.1f}% ({counts['better_valid']}/{counts['valid']})\n"
        f"Invalid Parent Stagnation: {get_rate(counts['stagnant_invalid'], counts['invalid']):.1f}% ({counts['stagnant_invalid']}/{counts['invalid']})\n"
        f"Invalid Parent Betterment: {get_rate(counts['better_invalid'], counts['invalid']):.1f}% ({counts['better_invalid']}/{counts['invalid']})\n"
    )
    return counts, report

def run_session_analysis(session_name):
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    SESSION_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data', session_name)
    run_folders = sorted(glob.glob(os.path.join(SESSION_DIR, 'Run_*')))
    
    agg = {k: 0 for k in ["total", "stagnant", "better", "valid", "stagnant_valid", "better_valid", "invalid", "stagnant_invalid", "better_invalid"]}

    print(f"🚀 Analyzing {len(run_folders)} runs in {session_name}...")
    
    for run_dir in run_folders:
        stats, report = analyze_single_run(run_dir)
        if stats:
            for k in agg: agg[k] += stats[k]
            os.makedirs(os.path.join(run_dir, 'results'), exist_ok=True)
            with open(os.path.join(run_dir, 'results', 'evolution_report.txt'), 'w') as f: f.write(report)
            print(f"✅ {os.path.basename(run_dir)} processed.")

    def get_rate(n, d): return (n / d * 100) if d > 0 else 0
    
    summary_text = (
        f"AGGREGATED SESSION EVOLUTIONARY SUMMARY: {session_name}\n"
        f"========================================================\n"
        f"Global Stagnation: {get_rate(agg['stagnant'], agg['total']):.2f}% ({agg['stagnant']}/{agg['total']})\n"
        f"Global Betterment: {get_rate(agg['better'], agg['total']):.2f}% ({agg['better']}/{agg['total']})\n"
        f"--------------------------------------------------------\n"
        f"Conditional Stagnation (Valid Parents): {get_rate(agg['stagnant_valid'], agg['valid']):.2f}% ({agg['stagnant_valid']}/{agg['valid']})\n"
        f"Conditional Betterment (Valid Parents): {get_rate(agg['better_valid'], agg['valid']):.2f}% ({agg['better_valid']}/{agg['valid']})\n"
        f"--------------------------------------------------------\n"
        f"Conditional Stagnation (Invalid Parents): {get_rate(agg['stagnant_invalid'], agg['invalid']):.2f}% ({agg['stagnant_invalid']}/{agg['invalid']})\n"
        f"Conditional Betterment (Invalid Parents): {get_rate(agg['better_invalid'], agg['invalid']):.2f}% ({agg['better_invalid']}/{agg['invalid']})\n"
    )
    
    with open(os.path.join(SESSION_DIR, 'session_evolution_summary.txt'), 'w') as f:
        f.write(summary_text)
    print(f"\n🎉 Analysis complete! Summary saved to: {os.path.join(SESSION_DIR, 'session_evolution_summary.txt')}")

if __name__ == '__main__':
    run_session_analysis("zycos_009")