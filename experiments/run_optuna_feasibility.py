import os
import sys
import glob
import time
import json
import shutil
import hashlib
import optuna
import networkx as nx
from pathlib import Path

# Fix path to import pipeline modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline.simulation.local.ltspice_runner import LTSpiceSimulator
from PyLTSpice import SpiceEditor
from pipeline.simulation.local.raw_extractor import RawExtractor
from pipeline.reward_evaluation.reward_function_norm import RewardFunctionNorm

# --- CONFIGURATION ---
DATA_DIR = PROJECT_ROOT / "pipeline" / "data"
ZYCOS_DIR = DATA_DIR / "zycos_010"
STAGING_DIR = SCRIPT_DIR / "optuna_staging"
OPTUNA_TEMP_DIR = SCRIPT_DIR / "optuna_temp_sims"

TARGET_CONSTRAINTS = [
    {"phase": "Phase3", "label": "P3_Buck_Mains",    "idx": 0},
    {"phase": "Phase3", "label": "P3_Boost_Extreme", "idx": 1},
    {"phase": "Phase3", "label": "P3_Buck_MaxPwr",   "idx": 10}
]

WEIGHTS = {
    "v_out": 20.0, "efficiency": 10.0, "volume": 2.0, "component_cost": 0.05,
    "components": {"mosfet": 1.0, "diode": 1.0, "inductor": 1.0, "capacitor": 1.0}
}


# --- GRAPH THEORETICAL SYMMETRY DETECTION ---
def get_symmetry_groups(netlist_text: str, iterations=2) -> dict:
    """
    Uses Weisfeiler-Lehman node coloring to find structurally symmetric components.
    Returns a dict mapping component types to lists of grouped component names.
    e.g., {'L': [['L1', 'L2', 'L3'], ['L4_EMI']]}
    """
    raw_lines = netlist_text.encode('ascii', 'ignore').decode('ascii').lower().split('\n')
    merged_lines = []
    
    for raw in raw_lines:
        line = raw.split(';')[0].strip()
        if not line or line.startswith('*') or line.startswith('.'): continue
        if line.startswith('+') and merged_lines: merged_lines[-1] += " " + line[1:].strip()
        else: merged_lines.append(line)

    G = nx.Graph()
    comp_types = {}
    
    # 1. Build Bipartite Graph (Components <-> Nets)
    for line in merged_lines:
        parts = line.split()
        if len(parts) < 3: continue
        comp_name = parts[0]
        comp_type = comp_name[0]
        if comp_type not in {'v', 'r', 'l', 'c', 'd', 'm'}: continue
        
        comp_id = f"comp_{comp_name}"
        G.add_node(comp_id, label=comp_type, type='comp', orig_name=comp_name)
        comp_types[comp_name] = comp_type
        
        if comp_type == 'm': nets = parts[1:5]
        elif comp_type in ['r', 'l', 'c', 'd', 'v']: nets = parts[1:3]
        else: continue
        
        for net in nets:
            net_id = f"net_{net}"
            if net == '0': base_label = 'gnd'
            elif net == 'in': base_label = 'input'
            elif net == 'out': base_label = 'output'
            else: base_label = 'net'
            
            if net_id not in G: G.add_node(net_id, label=base_label, type='net')
            G.add_edge(comp_id, net_id)

    # 2. Iterative Node Coloring (WL)
    for _ in range(iterations):
        new_labels = {}
        for node in G.nodes():
            curr_label = G.nodes[node]['label']
            neighbor_labels = sorted([G.nodes[nbr]['label'] for nbr in G.neighbors(node)])
            hash_input = curr_label + "|" + ",".join(neighbor_labels)
            new_labels[node] = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        for node, lbl in new_labels.items():
            G.nodes[node]['label'] = lbl
            
    # 3. Group by identical color hashes
    groups = {'l': {}, 'c': {}, 'v': {}}
    for node, data in G.nodes(data=True):
        if data['type'] == 'comp':
            c_type = comp_types[data['orig_name']]
            if c_type in groups:
                lbl = data['label']
                if lbl not in groups[c_type]: groups[c_type][lbl] = []
                # PyLTSpice prefers uppercase references
                groups[c_type][lbl].append(data['orig_name'].upper()) 
                
    return {
        'L': list(groups['l'].values()),
        'C': list(groups['c'].values()),
        'V': list(groups['v'].values())
    }

# --- HELPER FUNCTIONS ---
def get_pure_topology_hash(netlist_text: str) -> str:
    """Simplified hasher just for unique staging."""
    return hashlib.md5(netlist_text.encode('ascii', 'ignore')).hexdigest()[:12]

def extract_unique_valid_topologies():
    print("🔍 [PROFILING] Starting Topology Extraction...")
    start_time = time.perf_counter()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes, extracted_count = set(), 0
    
    for batch in glob.glob(str(ZYCOS_DIR / "Run_*" / "batch_*")):
        val_path = Path(batch) / "validation_results.json"
        if not val_path.exists(): continue
        with open(val_path, "r") as f: val_data = json.load(f)
            
        for cand_name, data in val_data.items():
            if data.get("passed", False):
                net_path = Path(batch) / "LLM_output" / f"{cand_name}.net"
                if not net_path.exists(): continue
                with open(net_path, 'r', encoding='utf-8') as f: net_text = f.read()
                pure_hash = get_pure_topology_hash(net_text)
                if pure_hash not in seen_hashes:
                    seen_hashes.add(pure_hash)
                    shutil.copy(net_path, STAGING_DIR / f"{pure_hash}.net")
                    extracted_count += 1

    print(f"✅ [PROFILING] Extraction Complete: Found {extracted_count} topologies in {time.perf_counter() - start_time:.2f}s.\n")
    return list(STAGING_DIR.glob("*.net"))

def _to_float(s):
    if isinstance(s, (int, float)): return float(s)
    s = s.strip().lower()
    for suffix, mult in [("meg", 1e6), ("g", 1e9), ("f", 1e-15), ("p", 1e-12), ("n", 1e-9), ("u", 1e-6), ("m", 1e-3), ("k", 1e3)]:
        if s.endswith(suffix): return float(s[:-len(suffix)]) * mult
    try: return float(s)
    except: return 0.0

def modify_pulse_string(pulse_str, group_t_per, group_duty):
    """Safely scales Ton and Delay to match the new Frequency."""
    parts = pulse_str.upper().replace('PULSE(', '').replace(')', '').split()
    if len(parts) >= 7:
        orig_delay, orig_tper = _to_float(parts[2]), _to_float(parts[6])
        phase_ratio = (orig_delay / orig_tper) if orig_tper > 0 else 0.0
        parts[2] = f"{(group_t_per * phase_ratio):.3e}"
        parts[5] = f"{(group_t_per * group_duty):.3e}"
        parts[6] = f"{group_t_per:.3e}"
        return f"PULSE({' '.join(parts)})"
    return pulse_str

def load_constraint_dict(idx):
    with open(PROJECT_ROOT / "pipeline" / "data" / "datasets" / "constraints_hard.json", 'r') as f:
        return json.load(f)[idx]

# --- OPTUNA OBJECTIVE FUNCTION ---
def create_objective(base_net_path, constraint_dict, sym_groups):
    reward_calculator = RewardFunctionNorm()
    
    def objective(trial):
        trial_start = time.perf_counter()
        editor = SpiceEditor(base_net_path)
        
        # 1. Apply Inductor Groups
        for idx, l_group in enumerate(sym_groups.get('L', [])):
            val = trial.suggest_float(f"L_Grp_{idx}", 1e-6, 1e-3, log=True)
            for l_name in l_group: editor.set_component_value(l_name, f"{val:.3e}")
                
        # 2. Apply Capacitor Groups
        for idx, c_group in enumerate(sym_groups.get('C', [])):
            val = trial.suggest_float(f"C_Grp_{idx}", 1e-6, 10e-3, log=True)
            for c_name in c_group: editor.set_component_value(c_name, f"{val:.3e}")
                
        # 3. Apply PULSE Source Groups
        for idx, v_group in enumerate(sym_groups.get('V', [])):
            # Verify group contains at least one PULSE source before creating Optuna vars
            is_pulse = any('PULSE' in str(editor.get_component_value(v)).upper() for v in v_group)
            
            if is_pulse:
                t_per = trial.suggest_float(f"V_Grp_{idx}_Tper", 1e-6, 100e-6, log=True)
                duty = trial.suggest_float(f"V_Grp_{idx}_Duty", 0.05, 0.95)
                
                for v_name in v_group:
                    val_str = editor.get_component_value(v_name)
                    if val_str and 'PULSE' in val_str.upper():
                        editor.set_component_value(v_name, modify_pulse_string(val_str, t_per, duty))

        # 4. Simulate & Extract
        trial_net_path = OPTUNA_TEMP_DIR / f"trial_{trial.number}.net"
        editor.write_netlist(str(trial_net_path))
        
        simulator = LTSpiceSimulator(output_dir=OPTUNA_TEMP_DIR)
        netlist_map = simulator.simulate([trial_net_path])
        
        extractor = RawExtractor(output_dir=OPTUNA_TEMP_DIR)
        results_csv_path = OPTUNA_TEMP_DIR / "temp_results.csv"
        
        try: rows = extractor.extract(netlist_map, results_csv_path)
        except AssertionError: rows = []  # LTspice Instability Abort
        except Exception: rows = []

        # Cleanup Workspace
        for temp_file in OPTUNA_TEMP_DIR.glob("*"):
            try: temp_file.unlink()
            except PermissionError: pass 
            
        if not rows: return 1.0 # Max Penalty
            
        row = rows[0]
        total_loss, _ = reward_calculator.calculate_loss(row, constraint_dict, WEIGHTS)
        print(f"  [Trial {trial.number}] Loss: {total_loss:.4f} | Vout: {row.get('voltage_out_mean_V', 0):.2f}V | Eff: {row.get('efficiency', 0):.2%} | Time: {(time.perf_counter() - trial_start):.2f}s")
        return total_loss

    return objective


# --- MAIN EXECUTION ---
def run_feasibility_study():
    unique_nets = extract_unique_valid_topologies()
    if not unique_nets: return
    OPTUNA_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    test_net = unique_nets[0]
    print(f"\n🚀 [PROFILING] Starting Optuna Feasibility on Topology: {test_net.stem}")
    
    with open(test_net, 'r', encoding='utf-8') as f:
        net_text = f.read()
        
    # --- PROFILING THE GRAPH ALGORITHM ---
    wl_start = time.perf_counter()
    sym_groups = get_symmetry_groups(net_text)
    wl_elapsed_ms = (time.perf_counter() - wl_start) * 1000
    # -------------------------------------
    print(f"\n🧩 Detected Symmetry Groups (Completed in {wl_elapsed_ms:.3f} ms):")
    print(json.dumps(sym_groups, indent=2))
    
    for target in TARGET_CONSTRAINTS:
        print(f"\n{'='*50}")
        print(f"🎯 Evaluating Constraint: {target['label']}")
        
        study_start = time.perf_counter()
        study = optuna.create_study(direction="minimize")
        study.optimize(create_objective(test_net, load_constraint_dict(target['idx']), sym_groups), n_trials=15)
        
        print(f"\n🏆 Study Complete: {target['label']} | Best Loss: {study.best_value:.4f} | Time: {(time.perf_counter() - study_start):.2f}s")
        print(f"   -> Best Params: {study.best_params}")

if __name__ == "__main__":
    run_feasibility_study()