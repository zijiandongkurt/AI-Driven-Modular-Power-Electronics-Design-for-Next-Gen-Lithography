import os
import glob
import networkx as nx

def get_pure_topology_hash(netlist_text: str) -> str:
    """Compute a WL graph hash based solely on physical wiring structure.

    Ignores component values and MOSFET parameters entirely; only the
    connectivity topology is considered.

    Args:
        netlist_text (str): Raw SPICE netlist text to hash.

    Returns:
        str: Weisfeiler-Lehman graph hash string representing the wiring structure.
    """
    netlist_text = netlist_text.encode('ascii', 'ignore').decode('ascii').lower()
    raw_lines = netlist_text.split('\n')
    merged_lines = []
    
    for raw in raw_lines:
        line = raw.split(';')[0].strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
        if line.startswith('+'):
            if merged_lines:
                merged_lines[-1] += " " + line[1:].strip()
        else:
            merged_lines.append(line)
            
    G = nx.Graph()
    comp_counter = 0
    allowed_prefixes = {'v', 'r', 'l', 'c', 'd', 'm'}
    
    for line in merged_lines:
        parts = line.split()
        if len(parts) < 3: 
            continue
        
        comp_type = parts[0][0]
        if comp_type not in allowed_prefixes:
            continue
            
        if comp_type == 'm': 
            nets = parts[1:5]
        elif comp_type in ['r', 'l', 'c', 'd', 'v']: 
            nets = parts[1:3]
        else:
            continue
            
        comp_node = f"comp_{comp_counter}"
        comp_counter += 1
        
        # KEY DIFFERENCE: We label the node purely as 'r', 'c', 'm', etc.
        # We completely discard the SPICE values (e.g., 47u, W=1)
        G.add_node(comp_node, label=comp_type)
        
        for pin_idx, net in enumerate(nets):
            if net == '0':
                net_label = 'gnd'
            elif net == 'in':
                net_label = 'input'
            elif net == 'out':
                net_label = 'output'
            elif 'gate' in net:
                net_label = 'gate_drive'
            else:
                net_label = 'internal_net'
                
            G.add_node(net, label=net_label)
            
            if comp_type in ['r', 'l', 'c']:
                G.add_edge(comp_node, net)
            else:
                pin_node = f"{comp_node}_pin_{pin_idx}"
                G.add_node(pin_node, label=f"pin_{pin_idx}")
                G.add_edge(comp_node, pin_node)
                G.add_edge(pin_node, net)

    return nx.weisfeiler_lehman_graph_hash(G, node_attr='label')

def analyze_topologies():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline', 'data')
    
    grand_total_hashes = set()
    output_lines = []
    
    output_lines.append("=== PURE STRUCTURAL TOPOLOGY EXPLORATION COUNTS ===\n")
    output_lines.append("Note: This counts EVERY topology explored (Valid + Invalid), ignoring component values.\n\n")

    print("🔍 Scanning zycos_008 through zycos_010...")

    for i in range(8, 11):
        zycos_name = f"zycos_{i:03d}"
        zycos_dir = os.path.join(DATA_DIR, zycos_name)
        
        if not os.path.exists(zycos_dir):
            print(f"  ⚠️ Skipping {zycos_name} (Directory not found)")
            continue
            
        output_lines.append(f"[ {zycos_name} ]")
        
        zycos_global_hashes = set()
        
        # Sort runs so they appear in order (Run_001, Run_002, etc.)
        run_folders = sorted(glob.glob(os.path.join(zycos_dir, 'Run_*')))
        
        for run_folder in run_folders:
            run_name = os.path.basename(run_folder)
            run_hashes = set()
            
            batch_folders = glob.glob(os.path.join(run_folder, 'batch_*'))
            for batch_folder in batch_folders:
                llm_dir = os.path.join(batch_folder, 'LLM_output')
                if os.path.exists(llm_dir):
                    for net_file in glob.glob(os.path.join(llm_dir, '*.net')):
                        with open(net_file, 'r', encoding='utf-8') as f:
                            # Hash the structure without values
                            pure_hash = get_pure_topology_hash(f.read())
                            
                            # Add to specific run
                            run_hashes.add(pure_hash)
                            # Add to global zycos
                            zycos_global_hashes.add(pure_hash)
                            # Add to grand total
                            grand_total_hashes.add(pure_hash)
                            
            output_lines.append(f"  -> {run_name}: {len(run_hashes)} unique structures explored")
            
        output_lines.append(f"  >> GLOBAL FOR {zycos_name}: {len(zycos_global_hashes)} unique structures\n")
        print(f"  ✅ {zycos_name} processed.")

    output_lines.append("=" * 60)
    output_lines.append(f"GRAND TOTAL UNIQUE STRUCTURES (zycos 5-9): {len(grand_total_hashes)}")
    output_lines.append("=" * 60)

    output_path = os.path.join(SCRIPT_DIR, "pure_topology_counts.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print(f"\n🎉 Done! Results saved to: {output_path}")

if __name__ == "__main__":
    analyze_topologies()