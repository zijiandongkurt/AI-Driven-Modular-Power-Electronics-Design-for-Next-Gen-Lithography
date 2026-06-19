import networkx as nx
import re

def _normalize_spice_value(val_str: str) -> str:
    """Convert equivalent SPICE value formats to uniform scientific notation.

    Handles formats such as ``47u``, ``47uF``, ``47e-6``, and ``0.047m``.

    Args:
        val_str (str): Raw SPICE value string.

    Returns:
        str: Strictly formatted scientific notation string (e.g. ``4.7000e-05``),
            or the original string for non-numeric tokens like ``DIODE``.
    """
    val_str = val_str.lower()
    
    # Extract the numeric part and the scale/unit part (e.g., '47' and 'uf')
    match = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(.*)$", val_str)
    if not match:
        return val_str # Fallback for non-numeric labels (like 'PULSE(0' or 'NMOS')
        
    number_part, suffix = match.groups()
    try:
        val = float(number_part)
    except ValueError:
        return val_str
        
    # Standard SPICE multipliers
    multipliers = {
        't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
        'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
    }
    
    # Apply multiplier if found
    for prefix, mult in multipliers.items():
        if suffix.startswith(prefix):
            val *= mult
            break
            
    # Return a strictly formatted scientific notation string
    return f"{val:.4e}"

def get_topological_hash(netlist_text: str) -> str:
    """Convert a SPICE netlist into a topological hash via graph isomorphism.

    Impervious to line-swapping, internal node renaming, pin-swapping, inline
    comments, value formatting (``47u`` vs ``4.7e-5``), and parameter ordering.

    Args:
        netlist_text (str): Full SPICE netlist text.

    Returns:
        str: Weisfeiler-Lehman graph hash string uniquely identifying the
            circuit topology.
    """
    # 1. Sanitize unicode
    netlist_text = netlist_text.encode('ascii', 'ignore').decode('ascii').lower()
    
    # 2. Pre-Processing Pass (Strip comments and merge multiline)
    raw_lines = netlist_text.split('\n')
    merged_lines = []
    
    for raw in raw_lines:
        line = raw.split(';')[0].strip()
        if not line or line.startswith('*'):
            continue
        if line.lower().startswith('.end'):
            break  # SPICE ignores everything after .end; so do we
        if line.startswith('.'):
            continue
        if line.startswith('+'):
            if merged_lines:
                merged_lines[-1] += " " + line[1:].strip()
        else:
            merged_lines.append(line)
            
    # 3. Build the Mathematical Graph
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
        
        # Intelligently extract pins and mathematically normalize values
        if comp_type == 'm': 
            nets = parts[1:5]
            # Alphabetize parameters (W=1 L=1 becomes identical to L=1 W=1)
            params = sorted(parts[5:])
            value = "_".join(params)
        elif comp_type in ['r', 'l', 'c', 'd', 'v']: 
            nets = parts[1:3]
            # FIX: Normalize and capture ALL parameters after the nets (e.g. PULSE args)
            if len(parts) > 3:
                normalized_params = [_normalize_spice_value(p) for p in parts[3:]]
                value = "_".join(normalized_params)
            else:
                value = ""
        else:
            continue
            
        comp_node = f"comp_{comp_counter}"
        comp_counter += 1
        G.add_node(comp_node, label=f"{comp_type}_{value}")
        
        for pin_idx, net in enumerate(nets):
            # Smart Node Masking
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