import networkx as nx

def get_topological_hash(netlist_text: str) -> str:
    """
    Converts a SPICE netlist into a mathematical graph and returns a strict topological hash.
    Impervious to line-swapping, internal node renaming, pin-swapping, and inline comments.
    Enforces strict component synchronization with the upstream Validator.
    """
    # 1. Sanitize LLM output (strip weird unicode characters like arrows/emojis)
    netlist_text = netlist_text.encode('ascii', 'ignore').decode('ascii').lower()
    
    # 2. Pre-Processing Pass: Clean comments and merge multi-line statements
    raw_lines = netlist_text.split('\n')
    merged_lines = []
    
    for line in raw_lines:
        # Strip inline comments (everything after ';')
        line = line.split(';')[0].strip()
        
        # Ignore empty lines or full-line comments
        if not line or line.startswith('*') or line.startswith('.'):
            continue
            
        # Merge multi-line statements starting with '+'
        if line.startswith('+'):
            if merged_lines:
                # Append to the previous line
                merged_lines[-1] += " " + line[1:].strip()
        else:
            merged_lines.append(line)
            
    # 3. Build the Mathematical Graph
    G = nx.Graph()
    fixed_nets = {'0', 'in', 'out', 'gate'}
    comp_counter = 0
    
    # MUST perfectly match validator.py's VALID_PREFIXES
    allowed_prefixes = {'v', 'r', 'l', 'c', 'd', 'm'}
    
    for line in merged_lines:
        parts = line.split()
        if len(parts) < 3: 
            continue
        
        comp_type = parts[0][0] # 'r', 'l', 'm', etc.
        
        # Strict Architectural Boundary: Catch Validator Leaks
        if comp_type not in allowed_prefixes:
            raise ValueError(f"CRITICAL: Validator leaked an illegal component into the Hasher: '{comp_type.upper()}' from line: '{line}'")
        
        # Intelligently extract pins and values based on known, valid component types
        if comp_type == 'm': # MOSFETs typically have 4 pins
            nets = parts[1:5]
            value = "_".join(parts[5:])
        elif comp_type in ['r', 'l', 'c', 'd', 'v']: # Standard 2-pin components
            nets = parts[1:3]
            value = "_".join(parts[3:])
        else:
            continue
            
        # Create a unique node for this specific component instance
        comp_node = f"comp_{comp_counter}"
        comp_counter += 1
        
        # Label it with its type and value (e.g., 'l_47u' or 'm_nmos_w=1_l=1')
        G.add_node(comp_node, label=f"{comp_type}_{value}")
        
        for pin_idx, net in enumerate(nets):
            # Mask internal nodes so renaming 'sw' to 'n1' doesn't change the hash
            net_label = net if net in fixed_nets else "internal_net"
            G.add_node(net, label=net_label)
            
            # Symmetrical components (R, L, C): Pin order doesn't matter physically
            if comp_type in ['r', 'l', 'c']:
                G.add_edge(comp_node, net)
            # Directional components (M, D, V): Pin order strictly matters
            else:
                pin_node = f"{comp_node}_pin_{pin_idx}"
                G.add_node(pin_node, label=f"pin_{pin_idx}")
                G.add_edge(comp_node, pin_node)
                G.add_edge(pin_node, net)

    # Generate the Weisfeiler-Lehman graph hash based on the physical topology
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='label')