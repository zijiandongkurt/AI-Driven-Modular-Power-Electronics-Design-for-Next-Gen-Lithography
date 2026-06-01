import networkx as nx

def get_topological_hash(netlist_text: str) -> str:
    """
    Converts a SPICE netlist into a mathematical graph and returns a strict topological hash.
    Impervious to line-swapping, internal node renaming, and pin-swapping on symmetrical components.
    """
    G = nx.Graph()
    lines = [l.strip().lower() for l in netlist_text.split('\n')]
    
    # We lock external constraints so the AI can't cheat by swapping input/output
    fixed_nets = {'0', 'in', 'out', 'gate'}
    
    comp_counter = 0
    for line in lines:
        if not line or line.startswith('*') or line.startswith('.'):
            continue
            
        parts = line.split()
        if len(parts) < 3: continue
        
        comp_type = parts[0][0] # r, l, c, m, d, v
        
        if comp_type == 'm' and len(parts) >= 5: # MOSFETs
            nets = parts[1:5]
            value = "_".join(parts[5:])
        elif comp_type in ['r', 'l', 'c', 'd', 'v'] and len(parts) >= 3:
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
            
            if comp_type in ['m', 'd', 'v']:
                # Directional components: Pin order strictly matters
                pin_node = f"{comp_node}_pin_{pin_idx}"
                G.add_node(pin_node, label=f"pin_{pin_idx}")
                G.add_edge(comp_node, pin_node)
                G.add_edge(pin_node, net)
            else:
                # Symmetrical components (R, L, C): Pin order doesn't matter physically
                G.add_edge(comp_node, net)

    # Generate the Weisfeiler-Lehman graph hash based on the physical topology
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='label')