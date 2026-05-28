import json
import subprocess
import os

def flat_spice_to_json(spice_file_path, json_output_path):
    """
    Reads a raw/flat SPICE netlist, ignoring simulation commands,
    and converts physical components into a strict netlistsvg JSON format.
    """
    with open(spice_file_path, 'r') as f:
        lines = f.readlines()

    cells = {}
    net_connections = {}
    wire_to_id = {}
    next_wire_id = 2 
    
    type_map = {
        'R': 'resistor', 'C': 'capacitor', 'L': 'inductor',
        'D': 'diode', 'M': 'nmos', 'V': 'vsource' 
    }

    for line in lines:
        line = line.strip().upper()
        
        if not line or line.startswith('*') or line.startswith('.'):
            continue
            
        parts = line.split()
        if not parts: continue
            
        comp_name = parts[0]
        prefix = comp_name[0]
        
        if prefix not in type_map: continue
            
        if prefix in ['R', 'C', 'L', 'D', 'V']:
            if len(parts) >= 3:
                nodes = {'p': parts[1], 'n': parts[2]}
            else: continue
        elif prefix == 'M':
            if len(parts) >= 4:
                nodes = {'D': parts[1], 'G': parts[2], 'S': parts[3]}
            else: continue
        else: continue

        comp_type = type_map.get(prefix, 'box')

        for port_name, wire_name in nodes.items():
            if wire_name not in wire_to_id:
                wire_to_id[wire_name] = next_wire_id
                next_wire_id += 1
            
            if wire_name not in net_connections:
                net_connections[wire_name] = []
            net_connections[wire_name].append(comp_name)

        # --- FIX: Forced Strict Port Direction Mapping ---
        # We MUST assign strict input/output directions so netlistsvg knows 
        # how to draw the generic boxes for passives.
        port_dirs = {}
        if prefix == 'M':
            # Gate and Drain act as inputs to the block, Source as output
            port_dirs = {'G': 'input', 'D': 'input', 'S': 'output'}
        elif prefix in ['R', 'C', 'L', 'D', 'V']:
            # Arbitrarily force 'p' to input and 'n' to output so they render
            port_dirs = {'p': 'input', 'n': 'output'}

        cells[comp_name] = {
            "type": comp_type,
            "connections": nodes,
            "port_directions": port_dirs
        }

    # --- 1. INJECT EXTERNAL PORTS ---
    circuit_ports = {}
    if "IN" in wire_to_id:
        circuit_ports["V_in"] = {
            "direction": "input", 
            "bits": [wire_to_id["IN"]]
        }
    if "OUT" in wire_to_id:
        circuit_ports["V_out"] = {
            "direction": "output", 
            "bits": [wire_to_id["OUT"]]
        }

    # --- 2. INJECT GROUND SYMBOL FOR NODE '0' ---
    if "0" in wire_to_id:
        cells["GND_AUTO"] = {
            "type": "gnd",
            "connections": {"p": "0"},
            "port_directions": {"p": "output"}
        }

    # Build the strict JSON structure
    netlist_json = {
        "modules": {
            "AI_Generated_Circuit": {
                "ports": circuit_ports, 
                "cells": {
                    comp: {
                        "type": data["type"],
                        "port_directions": data["port_directions"],
                        "connections": {p: [wire_to_id[wire]] for p, wire in data["connections"].items()}
                    } for comp, data in cells.items()
                },
                "netnames": {
                    wire: {
                        "hide_name": 0,
                        "bits": [wire_to_id[wire]],
                        "attributes": {"src": wire}
                    } for wire in net_connections.keys()
                }
            }
        }
    }

    with open(json_output_path, 'w') as f:
        json.dump(netlist_json, f, indent=2)
    print(f"✅ Found and parsed {len(cells)} components!")

def generate_schematic(spice_file, output_svg):
    json_file = "temp_circuit.json"
    
    flat_spice_to_json(spice_file, json_file)
    
    print(f"🎨 Generating SVG using netlistsvg...")
    try:
        subprocess.run(f"netlistsvg {json_file} -o {output_svg}", check=True, shell=True)
        
        # --- NEW AUTOMATED BACKGROUND FIX ---
        # 1. Open the newly generated SVG file
        with open(output_svg, 'r') as f:
            svg_content = f.read()
            
        # 2. Inject a white background directly into the root <svg> tag
        svg_content = svg_content.replace('<svg ', '<svg style="background-color: white;" ')
        
        # 3. Save it back
        with open(output_svg, 'w') as f:
            f.write(svg_content)
        # ------------------------------------
            
        print(f"🚀 Success! Schematic with white background saved as: {output_svg}")
        
    except subprocess.CalledProcessError:
        print("❌ Error: netlistsvg failed to render the image.")
    
    if os.path.exists(json_file):
        os.remove(json_file)

# --- Run the Script (Batch Processing) ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Define the folder where all the AI netlists live
    netlist_dir = os.path.join(script_dir, "..", "data", "batch_6", "LLM_output")
    
    # --- NEW FOLDER LOGIC ---
    # Get the parent batch directory (e.g., "batch_2")
    batch_dir = os.path.dirname(netlist_dir)
    
    # Create a new folder named "schematics" inside the batch directory
    svg_output_dir = os.path.join(batch_dir, "schematics")
    os.makedirs(svg_output_dir, exist_ok=True)
    # ------------------------
    
    # Check if the directory actually exists to prevent crashes
    if not os.path.exists(netlist_dir):
        print(f"❌ Error: Cannot find directory at {netlist_dir}")
    else:
        print(f"🔍 Scanning {netlist_dir} for netlists...\n")
        
        # 2. Loop through every file in that folder
        for filename in os.listdir(netlist_dir):
            # Only process files that end with .net
            if filename.endswith(".net"):
                print(f"⚙️ Processing: {filename}")
                
                # Construct the full path to the input .net file
                input_spice = os.path.join(netlist_dir, filename)
                
                # Create a matching output name and save it in the newly created folder
                base_name = os.path.splitext(filename)[0] 
                output_image = os.path.join(svg_output_dir, f"{base_name}_schematic.svg")
                
                # 3. Run the generator for this specific file
                generate_schematic(input_spice, output_image)
                print("-" * 40) # Print a divider line for terminal readability
                
        print(f"✅ Batch processing complete! All SVGs generated in: {svg_output_dir}")