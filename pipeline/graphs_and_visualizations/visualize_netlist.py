import json
import subprocess
import os

def flat_spice_to_json(spice_file_path, json_output_path):
    """
    Reads a raw/flat SPICE netlist, ignoring simulation commands,
    and converts physical components into a netlistsvg JSON format.
    """
    with open(spice_file_path, 'r') as f:
        lines = f.readlines()

    cells = {}
    net_connections = {}
    
    # Map SPICE prefixes to netlistsvg skins
    type_map = {
        'R': 'resistor',
        'C': 'capacitor',
        'L': 'inductor',
        'D': 'diode',
        'M': 'nmos', 
        'V': 'vsource' # Includes voltage sources in the drawing
    }

    for line in lines:
        line = line.strip().upper()
        
        # Skip empty lines, comments (*), and simulation directives (.)
        if not line or line.startswith('*') or line.startswith('.'):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        comp_name = parts[0]
        prefix = comp_name[0]
        
        # If the prefix isn't a known component, skip it
        if prefix not in type_map:
            continue
            
        # Parse standard 2-terminal components (R, C, L, D, V)
        if prefix in ['R', 'C', 'L', 'D', 'V']:
            if len(parts) >= 3:
                nodes = {'p': parts[1], 'n': parts[2]}
            else:
                continue
                
        # Parse 3+ terminal components (MOSFETs)
        elif prefix == 'M':
            if len(parts) >= 4:
                nodes = {'D': parts[1], 'G': parts[2], 'S': parts[3]}
            else:
                continue
        else:
            continue

        comp_type = type_map.get(prefix, 'box')

        # Add to our cells dictionary
        cells[comp_name] = {
            "type": comp_type,
            "connections": nodes
        }
        
        # Track wire connections
        for port_name, wire_name in nodes.items():
            if wire_name not in net_connections:
                net_connections[wire_name] = []
            net_connections[wire_name].append(comp_name)

    # Build the final JSON structure
    netlist_json = {
        "modules": {
            "AI_Generated_Circuit": {
                "ports": {}, 
                "cells": {
                    comp: {
                        "type": data["type"],
                        "port_directions": {p: "inout" for p in data["connections"].keys()},
                        "connections": {p: [wire] for p, wire in data["connections"].items()}
                    } for comp, data in cells.items()
                },
                "netnames": {
                    wire: {
                        "hide_name": 0,
                        "bits": [wire]
                    } for wire in net_connections.keys()
                }
            }
        }
    }

    # Write the JSON to a file
    with open(json_output_path, 'w') as f:
        json.dump(netlist_json, f, indent=2)
    print(f"✅ Found and parsed {len(cells)} components!")

def generate_schematic(spice_file, output_svg):
    """
    Runs the full pipeline: SPICE -> JSON -> SVG
    """
    json_file = "temp_circuit.json"
    
    # 1. Translate
    flat_spice_to_json(spice_file, json_file)
    
    # 2. Run netlistsvg via command line
    print(f"🎨 Generating SVG using netlistsvg...")
    try:
        subprocess.run(["netlistsvg", json_file, "-o", output_svg], check=True)
        print(f"🚀 Success! Schematic saved as: {output_svg}")
    except subprocess.CalledProcessError:
        print("❌ Error: netlistsvg failed to render the image.")
    except FileNotFoundError:
        print("❌ Error: 'netlistsvg' command not found. Did you run 'npm install -g netlistsvg'?")
    
    # Cleanup the temp JSON file
    if os.path.exists(json_file):
        os.remove(json_file)

# --- Run the Script ---
if __name__ == "__main__":
    # Get the exact folder where this script is currently located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up one level (..), then down into the data folder to find top4.net
    INPUT_SPICE = os.path.join(script_dir, "..", "data", "batch_2", "LLM_output", "top4.net")
    
    # Save the generated SVG right next to this python script
    OUTPUT_IMAGE = os.path.join(script_dir, "top4_schematic.svg")
    
    generate_schematic(INPUT_SPICE, OUTPUT_IMAGE)s