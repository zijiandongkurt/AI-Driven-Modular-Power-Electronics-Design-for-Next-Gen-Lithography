import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.netlist_validation.validator import validator

@pytest.fixture
def val():
    """Provide a fresh validator instance for each test."""
    v = validator()
    # Ensure the test suite uses our patched, strict component list
    v.VALID_PREFIXES = set("VRLCDM")
    return v

# --- BASELINE: The Perfect Netlist ---
VALID_BUCK_NETLIST = """* 12V to 5V Buck Converter
Vin in 0 12
M1 in gate sw 0 NMOS W=1 L=1
D1 0 sw DIODE
L1 sw out 47u
C1 out 0 220u
Rload out 0 10
Vgate gate 0 PULSE(0 12 0 1n 1n 4.16u 10u)
.model NMOS NMOS(Vto=1 Kp=2 Lambda=0)
.model DIODE D
.tran 1u 10m
.end
"""

def test_perfect_netlist(val):
    """Verify that a standard valid buck converter passes all validation checks."""
    checklist = val._validateOne(VALID_BUCK_NETLIST)
    assert all(checklist.values()), f"Perfect netlist failed: {[k for k, v in checklist.items() if not v]}"

# --- 1. SYNTAX & COMPONENT CHECKS ---

def test_banned_components(val):
    """Verify that BJTs (Q) and current sources (I) are strictly rejected."""
    bad_netlist = VALID_BUCK_NETLIST.replace("D1 0 sw DIODE", "Q1 0 sw 0 2N2222")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["invalid_components"], "Banned components (Q) must fail validation"

def test_missing_end(val):
    """Verify that netlists without .end are rejected."""
    bad_netlist = VALID_BUCK_NETLIST.replace(".end", "")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["netlist_syntax"], "Missing .end must fail syntax check"

def test_missing_model_declaration(val):
    """Verify that using a MOSFET or Diode without a .model directive fails."""
    bad_netlist = VALID_BUCK_NETLIST.replace(".model DIODE D\n", "")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["model_declarations"], "Missing .model directive must fail"

# --- 2. TOPOLOGICAL & NAMING CHECKS ---

def test_missing_ground(val):
    """Verify that node '0' (ground reference) is strictly required."""
    bad_netlist = VALID_BUCK_NETLIST.replace(" 0 ", " gnd ")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["ground_reference"], "Missing node '0' must fail"

def test_missing_vin_or_rload(val):
    """Verify that Vin and Rload naming conventions are strictly required."""
    bad_netlist = VALID_BUCK_NETLIST.replace("Vin", "V1")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["naming_conventions"], "Missing Vin naming convention must fail"

def test_broken_current_path(val):
    """Verify that a broken physical path from 'in' to 'out' to '0' fails the KCL check."""
    # Break the path by disconnecting the inductor from the output
    bad_netlist = VALID_BUCK_NETLIST.replace("L1 sw out 47u", "L1 sw n1 47u")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["current_path"], "Broken current path must fail KCL check"

# --- 3. GRAPH THEORY & SINGULAR MATRIX CHECKS ---

def test_kvl_voltage_loop(val):
    """Verify that two voltage sources forming a closed loop (KVL violation) are caught."""
    # Add a second voltage source directly in parallel with Vin
    bad_netlist = VALID_BUCK_NETLIST + "V2 in 0 12\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["kvl_voltage_loop"], "Voltage source loops (KVL violation) must fail"

def test_singular_cap_loop(val):
    """Verify that a loop made entirely of capacitors (singular matrix) is caught."""
    # Add two capacitors in parallel
    bad_netlist = VALID_BUCK_NETLIST + "C2 out 0 100u\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["singular_cap_loop"], "Capacitor-only loops must fail"

def test_voltage_cap_parallel(val):
    """Verify that a voltage source directly across a capacitor (singular initial condition) is caught."""
    # Connect a capacitor directly across Vin
    bad_netlist = VALID_BUCK_NETLIST + "C2 in 0 10u\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["singular_voltage_cap"], "Capacitors in parallel with Voltage sources must fail"

def test_cl_loop_instability(val):
    """Verify that a node connected only to inductors and capacitors is flagged as unstable."""
    # Connect L and C to a floating middle node 'n1'
    bad_netlist = VALID_BUCK_NETLIST.replace("L1 sw out 47u", "L1 sw n1 47u\nC2 n1 out 10u")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["cl_loops"], "LC-only intermediate nodes must fail"

# --- 4. SHORT CIRCUIT CHECKS ---

def test_zero_ohm_resistor(val):
    """Verify that resistors with a value of 0 ohms fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace("Rload out 0 10", "Rload out 0 0.0")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["short_zero_resistor"], "0-ohm resistors must fail"

def test_input_short_to_ground(val):
    """Verify that a zero-impedance path from 'in' to ground is caught."""
    # Add a wire (0V source) from 'in' to '0'
    bad_netlist = VALID_BUCK_NETLIST + "Vshort in 0 0\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["short_input_to_gnd"], "Dead shorts from input to ground must fail"

def test_same_node_short(val):
    """Verify that components with both pins on the same node fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace("L1 sw out 47u", "L1 sw sw 47u")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["short_same_node"], "Components shorted to their own pins must fail"

# --- 5. COMPONENT VALUES & SYNTAX TRAPS ---

def test_duplicate_refs(val):
    """Verify that duplicate component reference names fail validation."""
    # Add a second capacitor also named 'C1'
    bad_netlist = VALID_BUCK_NETLIST + "C1 sw 0 10u\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["duplicate_refs"], "Duplicate component references must fail"

def test_negative_component_values(val):
    """Verify that passive components with negative values fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace("47u", "-47u")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["component_values"], "Negative component values must fail"

def test_zero_l_c_values(val):
    """Verify that capacitors and inductors with a value of 0 fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace("220u", "0u")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["component_values"], "0-value capacitors/inductors must fail"

def test_gibberish_values(val):
    """Verify that non-numeric (gibberish) component values fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace("220u", "two_hundred_uF")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["component_values"], "Gibberish text values must fail"

def test_malformed_mosfet(val):
    """Verify that MOSFETs missing required pins or a model declaration fail validation."""
    # We must replace the ENTIRE line so W=1 L=1 doesn't trick the token counter
    bad_netlist = VALID_BUCK_NETLIST.replace("M1 in gate sw 0 NMOS W=1 L=1", "M1 in gate sw")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["mosfet_bulk"], "MOSFETs missing bulk nodes or models must fail"


# --- 6. SIMULATION & MODEL TRAPS ---

def test_missing_tran(val):
    """Verify that netlists without a .tran directive fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace(".tran 1u 10m\n", "")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["simulation_parameters"], "Missing .tran must fail"

def test_undeclared_model(val):
    """Verify that components referencing an undeclared model fail validation."""
    # Change the MOSFET model from NMOS to PMOS, but don't add a .model PMOS line
    bad_netlist = VALID_BUCK_NETLIST.replace("M1 in gate sw 0 NMOS", "M1 in gate sw 0 PMOS")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["model_declarations"], "Undeclared models must fail"

def test_malformed_pulse(val):
    """Verify that a PULSE statement with fewer than 7 parameters fails validation."""
    # Remove the last two parameters (t_on and period)
    bad_netlist = VALID_BUCK_NETLIST.replace("PULSE(0 12 0 1n 1n 4.16u 10u)", "PULSE(0 12 0 1n 1n)")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["simulation_parameters"], "Malformed PULSE statements must fail"

def test_missing_gate_drive(val):
    """Verify that netlists without at least one gate drive voltage source fail validation."""
    bad_netlist = VALID_BUCK_NETLIST.replace("Vgate gate 0 PULSE(0 12 0 1n 1n 4.16u 10u)\n", "")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["gate_drive_present"], "Missing Vgate source must fail"

def test_undriven_mosfet_gate(val):
    """Verify that every MOSFET gate pin must be driven by a Vgate source."""
    # The Vgate drives 'gate', but we'll connect the MOSFET to a floating 'gate2'
    bad_netlist = VALID_BUCK_NETLIST.replace("M1 in gate sw 0 NMOS", "M1 in gate2 sw 0 NMOS")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["gate_per_mosfet"], "Undriven MOSFET gates must fail"


# --- 7. ADVANCED TOPOLOGY TRAPS ---

def test_disconnected_island(val):
    """Verify that isolated sub-circuits completely detached from the main circuit fail validation."""
    # Add a random resistor floating between two arbitrary, unconnected nodes
    bad_netlist = VALID_BUCK_NETLIST + "R_island n1 n2 10k\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["connected_graph"], "Disconnected graph islands must fail"

def test_floating_node_no_bleed(val):
    """Verify that nodes connected only to capacitors or inductors without a DC path to ground fail."""
    # Split the output capacitor into two capacitors in series, creating floating node 'n_float'
    bad_netlist = VALID_BUCK_NETLIST.replace("C1 out 0 220u", "C1a out n_float 110u\nC1b n_float 0 110u")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["floating_nodes"], "Floating nodes without a bleed resistor must fail"

def test_zero_dc_voltage_source(val):
    """Verify that DC voltage sources set to 0V (equivalent to a wire) fail validation."""
    # Note: 0V is allowed for PULSE/SIN sources, but not pure DC sources.
    bad_netlist = VALID_BUCK_NETLIST + "V_test sw out 0\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["short_zero_voltage"], "0V DC sources must fail"

def test_missing_sw_naming_convention(val):
    """Verify that node naming conventions (sw, gate) are strictly enforced."""
    # Rename 'sw' to 'n1' everywhere
    bad_netlist = VALID_BUCK_NETLIST.replace(" sw ", " n1 ")
    checklist = val._validateOne(bad_netlist)
    assert not checklist["naming_conventions"], "Missing 'sw' node name must fail"

# --- 8. THE BOSS LEVEL TRAPS (PARSING & GRAPH LIMITATIONS) ---

def test_validator_multiline_support(val):
    """Verify that the validator correctly stitches SPICE '+' continuation lines together."""
    # Break the MOSFET into two lines using standard SPICE '+' syntax
    bad_netlist = VALID_BUCK_NETLIST.replace(
        "M1 in gate sw 0 NMOS W=1 L=1", 
        "M1 in gate sw 0 NMOS\n+ W=1 L=1"
    )
    checklist = val._validateOne(bad_netlist)
    assert checklist["invalid_components"], "Validator crashed on a '+' line continuation"
    assert checklist["mosfet_bulk"], "Validator lost the MOSFET parameters on the second line"

@pytest.mark.parametrize("commented_value", [
    "47u ; The LLM used spaces before and after",
    "47u;The LLM forgot spaces completely",
    "47u ;No space after the semicolon",
    "47u; Space after the semicolon but not before",
    "47u\t;\tTabs instead of spaces"
])
def test_validator_inline_comments(val, commented_value):
    """Verify the validator strips inline comments regardless of spacing around the semicolon."""
    # Inject our specific string permutation into the netlist
    bad_netlist = VALID_BUCK_NETLIST.replace("47u", commented_value)
    checklist = val._validateOne(bad_netlist)
    
    # We add a custom error message so if one specific format fails, pytest tells us exactly which one it was!
    assert checklist["component_values"], f"Validator failed to parse value because of comment format: '{commented_value}'"

def test_zombie_gate_drive(val):
    """Verify that Vgate sources not connected to any MOSFET gate fail validation."""
    # Add a perfectly valid gate drive, but connect it to a node ('ghost_gate') that no MOSFET uses.
    bad_netlist = VALID_BUCK_NETLIST + "Vgate2 ghost_gate 0 PULSE(0 12 0 1n 1n 4u 10u)\n"
    checklist = val._validateOne(bad_netlist)
    assert not checklist["gate_per_mosfet"], "Zombie gate drives (not attached to any MOSFET) must fail"

def test_dangling_open_circuit(val):
    """Verify that dangling open-circuit nodes (degree 1) fail validation."""
    # Add a resistor connected to the output, but the other pin goes to a node ('nowhere') that nothing else touches.
    # Because it is connected to 'out', it passes the Island check, but it is physically useless!
    bad_netlist = VALID_BUCK_NETLIST + "R_dangle out nowhere 10k\n"
    checklist = val._validateOne(bad_netlist)
    # A node with only 1 connection (degree of 1) is a dead end.
    assert not checklist["connected_graph"], "Dangling open-circuit nodes (Degree=1) must fail"