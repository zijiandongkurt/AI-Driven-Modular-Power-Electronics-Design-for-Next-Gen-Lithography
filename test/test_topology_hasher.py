import pytest
import sys
from pathlib import Path

# Ensure the pipeline is in the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.utility.topology_hasher import get_topological_hash

def test_identical_netlists():
    netlist = """
    L1 in sw 47u
    C1 out 0 10u
    M1 sw gate 0 0 nmos_w=1_l=1
    """
    hash1 = get_topological_hash(netlist)
    hash2 = get_topological_hash(netlist)
    assert hash1 == hash2, "Identical netlists must produce the same hash"

def test_line_swapping():
    netlist1 = """
    L1 in sw 47u
    C1 out 0 10u
    """
    netlist2 = """
    C1 out 0 10u
    L1 in sw 47u
    """
    assert get_topological_hash(netlist1) == get_topological_hash(netlist2), "Line order should not affect the hash"

def test_internal_node_renaming():
    # Renaming 'sw' to 'n1' (internal nodes)
    netlist_sw = """
    L1 in sw 47u
    C1 out sw 10u
    """
    netlist_n1 = """
    L1 in n1 47u
    C1 out n1 10u
    """
    assert get_topological_hash(netlist_sw) == get_topological_hash(netlist_n1), "Internal node renaming should not change the hash"

def test_fixed_node_strictness():
    # Swapping 'in' and 'out' (fixed external nodes) MUST change the hash
    netlist1 = "L1 in sw 47u"
    netlist2 = "L1 out sw 47u"
    assert get_topological_hash(netlist1) != get_topological_hash(netlist2), "Changing external fixed nodes must alter the hash"

def test_symmetrical_pin_swapping():
    # For a resistor (R), swapping the two pins shouldn't physically matter
    netlist1 = "R1 in out 10k"
    netlist2 = "R1 out in 10k"
    assert get_topological_hash(netlist1) == get_topological_hash(netlist2), "Symmetrical component pin-swapping should not alter the hash"

def test_directional_pin_swapping():
    # For a MOSFET (M), swapping the Drain and Gate completely breaks the physics
    netlist1 = "M1 drain gate source bulk nmos"
    netlist2 = "M1 gate drain source bulk nmos"
    assert get_topological_hash(netlist1) != get_topological_hash(netlist2), "Directional component pin-swapping must alter the hash"

def test_component_value_changes():
    # 47u vs 22u
    netlist1 = "L1 in sw 47u"
    netlist2 = "L1 in sw 22u"
    assert get_topological_hash(netlist1) != get_topological_hash(netlist2), "Different component values must alter the hash"

def test_unicode_sanitization():
    # The LLM hallucinated an arrow (→) - the hasher must not crash
    netlist_clean = "L1 in sw 47u"
    netlist_dirty = "L1 in sw 47u → 50u"
    
    try:
        hash_dirty = get_topological_hash(netlist_dirty)
        hash_clean = get_topological_hash(netlist_clean)
        # It should strip the weird text and hash safely
        assert isinstance(hash_dirty, str)
    except Exception as e:
        pytest.fail(f"Hasher crashed on unicode characters: {e}")

def test_inline_comments():
    netlist_clean = "L1 in sw 47u"
    netlist_commented = "L1 in sw 47u ; The LLM decided to explain this inductor"
    
    assert get_topological_hash(netlist_clean) == get_topological_hash(netlist_commented), "Inline comments must be stripped and not alter the hash"

def test_unrecognized_components():
    netlist_with_bjt = """
    L1 in sw 47u
    C1 out 0 10u
    Q1 sw out 0 2N2222
    """
    with pytest.raises(ValueError, match="CRITICAL: Validator leaked an illegal component"):
        get_topological_hash(netlist_with_bjt)
        
def test_multiline_spice():
    netlist_single = "M1 sw gate 0 0 nmos w=1 l=1"
    netlist_multi = "M1 sw gate 0 0 nmos\n+ w=1 l=1"
    
    assert get_topological_hash(netlist_single) == get_topological_hash(netlist_multi), "Multi-line (+) statements must be merged and parsed correctly"

def test_formatting_variations():
    netlist_neat = "R1 in out 10k"
    netlist_messy = "  r1 \t IN   OuT \t\t 10K  "
    
    assert get_topological_hash(netlist_neat) == get_topological_hash(netlist_messy), "Whitespace, tabs, and capitalization must not alter the hash"

