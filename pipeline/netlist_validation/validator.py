import re
import networkx as nx
from collections import defaultdict
from pathlib import Path


class validator():
    def __init__(self):
        self.BASE_DIR  = Path(__file__).parent
        self.DATA_DIR  = self.BASE_DIR.parent / "data"
        self.VALID_PREFIXES = set("VRLCDM")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parseLines(self, netlist):
        """Return component lines and directive lines separately, stripping comments and merging multiline statements."""
        component_lines = []
        directive_lines = []
        
        # Pre-Processing: Strip comments and merge '+' lines
        merged_lines = []
        for raw in netlist.splitlines():
            # Strip inline comments (everything after ';')
            line = raw.split(';')[0].strip()
            
            if not line or line.startswith("*"):
                continue
                
            if line.startswith("+"):
                if merged_lines:
                    merged_lines[-1] += " " + line[1:].strip()
            else:
                merged_lines.append(line)

        # Categorize the cleaned lines
        for line in merged_lines:
            if line.startswith("."):
                directive_lines.append(line)
            else:
                component_lines.append(line)
                
        return component_lines, directive_lines
    
    def _buildGraph(self, component_lines):
        """Build a NetworkX MultiGraph where nodes are SPICE nodes and edges are components.
        Each edge carries ref and prefix as attributes.
        MOSFETs produce edges between all terminal pairs.
        """
        G = nx.MultiGraph()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            ref = parts[0].upper()
            prefix = ref[0]
            if prefix == "M":
                if len(parts) < 5:
                    continue
                terminals = parts[1:5]  # drain, gate, source, bulk (standard LTspice order)
            else:
                terminals = parts[1:3]
            seen = set()
            for i, a in enumerate(terminals):
                for b in terminals[i+1:]:
                    if (a, b) not in seen:
                        G.add_edge(a, b, ref=ref, prefix=prefix)
                        seen.add((a, b))
        return G

    def _buildZeroImpedanceGraph(self, component_lines):
        """Build a multi-graph containing only zero-impedance edges."""
        G = nx.MultiGraph()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            ref = parts[0].upper()
            node_a, node_b = parts[1].lower(), parts[2].lower()
            
            if ref[0] == "R" and len(parts) >= 4 and parts[3] in {"0", "0.0"}:
                G.add_edge(node_a, node_b, ref=ref, prefix="R")
                
            # Allow ALL voltage sources (including VIN) to be evaluated for loops
            if ref[0] == "V":
                G.add_edge(node_a, node_b, ref=ref, prefix="V")
        return G

    def _hasFloatingNodes(self, G):
        """Return True if any non-GND node has only reactive/switch connections."""
        GROUNDING = {"R", "V", "I", "D"}
        for node in G.nodes:
            if node == "0":
                continue
            prefixes = {d["prefix"] for _, _, d in G.edges(node, data=True)}
            if prefixes and prefixes.isdisjoint(GROUNDING):
                return True
        return False

    # ── public ───────────────────────────────────────────────────────────────

    def validate(self, batchID):
        """Validate all .net files in data/<batchID>/llm_output/.

        Injects .save and title comment into passing netlists in-place.
        Writes validation_results.json to data/<batchID>/.

        Args:
            batchID (str): The ID of a batch.

        Returns:
            dict: Mapping of filename to ``(passed, checklist)`` where
                ``passed`` is bool and ``checklist`` is a dict of check
                name to bool.
        """
        import json

        batch_dir = self.DATA_DIR / batchID / "LLM_output"
        assert batch_dir.exists(), f"Batch folder not found: {batch_dir.resolve()}"

        net_files = list(batch_dir.glob("*.net"))
        assert net_files, f"No .net files found in: {batch_dir}"

        results     = {}
        json_output = {}

        for net_path in net_files:
            raw_text  = net_path.read_text()
            checklist = self._validateOne(raw_text)
            passed    = all(checklist.values())

            for check, result in checklist.items():
                if not result:
                    print(f"[{net_path.name}] Failed: {check}")
            if passed:
                print(f"[{net_path.name}] Validation passed.")
                # Inject title comment if missing
                if not raw_text.lstrip().startswith("*"):
                    raw_text = f"* {net_path.stem}\n" + raw_text
                # Inject .save to ensure all currents and voltages are in .raw output
                if ".save" not in raw_text.lower():
                    raw_text = raw_text.replace(".end", ".save V(*) I(*)\n.end") #LTSPICE
                    #raw_text = raw_text.replace(".end", ".probe V(*) I(*)\n.end") #NGSPICE
                net_path.write_text(raw_text)

            results[net_path.name]     = (passed, checklist)
            json_output[net_path.stem] = {"passed": passed, "checks": checklist}

        # Write validation_results.json into the batch folder
        json_path = self.DATA_DIR / batchID / "validation_results.json"
        json_path.write_text(json.dumps(json_output, indent=4))

        print(f"Results saved to {json_path}")
        print(f"Valid:   {sum(v['passed'] for v in json_output.values())}/{len(json_output)}")
        print(f"Invalid: {sum(not v['passed'] for v in json_output.values())}/{len(json_output)}")

        return results

    def _validateOne(self, netlist):
        return {
            "netlist_syntax":        self._checkNetlistSyntax(netlist),
            "ground_reference":      self._checkGroundReference(netlist),
            "power_supply":          self._checkPowerSupply(netlist),
            "naming_conventions":    self._checkNamingConventions(netlist),
            "invalid_components":    self._checkInvalidComponents(netlist),
            "duplicate_refs":        self._checkDuplicateRefs(netlist),
            "component_values":      self._checkComponentValues(netlist),
            "mosfet_bulk":           self._checkMosfetBulk(netlist),
            "model_declarations":    self._checkModelDeclarations(netlist),
            "simulation_parameters": self._checkSimulationParameters(netlist),
            "gate_drive_present":    self._checkGateDrivePresent(netlist),
            "gate_per_mosfet":       self._checkGatePerMosfet(netlist),
            "connected_graph":       self._checkConnectedGraph(netlist),
            "floating_nodes":        self._checkFloatingNodes(netlist),
            "cl_loops":              self._checkCLLoops(netlist),
            "current_path":          self._checkCurrentPath(netlist),
            "short_same_node":       self._checkShortSameNode(netlist),
            "short_zero_resistor":   self._checkShortZeroResistor(netlist),
            "short_zero_voltage":    self._checkShortZeroVoltage(netlist),
            "singular_voltage_cap":   self._checkSingularVoltageCap(netlist),
            "short_input_to_gnd":    self._checkShortInputToGnd(netlist),
            "kvl_voltage_loop":      self._checkKvlVoltageLoop(netlist),
            "singular_cap_loop":     self._checkSingularCapLoop(netlist),
        }

    # ── checks ────────────────────────────────────────────────────────────────

    def _checkNetlistSyntax(self, netlist):
        """Every component line must have a valid prefix, at least ref + 2 nodes,
        and netlist must end with .end.
        """
        component_lines, _ = self._parseLines(netlist)
        if not netlist.strip().upper().endswith(".END"):
            return False
        for line in component_lines:
            parts = line.split()
            if parts[0][0].upper() not in self.VALID_PREFIXES:
                return False
            if len(parts) < 3:
                return False
        return True

    def _checkGroundReference(self, netlist):
        """Node 0 must appear in the netlist."""
        component_lines, _ = self._parseLines(netlist)
        G = self._buildGraph(component_lines)
        return "0" in G.nodes

    def _checkPowerSupply(self, netlist):
        """Vin must exist with positive terminal on node 'in'."""
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            parts = line.split()
            if parts[0].upper() == "VIN":
                return len(parts) >= 3 and parts[1].lower() == "in"
        return False

    def _checkNamingConventions(self, netlist):
        """Required nodes and component refs per naming convention doc."""
        component_lines, _ = self._parseLines(netlist)
        refs  = {p.split()[0].upper() for p in component_lines if p.split()}
        G     = self._buildGraph(component_lines)
        nodes = set(G.nodes)

        if not {"in", "0", "out"}.issubset(nodes):
            return False
        if not {"VIN", "RLOAD"}.issubset(refs):
            return False
        if not any(re.fullmatch(r"sw\d*", n.lower()) for n in nodes):
            return False
        if not any(re.fullmatch(r"gate\d*", n.lower()) for n in nodes):
            return False
        return True

    def _checkInvalidComponents(self, netlist):
        """All component refs must start with a known SPICE prefix."""
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            if line.split()[0][0].upper() not in self.VALID_PREFIXES:
                return False
        return True

    def _checkDuplicateRefs(self, netlist):
        """Component reference designators must be unique."""
        component_lines, _ = self._parseLines(netlist)
        refs = [line.split()[0].upper() for line in component_lines]
        return len(refs) == len(set(refs))

    def _checkComponentValues(self, netlist):
        """Passives (R, L, C) must have a positive, parseable numeric value."""
        SPICE_VAL = re.compile(
            r"^[+]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+|Meg|[TGMKkmunpf]u?)?$"
        )
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            parts = line.split()
            ref = parts[0].upper()
            if ref[0] in {"R", "L", "C"}:
                if len(parts) < 4:
                    return False
                val = parts[3]
                if not SPICE_VAL.match(val):
                    return False
                try:
                    if ref[0] in {"L", "C"} and float(re.sub(r"[TGMKkmunpf]u?$", "", val)) == 0:
                        return False
                except ValueError:
                    return False
        return True

    def _checkMosfetBulk(self, netlist):
        """Every MOSFET must declare drain, gate, source, bulk and model — 6 tokens min."""
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            parts = line.split()
            if parts[0][0].upper() == "M" and len(parts) < 6:
                return False
        return True

    def _checkModelDeclarations(self, netlist):
        """Every model referenced by MOSFETs and diodes must have a .model directive."""
        component_lines, directive_lines = self._parseLines(netlist)

        declared = set()
        for d in directive_lines:
            parts = d.split()
            if parts[0].lower() == ".model" and len(parts) >= 2:
                declared.add(parts[1].upper())

        for line in component_lines:
            parts = line.split()
            ref = parts[0].upper()
            if ref[0] == "M":
                model_tokens = [p for p in parts[5:] if "=" not in p]
                if model_tokens and model_tokens[0].upper() not in declared:
                    return False
            if ref[0] == "D" and len(parts) >= 4:
                if parts[3].upper() not in declared:
                    return False
        return True

    def _checkSimulationParameters(self, netlist):
        """Must have .tran with timestep and stop time, standardised MOSFET model,
        and valid Vgate PULSE with 7 params.
        """
        component_lines, directive_lines = self._parseLines(netlist)

        tran_ok = any(
            d.lower().startswith(".tran") and len(d.split()) >= 3
            for d in directive_lines
        )
        if not tran_ok:
            return False

        model_ok = any(
            re.search(r"NMOS\s*NMOS\s*\(Vto=1\s+Kp=2\s+Lambda=0\)", d, re.IGNORECASE)
            for d in directive_lines
        )
        if not model_ok:
            return False

        for line in component_lines:
            parts = line.split()
            if re.match(r"VGATE(_HI|_LO)?\d*$", parts[0], re.IGNORECASE):
                pulse_match = re.search(r"PULSE\s*\(([^)]+)\)", line, re.IGNORECASE)
                if not pulse_match or len(pulse_match.group(1).split()) != 7:
                    return False

        return True

    def _checkGateDrivePresent(self, netlist):
        """At least one Vgate source must be present."""
        component_lines, _ = self._parseLines(netlist)
        return any(
            re.match(r"VGATE(_HI|_LO)?\d*$", line.split()[0], re.IGNORECASE)
            for line in component_lines
        )

    def _checkGatePerMosfet(self, netlist):
        """Each MOSFET gate node must be driven by a Vgate, AND every Vgate must drive a MOSFET."""
        component_lines, _ = self._parseLines(netlist)
        gate_nodes   = set()
        driven_nodes = set()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3: continue
            
            ref = parts[0].upper()
            if ref[0] == "M" and len(parts) >= 5:
                gate_nodes.add(parts[2].lower())  # Drain Gate Source Bulk
            if re.match(r"VGATE(_HI|_LO)?\d*$", ref, re.IGNORECASE):
                driven_nodes.add(parts[1].lower())
                
        # The sets must be exactly identical. No missing drives, no zombie drives.
        return gate_nodes == driven_nodes

    def _checkConnectedGraph(self, netlist):
        """All nodes must form a single connected graph, and no nodes can be dangling dead-ends."""
        component_lines, _ = self._parseLines(netlist)
        G = self._buildGraph(component_lines)

        if len(G.nodes) == 0:
            return False

        # 1. No isolated islands
        if not nx.is_connected(G):
            return False
            
        # 2. No dangling open circuits (nodes with only 1 connection)
        for node, degree in G.degree():
            if degree < 2:
                return False
                
        return True
    
    def _checkFloatingNodes(self, netlist):
        """Any node connected only to reactive/switch elements needs an Rbleed."""
        component_lines, _ = self._parseLines(netlist)
        G = self._buildGraph(component_lines)
        return not self._hasFloatingNodes(G)

    def _checkCLLoops(self, netlist):
        """A node connected exclusively to inductors and capacitors is unstable."""
        component_lines, _ = self._parseLines(netlist)
        G = self._buildGraph(component_lines)
        for node in G.nodes:
            prefixes = {d["prefix"] for _, _, d in G.edges(node, data=True)}
            if prefixes and prefixes.issubset({"L", "C"}):
                return False
        return True

    def _checkCurrentPath(self, netlist):
        """A valid current path must exist: in -> out and out -> 0."""
        component_lines, _ = self._parseLines(netlist)
        G = self._buildGraph(component_lines)
        try:
            # 1. Output must have a return path to ground
            if not nx.has_path(G, "out", "0"):
                return False
                
            # 2. Input must reach output THROUGH components, not by backtracking through ground
            G_no_gnd = G.copy()
            if "0" in G_no_gnd:
                G_no_gnd.remove_node("0")
                
            return nx.has_path(G_no_gnd, "in", "out")
        except nx.NodeNotFound:
            return False

    # ── short circuit checks ──────────────────────────────────────────────────

    def _checkShortSameNode(self, netlist):
        """Both terminals of a component must be different nodes."""
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[1].lower() == parts[2].lower():
                return False
        return True

    def _checkShortZeroResistor(self, netlist):
        """No resistor may have a value of zero."""
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            parts = line.split()
            if parts[0][0].upper() == "R" and len(parts) >= 4:
                if parts[3] in {"0", "0.0"}:
                    return False
        return True

    def _checkShortZeroVoltage(self, netlist):
        """No DC voltage source may have a value of zero — this is a dead short."""
        component_lines, _ = self._parseLines(netlist)
        for line in component_lines:
            parts = line.split()
            if parts[0][0].upper() == "V" and len(parts) >= 4:
                if parts[3] in {"0", "0.0"} and not re.search(
                    r"PULSE|SIN|PWL|AC", line, re.IGNORECASE
                ):
                    return False
        return True

    def _checkSingularVoltageCap(self, netlist):
        component_lines, _ = self._parseLines(netlist)
        v_pairs = set()
        c_pairs = set()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            ref = parts[0].upper()
            pair = frozenset([parts[1].lower(), parts[2].lower()])
            
            # Catch ALL voltage sources
            if ref[0] == "V":
                v_pairs.add(pair)
            if ref[0] == "C":
                c_pairs.add(pair)

        return len(v_pairs & c_pairs) == 0

    def _checkShortInputToGnd(self, netlist):
        component_lines, _ = self._parseLines(netlist)
        G_zero = self._buildZeroImpedanceGraph(component_lines)

        # Remove the intended power supply so it doesn't flag itself as a short circuit
        edges_to_remove = [(u, v, k) for u, v, k, d in G_zero.edges(data=True, keys=True) if d.get("ref") == "VIN"]
        G_zero.remove_edges_from(edges_to_remove)

        if "in" not in G_zero.nodes or "0" not in G_zero.nodes:
            return True
        return not nx.has_path(G_zero, "in", "0")

    def _checkKvlVoltageLoop(self, netlist):
        """Two or more voltage sources forming a closed loop violates KVL —
        LTspice will fail with a singular matrix.
        """
        component_lines, _ = self._parseLines(netlist)
        G_zero = self._buildZeroImpedanceGraph(component_lines)

        # 1. Detect 2-Node Cycles (Parallel Voltage Sources)
        # G_zero is a MultiGraph. We check if multiple V sources exist between the exact same two nodes.
        for u, v in set(G_zero.edges(keys=False)):
            if G_zero.number_of_edges(u, v) >= 2:
                # Count how many of these parallel edges are voltage sources
                v_count = sum(1 for attrs in G_zero[u][v].values() if attrs.get("prefix") == "V")
                if v_count >= 2:
                    return False

        # 2. Detect 3+ Node Cycles (Large KVL Loops)
        G_simple = nx.Graph(G_zero)
        for cycle in nx.cycle_basis(G_simple):
            cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
            v_count = 0
            for a, b in cycle_edges:
                edge_data = G_zero.get_edge_data(a, b) or G_zero.get_edge_data(b, a)
                if edge_data:
                    # If ANY of the parallel edges in this segment is a V, count it
                    if any(attrs.get("prefix") == "V" for attrs in edge_data.values()):
                        v_count += 1
            if v_count >= 2:
                return False
                
        return True

    def _checkSingularCapLoop(self, netlist):
        component_lines, _ = self._parseLines(netlist)

        # MUST use MultiGraph to detect parallel capacitors
        G_cap = nx.MultiGraph()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0][0].upper() == "C":
                G_cap.add_edge(parts[1].lower(), parts[2].lower())

        # Mathematical cycle detection for MultiGraphs: Cycles = Edges - Nodes + Connected Components
        num_cycles = G_cap.number_of_edges() - G_cap.number_of_nodes() + nx.number_connected_components(G_cap)
        return num_cycles == 0

# validator = validator()
# validation_results = validator.validate(batchID="batch_2")