import re
import networkx as nx
from collections import defaultdict
from pathlib import Path


class validator():
    def __init__(self):
        self.BASE_DIR  = Path(__file__).parent
        self.DATA_DIR  = self.BASE_DIR.parent / "data"
        self.VALID_PREFIXES = set("VRICLDMQ")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parseLines(self, netlist):
        """Return component lines and directive lines separately, stripping comments."""
        component_lines = []
        directive_lines = []
        for raw in netlist.splitlines():
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
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
        """Build a simple graph containing only zero-impedance edges:
        voltage sources and zero-ohm resistors.
        Used for short circuit path detection.
        """
        G = nx.MultiGraph()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            ref = parts[0].upper()
            node_a, node_b = parts[1].lower(), parts[2].lower()
            # zero-ohm resistor
            if ref[0] == "R" and len(parts) >= 4 and parts[3] in {"0", "0.0"}:
                G.add_edge(node_a, node_b, ref=ref, prefix="R")
            # voltage sources except Vin (the supply) — Vin is intentional, not a short
            if ref[0] == "V" and ref != "VIN":
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

        PARAMS:
        batchID <string> : The ID of a batch

        RETURNS:
        results <dict> : { filename: (passed <bool>, checklist <dict>) }
        """
        import json

        batch_dir = self.DATA_DIR / batchID / "llm_output"
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
                    raw_text = raw_text.replace(".end", ".save V(*) I(*)\n.end")
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
        """Each MOSFET gate node must be driven by a Vgate source."""
        component_lines, _ = self._parseLines(netlist)
        gate_nodes   = set()
        driven_nodes = set()
        for line in component_lines:
            parts = line.split()
            ref = parts[0].upper()
            if ref[0] == "M" and len(parts) >= 5:
                gate_nodes.add(parts[2].lower())  # drain gate source bulk — standard LTspice
            if re.match(r"VGATE(_HI|_LO)?\d*$", ref, re.IGNORECASE):
                driven_nodes.add(parts[1].lower())
        return gate_nodes.issubset(driven_nodes)

    def _checkConnectedGraph(self, netlist):
        """All nodes must form a single connected graph — no isolated islands."""
        component_lines, _ = self._parseLines(netlist)
        G = self._buildGraph(component_lines)
        return nx.is_connected(G)

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
            return (
                nx.has_path(G, "in", "out") and
                nx.has_path(G, "out", "0")
            )
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
        """A voltage source directly in parallel with a capacitor (no series R between them)
        causes a singular matrix — LTspice cannot solve the initial conditions.
        Detected by finding a V and C that share exactly the same two nodes.
        """
        component_lines, _ = self._parseLines(netlist)

        # collect node pairs per prefix
        v_pairs = set()
        c_pairs = set()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            ref    = parts[0].upper()
            pair   = frozenset([parts[1].lower(), parts[2].lower()])
            if ref[0] == "V" and ref != "VIN":
                v_pairs.add(pair)
            if ref[0] == "C":
                c_pairs.add(pair)

        return len(v_pairs & c_pairs) == 0

    def _checkShortInputToGnd(self, netlist):
        """A zero-impedance path must not exist from 'in' to '0' —
        this would short the input supply.
        Detected by BFS on the zero-impedance subgraph.
        """
        component_lines, _ = self._parseLines(netlist)
        G_zero = self._buildZeroImpedanceGraph(component_lines)

        if "in" not in G_zero.nodes or "0" not in G_zero.nodes:
            return True
        return not nx.has_path(G_zero, "in", "0")

    def _checkKvlVoltageLoop(self, netlist):
        """Two or more voltage sources forming a closed loop violates KVL —
        LTspice will fail with a singular matrix.
        Detected by checking if any cycle exists in the zero-impedance subgraph
        that contains at least two voltage source edges.
        """
        component_lines, _ = self._parseLines(netlist)
        G_zero = self._buildZeroImpedanceGraph(component_lines)

        # convert to simple graph for cycle detection
        G_simple = nx.Graph(G_zero)
        for cycle in nx.cycle_basis(G_simple):
            # collect all edges in this cycle
            cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
            v_count = 0
            for a, b in cycle_edges:
                edge_data = G_zero.get_edge_data(a, b) or G_zero.get_edge_data(b, a)
                if edge_data:
                    for _, attrs in edge_data.items():
                        if attrs.get("prefix") == "V":
                            v_count += 1
            if v_count >= 2:
                return False
        return True

    def _checkSingularCapLoop(self, netlist):
        """A loop where every branch is a capacitor causes a singular matrix at t=0
        because initial conditions are undefined.
        Detected by cycle detection on a capacitor-only subgraph.
        """
        component_lines, _ = self._parseLines(netlist)

        G_cap = nx.Graph()
        for line in component_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0][0].upper() == "C":
                G_cap.add_edge(parts[1].lower(), parts[2].lower())

        return len(nx.cycle_basis(G_cap)) == 0

# validator = validator()
# validation_results = validator.validate(batchID="batch_2")