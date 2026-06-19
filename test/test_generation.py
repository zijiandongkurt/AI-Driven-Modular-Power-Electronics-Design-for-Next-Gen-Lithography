import json
import shutil
import sys
import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline.llm_topology_generation.prompt_input import (
    load_constraints,
    load_constraint,
    make_prompt,
    slug,
    SYSTEM_PROMPT,
)
from pipeline.llm_topology_generation.net_writer import (
    write_single_netlist,
    write_netlists,
    get_llm_output_dir,
)


#  Shared fixtures 

SAMPLE_CONSTRAINT = {
    "_comment": "Phase 1: Standard 12V to 5V Logic (Buck)",
    "vin_min": 10,
    "vin_max": 14,
    "vout_target": 5,
    "efficiency_target": 0.80,
    "power_in": 15,
}

DUMMY_NETLIST = "* Buck\nVin in 0 12\nRload out 0 10\n.end"


@pytest.fixture
def constraint_json(tmp_path):
    """Write a minimal two-entry constraint JSON file and return its path."""
    data = [SAMPLE_CONSTRAINT, {**SAMPLE_CONSTRAINT, "_comment": "Phase 2"}]
    p = tmp_path / "constraints.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


#  prompt_input tests 

class TestLoadConstraints:
    def test_returns_list(self, constraint_json):
        result = load_constraints(constraint_json)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_non_list_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a JSON list"):
            load_constraints(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_constraints(tmp_path / "nonexistent.json")


class TestLoadConstraint:
    def test_loads_first_entry(self, constraint_json):
        c = load_constraint(constraint_json, idx=0)
        assert c["vout_target"] == 5

    def test_loads_second_entry(self, constraint_json):
        c = load_constraint(constraint_json, idx=1)
        assert c["_comment"] == "Phase 2"

    def test_out_of_range_raises(self, constraint_json):
        with pytest.raises(IndexError):
            load_constraint(constraint_json, idx=99)


class TestMakePrompt:
    def test_system_prompt_is_included(self):
        prompt = make_prompt(SAMPLE_CONSTRAINT)
        assert SYSTEM_PROMPT in prompt

    def test_private_fields_are_stripped(self):
        prompt = make_prompt(SAMPLE_CONSTRAINT)
        assert "_comment" not in prompt

    def test_public_fields_are_present(self):
        prompt = make_prompt(SAMPLE_CONSTRAINT)
        assert "vout_target" in prompt
        assert "efficiency_target" in prompt

    def test_prompt_ends_with_netlist_marker(self):
        prompt = make_prompt(SAMPLE_CONSTRAINT)
        assert prompt.strip().endswith("### SPICE Netlist:")

    def test_constraint_values_serialized(self):
        prompt = make_prompt(SAMPLE_CONSTRAINT)
        assert "5" in prompt      # vout_target
        assert "0.8" in prompt    # efficiency_target


class TestSlug:
    def test_slug_includes_index(self):
        s = slug(SAMPLE_CONSTRAINT, 3)
        assert s.startswith("03_")

    def test_slug_sanitizes_spaces(self):
        s = slug(SAMPLE_CONSTRAINT, 0)
        assert " " not in s

    def test_slug_fallback_when_no_comment(self):
        s = slug({}, 7)
        assert s == "07_constraint_07"

    def test_slug_max_length(self):
        long_comment = {"_comment": "A" * 200}
        s = slug(long_comment, 0)
        # prefix "00_" + 50 chars max from comment
        assert len(s) <= 53


#  net_writer tests 

class TestGetLlmOutputDir:
    def test_path_structure(self):
        p = get_llm_output_dir("zycos_001/Run_001/batch_1")
        assert p.name == "LLM_output"
        assert "zycos_001" in str(p)


class TestWriteSingleNetlist:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "test.net"
        write_single_netlist(out, DUMMY_NETLIST, SAMPLE_CONSTRAINT)
        assert out.exists()

    def test_header_contains_constraint_fields(self, tmp_path):
        out = tmp_path / "test.net"
        write_single_netlist(out, DUMMY_NETLIST, SAMPLE_CONSTRAINT)
        content = out.read_text(encoding="utf-8")
        assert "vout_target" in content
        assert "vin_min" in content

    def test_private_fields_not_in_header(self, tmp_path):
        out = tmp_path / "test.net"
        write_single_netlist(out, DUMMY_NETLIST, SAMPLE_CONSTRAINT)
        content = out.read_text(encoding="utf-8")
        assert "_comment" not in content

    def test_netlist_body_present(self, tmp_path):
        out = tmp_path / "test.net"
        write_single_netlist(out, DUMMY_NETLIST, SAMPLE_CONSTRAINT)
        content = out.read_text(encoding="utf-8")
        assert ".end" in content

    def test_custom_name_in_header(self, tmp_path):
        out = tmp_path / "test.net"
        write_single_netlist(out, DUMMY_NETLIST, SAMPLE_CONSTRAINT, custom_name="g1_cand1")
        content = out.read_text(encoding="utf-8")
        assert "g1_cand1" in content

    def test_parent_dirs_created(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "test.net"
        write_single_netlist(out, DUMMY_NETLIST, SAMPLE_CONSTRAINT)
        assert out.exists()


class TestWriteNetlists:
    def test_writes_n_files(self, tmp_path):
        nets = [DUMMY_NETLIST] * 3
        paths = write_netlists(nets, SAMPLE_CONSTRAINT, label="Phase1_cons0", out_dir=tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_default_filename_pattern(self, tmp_path):
        nets = [DUMMY_NETLIST, DUMMY_NETLIST]
        paths = write_netlists(nets, SAMPLE_CONSTRAINT, label="Phase1_cons0", out_dir=tmp_path)
        names = [p.name for p in paths]
        assert "Phase1_cons0_cand1.net" in names
        assert "Phase1_cons0_cand2.net" in names

    def test_custom_names_used_in_filenames(self, tmp_path):
        nets = [DUMMY_NETLIST, DUMMY_NETLIST]
        paths = write_netlists(
            nets, SAMPLE_CONSTRAINT, label="Phase1_cons0",
            out_dir=tmp_path, custom_names=["g1_cand1", "g1_cand2"]
        )
        names = [p.name for p in paths]
        assert any("g1_cand1" in n for n in names)
        assert any("g1_cand2" in n for n in names)

    def test_custom_names_length_mismatch_raises(self, tmp_path):
        nets = [DUMMY_NETLIST, DUMMY_NETLIST]
        with pytest.raises(ValueError):
            write_netlists(
                nets, SAMPLE_CONSTRAINT, label="Phase1_cons0",
                out_dir=tmp_path, custom_names=["g1_cand1"]
            )

    def test_batch_suffix_extracted_from_batch_id(self, tmp_path):
        nets = [DUMMY_NETLIST]
        # Provide a batchID that contains batch_3 — should get _b3 suffix
        batch_dir = tmp_path / "zycos_001" / "batch_3"
        batch_dir.mkdir(parents=True)
        with patch("pipeline.llm_topology_generation.net_writer.get_llm_output_dir", return_value=batch_dir):
            paths = write_netlists(
                nets, SAMPLE_CONSTRAINT, label="Phase1_cons0",
                batchID="zycos_001/batch_3"
            )
        assert any("_b3" in p.name for p in paths)

    def test_no_batchid_or_outdir_raises(self):
        with pytest.raises(ValueError):
            write_netlists([DUMMY_NETLIST], SAMPLE_CONSTRAINT, label="Phase1_cons0")