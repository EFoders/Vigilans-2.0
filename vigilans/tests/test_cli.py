from __future__ import annotations

import json
from pathlib import Path

import pytest
from _support import fixture

from vigilans.cli import main


def test_validate_passes_good_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "good.jsonl"
    path.write_text(json.dumps(fixture("valid", "bearing-measured")) + "\n", "utf-8")
    assert main(["validate", str(path)]) == 0
    assert "1 of 1 observation(s) valid; 0 failed" in capsys.readouterr().out


def test_validate_fails_loudly_with_line_numbers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "mixed.jsonl"
    bad = fixture("invalid", "unlabelled-error")["message"]
    path.write_text(
        json.dumps(fixture("valid", "bearing-measured")) + "\n" + json.dumps(bad) + "\n{oops\n", "utf-8"
    )
    assert main(["validate", str(path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL mixed.jsonl:2" in out and "basis" in out
    assert "FAIL mixed.jsonl:3" in out
    assert "1 of 3 observation(s) valid; 2 failed" in out


def test_record_then_run_the_file_strictly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    observations = tmp_path / "m.observations.jsonl"
    picture = tmp_path / "m.picture.jsonl"
    assert main(["record", "--scenario", "mixed", "--out", str(observations)]) == 0
    code = main(
        [
            "run",
            "--file",
            str(observations),
            "--fast",
            "--record",
            str(picture),
            "--strict",
            "--run-id",
            "cli-test",
        ]
    )
    out = capsys.readouterr()
    assert code == 1  # the scenario's LEGACY-DF faults are rejected, and --strict says so
    assert "run cli-test (contracts" in out.out  # the header came first
    assert "4 rejected" in out.out
    assert "--strict" in out.err
    assert picture.is_file()


def test_run_header_states_where_settings_came_from(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--scenario", "mixed", "--fast", "--duration", "5", "--run-id", "h"]) == 0
    out = capsys.readouterr().out
    assert "inputs:   sim scenario mixed (seed 1)  [--scenario]" in out
    assert "picture:  not served  [default]" in out
    assert "CoT:      not built or published" in out
    assert "co-located emitters on one channel are one entity" in out


def test_bad_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "v.toml"
    path.write_text("[run]\nbogus = 1\n", "utf-8")
    assert main(["run", "--config", str(path), "--scenario", "mixed", "--fast"]) == 2
    assert "unknown key" in capsys.readouterr().err


def test_check_library(capsys: pytest.CaptureFixture[str]) -> None:
    library = Path(__file__).resolve().parents[1] / "libraries" / "synthetic.yaml"
    assert main(["check-library", str(library)]) == 0
    assert main(["check-library", str(library), "--private"]) == 1
