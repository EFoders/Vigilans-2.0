"""Configuration that is documented is actually read, and says where it came from (spec §12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vigilans.config import ConfigError, parse_listen, resolve
from vigilans.env import load_env_file, parse_env

EXAMPLES = Path(__file__).resolve().parents[1] / "config"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "vigilans.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_precedence_flag_over_env_over_file(tmp_path: Path) -> None:
    path = _write(tmp_path, '[run]\nrate = 2\nseed = 5\n[[inputs]]\nkind = "sim"\nscenario = "mixed"\n')
    config = resolve(
        {"config": str(path), "rate": "8"}, {"VIGILANS_RATE": "4", "VIGILANS_SEED": "6"}, tmp_path
    )
    assert (config.rate, config.origins["rate"]) == (8.0, "--rate")
    assert (config.seed, config.origins["seed"]) == (6, "VIGILANS_SEED")
    assert config.origins["inputs"] == "vigilans.toml"
    only_file = resolve({"config": str(path)}, {}, tmp_path)
    assert (only_file.rate, only_file.seed) == (2.0, 5)


def test_env_names_the_inputs(tmp_path: Path) -> None:
    config = resolve({}, {"VIGILANS_SCENARIO": "mixed"}, tmp_path)
    assert config.inputs[0].scenario == "mixed"
    assert config.origins["inputs"] == "VIGILANS_SCENARIO"


def test_no_inputs_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no inputs"):
        resolve({}, {}, tmp_path)


def test_live_serves_by_default_and_offline_does_not(tmp_path: Path) -> None:
    live = resolve({"scenario": "mixed"}, {}, tmp_path)
    assert live.listen == ("127.0.0.1", 8091) and live.linger
    offline = resolve({"scenario": "mixed", "fast": True}, {}, tmp_path)
    assert offline.rate is None and offline.listen is None and not offline.linger


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[run]\nrat = 2\n", "unknown key"),
        ("[picture]\nlisten = 'nowhere'\n", "HOST:PORT"),
        ("[run]\nrate = -1\n", "positive"),
        ("[[inputs]]\nkind = 'radio'\n", "kind"),
        ("[sources.X]\nassumed_bearing_sigma_deg = 3\n", "assumption_note"),
        ("[sources.X]\ncolour = 'red'\n", "unknown key"),
        ("not toml [", "not valid TOML"),
    ],
)
def test_bad_configuration_fails_loudly(tmp_path: Path, text: str, match: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ConfigError, match=match):
        resolve({"config": str(path), "scenario": "mixed"}, {}, tmp_path)


def test_paths_in_a_file_are_relative_to_the_file(tmp_path: Path) -> None:
    (tmp_path / "conf").mkdir()
    path = tmp_path / "conf" / "v.toml"
    path.write_text(
        '[[inputs]]\nkind = "file"\npath = "../obs.jsonl"\n[picture]\nrecord = "out/p.jsonl"\n', "utf-8"
    )
    config = resolve({"config": str(path)}, {}, tmp_path)
    assert config.inputs[0].path == tmp_path / "conf" / ".." / "obs.jsonl"
    assert config.record == tmp_path / "conf" / "out" / "p.jsonl"


#: Run configurations. corrections.example.toml is an operator corrections file (ADR-0015),
#: not a run configuration; test_corrections loads it.
RUN_CONFIGS = sorted(p for p in EXAMPLES.glob("*.toml") if p.name != "corrections.example.toml")


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_shipped_example_configs_load(path: Path) -> None:
    config = resolve({"config": str(path)}, {}, path.parent)
    assert config.inputs


def test_assumed_example_declares_an_assumption() -> None:
    config = resolve({"config": str(EXAMPLES / "assumed.toml")}, {}, EXAMPLES)
    legacy = config.sources["LEGACY-DF"]
    assert legacy.assumed_bearing_sigma_deg is not None
    assert legacy.assumption_note
    assert "assumed.toml" in legacy.declared_in


def test_listen_none() -> None:
    assert parse_listen("none", "x") is None
    assert parse_listen(":9000", "x") == ("127.0.0.1", 9000)


def test_env_file_is_read_and_real_env_wins(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# comment\nVIGILANS_RATE=4\nexport VIGILANS_SEED='9'\nnot a line\n", "utf-8")
    assert parse_env(path.read_text("utf-8")) == {"VIGILANS_RATE": "4", "VIGILANS_SEED": "9"}
    environ = {"VIGILANS_RATE": "2"}
    applied = load_env_file(path, environ=environ)
    assert applied == {"VIGILANS_SEED": "9"}
    assert environ["VIGILANS_RATE"] == "2"
