"""Tests for the allowlist candidate picker in the terminal interface.

These helpers turn the rules derived from a confirmation prompt into persisted
allowlist entries. They are pure user-interface glue, so they are exercised
here with mocked terminal input rather than through a full conversation.
"""

from types import SimpleNamespace

import yaml

from interpreter.terminal_interface.terminal_interface import (
    _choose_allowlist_candidates,
    _describe_allowlist_rule,
    _offer_allowlist_rules,
)


def _candidate(rule, label, warning=None):
    return {"rule": rule, "label": label, "warning": warning}


def _stub_interpreter(allowlist_file, replace_builtin=False):
    return SimpleNamespace(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
        auto_run_allowlist_rules=[],
        auto_run_allowlist_replace_builtin=replace_builtin,
        _session_allowlist_rules=[],
    )


def test_describe_allowlist_rule_formats_every_rule_shape():
    """Each structural rule shape renders as a readable one-line description."""
    assert (
        _describe_allowlist_rule({"language": "bash", "command": "git", "args": ["status"]})
        == "bash: git status"
    )
    assert _describe_allowlist_rule({"language": "bash", "command": "ls"}) == "bash: ls"
    assert (
        _describe_allowlist_rule({"language": "python", "call": "help", "args": ["object"]})
        == "python: help(...)"
    )
    assert (
        _describe_allowlist_rule({"language": "python", "import": "json"})
        == "python: import json"
    )


def test_choose_allowlist_candidates_returns_selection_in_order(monkeypatch):
    """Comma/space separated numbers select the matching candidates in the order given."""
    candidates = [
        _candidate({"language": "bash", "command": "a"}, "a"),
        _candidate({"language": "bash", "command": "b"}, "b"),
        _candidate({"language": "bash", "command": "c"}, "c"),
    ]
    monkeypatch.setattr("builtins.input", lambda prompt="": "3, 1")

    chosen = _choose_allowlist_candidates(candidates)

    assert [candidate["label"] for candidate in chosen] == ["c", "a"]


def test_choose_allowlist_candidates_empty_answer_selects_nothing(monkeypatch):
    """Pressing Enter (or only whitespace) adds no rules."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "   ")

    assert _choose_allowlist_candidates([_candidate({"language": "bash", "command": "a"}, "a")]) == []


def test_choose_allowlist_candidates_ignores_non_numbers_and_out_of_range(monkeypatch):
    """Tokens that are not in-range integers are skipped instead of erroring."""
    candidates = [
        _candidate({"language": "bash", "command": "a"}, "a"),
        _candidate({"language": "bash", "command": "b"}, "b"),
    ]
    monkeypatch.setattr("builtins.input", lambda prompt="": "x 0 2 99 -1")

    chosen = _choose_allowlist_candidates(candidates)

    assert [candidate["label"] for candidate in chosen] == ["b"]


def test_choose_allowlist_candidates_prints_warning(monkeypatch, capsys):
    """A broad candidate's warning is shown next to its label."""
    candidates = [_candidate({"language": "bash", "command": "git"}, "git (any arguments)", "broad")]
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    _choose_allowlist_candidates(candidates)

    assert "git (any arguments)  (broad)" in capsys.readouterr().out


def test_offer_allowlist_rules_persists_selected_rule(monkeypatch, tmp_path, capsys):
    """Selecting a candidate writes that structural rule to the allowlist file."""
    allowlist_file = tmp_path / "allowlist.yaml"
    interpreter = _stub_interpreter(allowlist_file)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    _offer_allowlist_rules(interpreter, "bash", "git status --short")

    output = capsys.readouterr().out
    assert "These rules cover the whole block." in output
    assert "Added to allowlist: bash: git status --short" in output
    assert yaml.safe_load(allowlist_file.read_text())["rules"] == [
        {"language": "bash", "command": "git", "args": ["status", "--short"]}
    ]


def test_offer_allowlist_rules_reports_still_uncovered(monkeypatch, tmp_path, capsys):
    """Selecting rules that only cover part of the block names what is left."""
    interpreter = _stub_interpreter(tmp_path / "allowlist.yaml", replace_builtin=True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")

    _offer_allowlist_rules(interpreter, "bash", "ls; pwd")

    assert "Still uncovered: pwd" in capsys.readouterr().out


def test_offer_allowlist_rules_reports_when_nothing_is_coverable(capsys):
    """Unsupported languages produce a clear message instead of an empty picker."""
    interpreter = SimpleNamespace(auto_run_allowlist_file="unused.yaml")

    _offer_allowlist_rules(interpreter, "ruby", "puts 1")

    assert "can't be covered by an allowlist rule" in capsys.readouterr().out


def test_offer_allowlist_rules_enter_persists_nothing(monkeypatch, tmp_path, capsys):
    """Skipping at the picker leaves the allowlist file untouched."""
    allowlist_file = tmp_path / "allowlist.yaml"
    interpreter = SimpleNamespace(auto_run_allowlist_file=str(allowlist_file))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    _offer_allowlist_rules(interpreter, "bash", "git status")

    assert not allowlist_file.exists()
    assert "Added to allowlist" not in capsys.readouterr().out
