import pytest

from interpreter.core.core import OpenInterpreter
from interpreter.core.utils.execution_allowlist import (
    is_execution_allowlisted,
    normalize_auto_run_mode,
    persist_allowlist_rule,
    should_require_execution_confirmation,
    should_require_execution_confirmation_for_code,
    suggest_allowlist_rules,
)


def _interpreter(**kwargs):
    interpreter = OpenInterpreter()
    for key, value in kwargs.items():
        setattr(interpreter, key, value)
    return interpreter


def test_normalize_auto_run_mode():
    assert normalize_auto_run_mode(False) == "prompt"
    assert normalize_auto_run_mode(True) == "all"
    assert normalize_auto_run_mode("allowlist") == "allowlist"
    assert normalize_auto_run_mode("all") == "all"


def test_auto_run_bool_property_backcompat():
    interpreter = OpenInterpreter()
    assert interpreter.auto_run is False
    assert interpreter.auto_run_mode == "prompt"

    interpreter.auto_run = True
    assert interpreter.auto_run is True
    assert interpreter.auto_run_mode == "all"

    interpreter.auto_run = "allowlist"
    assert interpreter.auto_run is False
    assert interpreter.auto_run_mode == "allowlist"


def test_builtin_rules_cover_read_only_commands():
    """The built-in preset allows read-only commands, with or without arguments."""
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert is_execution_allowlisted(interpreter, "bash", "ls")
    assert is_execution_allowlisted(interpreter, "bash", "ls -la")
    assert is_execution_allowlisted(interpreter, "shell", "pwd")
    assert is_execution_allowlisted(interpreter, "bash", "echo hi && ls")

    assert is_execution_allowlisted(interpreter, "bash", "git status")
    assert is_execution_allowlisted(interpreter, "bash", "git status --short")
    assert not is_execution_allowlisted(interpreter, "bash", "git push")


def test_builtin_rules_cover_safe_python():
    """Safe python calls and side-effect-free imports run; anything else asks."""
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert is_execution_allowlisted(interpreter, "python", "help(os)")
    assert is_execution_allowlisted(interpreter, "python", "print(1)")
    assert is_execution_allowlisted(interpreter, "python", "import os\nimport json")

    assert not is_execution_allowlisted(interpreter, "python", "import numpy")
    assert not is_execution_allowlisted(interpreter, "python", "x = 1")


def test_allowlist_disabled_outside_allowlist_mode():
    interpreter = _interpreter(auto_run_mode="prompt")
    assert not is_execution_allowlisted(interpreter, "bash", "ls")


def test_every_command_in_a_chain_must_be_allowed():
    """A block auto-runs only when no command in it is left uncovered."""
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert is_execution_allowlisted(interpreter, "bash", "ls && pwd")
    assert not is_execution_allowlisted(interpreter, "bash", "ls; rm -rf /")


def test_python_call_arguments_must_be_safe():
    """`help(os)` is inert, but `help(os.system(...))` evaluates a nested call."""
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert is_execution_allowlisted(interpreter, "python", "print(os.sep)")
    assert not is_execution_allowlisted(interpreter, "python", "help(os.system('rm -rf /'))")
    assert not is_execution_allowlisted(interpreter, "python", "print(open('f').read())")


def test_guards_block_substitution_and_sensitive_paths():
    """Even an allowed command name is refused for destructive argument forms."""
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert not is_execution_allowlisted(interpreter, "bash", "echo $(cat /etc/shadow)")
    assert not is_execution_allowlisted(interpreter, "bash", "cat /etc/shadow")
    assert not is_execution_allowlisted(interpreter, "bash", "cat ~/.ssh/id_rsa")
    assert not is_execution_allowlisted(interpreter, "bash", "ls ..")


def test_guards_block_file_writes_but_allow_null_device():
    """Redirects to a real file ask; redirecting to the null device is inert."""
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert not is_execution_allowlisted(interpreter, "bash", "echo hi > out.txt")
    assert is_execution_allowlisted(interpreter, "bash", "ls 2>/dev/null")


def test_interpreter_commands_can_never_be_allowlisted():
    """Wrappers like `python`/`xargs` stay un-runnable even with an explicit rule."""
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_rules=[
            {"language": "bash", "command": "python"},
            {"language": "bash", "command": "xargs"},
        ],
    )

    assert not is_execution_allowlisted(interpreter, "bash", "python script.py")
    assert not is_execution_allowlisted(interpreter, "bash", "xargs ls")
    # The preset still works alongside the refused rules.
    assert is_execution_allowlisted(interpreter, "bash", "ls")


def test_profile_rule_with_args_matches_prefix_only():
    """A `command + args` rule matches that exact leading argument sequence."""
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_rules=[{"language": "bash", "command": "git", "args": ["log"]}],
    )

    assert is_execution_allowlisted(interpreter, "bash", "git log --oneline")
    assert not is_execution_allowlisted(interpreter, "bash", "git push")


def test_empty_args_rule_matches_bare_command_only(tmp_path):
    """`args: []` means the bare command, not "any arguments".

    Omitting `args` is the broad form; an explicit empty list narrows to the
    command with no arguments, mirroring how python `args: []` matches zero
    arguments. A user who writes `args: []` must not accidentally allow every
    invocation of a mutating command.
    """
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(tmp_path / "allowlist.yaml"),
        auto_run_allowlist_replace_builtin=True,
        auto_run_allowlist_rules=[{"language": "bash", "command": "git", "args": []}],
    )

    assert is_execution_allowlisted(interpreter, "bash", "git")
    assert not is_execution_allowlisted(interpreter, "bash", "git status")


def test_malformed_file_rule_raises_instead_of_being_skipped(tmp_path):
    """A hand-edited rule with no language aborts rather than being ignored.

    Silently dropping a rule still fails closed (the code asks), but it leaves
    the user believing the file says something it does not. Fail loudly.
    """
    allowlist_file = tmp_path / "allowlist.yaml"
    allowlist_file.write_text("rules:\n  - command: ls\n", encoding="utf-8")
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
    )

    with pytest.raises(ValueError, match="missing 'language'"):
        is_execution_allowlisted(interpreter, "bash", "ls")


def test_malformed_file_shape_raises(tmp_path):
    """A file whose `rules` is not a list raises instead of loading as empty."""
    allowlist_file = tmp_path / "allowlist.yaml"
    allowlist_file.write_text("rules: oops\n", encoding="utf-8")
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
    )

    with pytest.raises(ValueError, match="must be a list"):
        is_execution_allowlisted(interpreter, "bash", "ls")


def test_invalid_profile_rule_raises(tmp_path):
    """Profile rules are hand-edited config too, so they are validated strictly."""
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(tmp_path / "allowlist.yaml"),
        auto_run_allowlist_rules=[{"language": "bash"}],
    )

    with pytest.raises(ValueError, match="needs a 'command'"):
        is_execution_allowlisted(interpreter, "bash", "ls")


def test_persist_does_not_rewrite_a_malformed_file(tmp_path):
    """Persisting a rule leaves a malformed file untouched instead of dropping entries."""
    allowlist_file = tmp_path / "allowlist.yaml"
    original = "rules:\n  - command: ls\n"
    allowlist_file.write_text(original, encoding="utf-8")
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
    )

    with pytest.raises(ValueError):
        persist_allowlist_rule(interpreter, {"language": "bash", "command": "uname"})

    assert allowlist_file.read_text(encoding="utf-8") == original
    assert interpreter._session_allowlist_rules == []


def test_should_require_execution_confirmation_for_code():
    interpreter = _interpreter(auto_run_mode="all")
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "rm -rf /") is False

    interpreter.auto_run_mode = "allowlist"
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "ls") is False
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "rm -rf /") is True

    interpreter.auto_run_mode = "prompt"
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "ls") is True


def test_should_require_execution_confirmation_chunk():
    interpreter = _interpreter(auto_run_mode="allowlist")

    allowlisted = {
        "type": "confirmation",
        "format": "execution",
        "content": {"type": "code", "format": "bash", "content": "ls"},
    }
    blocked = {
        "type": "confirmation",
        "format": "execution",
        "content": {"type": "code", "format": "bash", "content": "rm -rf /"},
    }
    edit = {
        "type": "confirmation",
        "format": "edit",
        "content": {"format": "python", "content": "x = 1", "target": "a.py"},
    }

    assert should_require_execution_confirmation(interpreter, allowlisted) is False
    assert should_require_execution_confirmation(interpreter, blocked) is True
    assert should_require_execution_confirmation(interpreter, edit) is True


def test_suggest_allowlist_rules_bash_narrowest_first():
    """Bash suggestions lead with the literal invocation, then broaden.

    The literal rule carries the exact arguments; the broad `git` rule is
    offered last and warns that it covers other subcommands.
    """
    candidates = suggest_allowlist_rules("bash", "git status --short")

    labels = [candidate["label"] for candidate in candidates]
    assert labels[0] == "git status --short"
    assert "git status" in labels

    broad = candidates[-1]
    assert broad["rule"] == {"language": "bash", "command": "git"}
    assert broad["warning"]


def test_suggest_allowlist_rules_skips_unmatchable_units():
    """Commands that the guards would always refuse are not offered."""
    assert suggest_allowlist_rules("bash", "cat /etc/shadow") == []
    assert suggest_allowlist_rules("bash", "$cmd foo") == []


def test_suggest_allowlist_rules_python_call_kinds():
    """A python call with arguments offers a typed form and a broad form."""
    candidates = suggest_allowlist_rules("python", "help(os)")

    rules = [candidate["rule"] for candidate in candidates]
    assert {"language": "python", "call": "help", "args": ["object"]} in rules
    assert {"language": "python", "call": "help"} in rules


def test_persist_allowlist_rule_round_trip(tmp_path):
    """A persisted rule is written to disk and makes the code auto-runnable."""
    allowlist_file = tmp_path / "allowlist.yaml"
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
    )

    assert not is_execution_allowlisted(interpreter, "bash", "uname -a")

    rule, added = persist_allowlist_rule(interpreter, {"language": "bash", "command": "uname"})
    assert added is True
    assert rule == {"language": "bash", "command": "uname"}

    assert is_execution_allowlisted(interpreter, "bash", "uname -a")
    assert allowlist_file.is_file()

    _, added_again = persist_allowlist_rule(interpreter, {"language": "bash", "command": "uname"})
    assert added_again is False
