from interpreter.core.core import OpenInterpreter
from interpreter.core.utils.execution_allowlist import (
    is_execution_allowlisted,
    normalize_auto_run_mode,
    persist_allowlist_rule,
    should_require_execution_confirmation,
    should_require_execution_confirmation_for_code,
)
from interpreter.core.utils.shell_chain import split_shell_chain


def _interpreter(**kwargs):
    interpreter = OpenInterpreter()
    for key, value in kwargs.items():
        setattr(interpreter, key, value)
    return interpreter


def test_split_shell_chain_basic():
    """`cd /x && ls` splits into its two chained commands on the && operator."""
    assert split_shell_chain("cd /x && ls") == ["cd /x", "ls"]


def test_split_shell_chain_all_operators():
    """All chain operators (&&, ||, ;, &) split at top level."""
    assert split_shell_chain("a && b || c; d & e") == ["a", "b", "c", "d", "e"]


def test_split_shell_chain_no_space_operator():
    """An operator directly adjacent to a command (`cd /x;ls`) still splits."""
    assert split_shell_chain("cd /x;ls") == ["cd /x", "ls"]
    assert split_shell_chain("cd /x&&ls") == ["cd /x", "ls"]


def test_split_shell_chain_quoted_operator_not_split():
    """An operator inside quotes is part of the command, not a split point."""
    assert split_shell_chain('echo "a && b"') == ['echo "a && b"']


def test_split_shell_chain_escaped_space_not_split():
    """An escaped space (`\\ `) stays inside one command; a real `&&` still splits."""
    assert split_shell_chain(r"cd /x\ with\ space && ls") == ["cd /x\\ with\\ space", "ls"]


def test_split_shell_chain_pipe_not_split():
    """A bare pipe is not a chain operator — the whole pipeline stays one command."""
    assert split_shell_chain("cd /x | wc") == ["cd /x | wc"]


def test_split_shell_chain_drops_empty_pieces():
    """A dangling operator (`cd /x &&`) yields just the real command."""
    assert split_shell_chain("cd /x &&") == ["cd /x"]


def test_split_shell_chain_custom_operators():
    """The operator set is configurable — cmd has no `||`/`;`, so those don't split."""
    assert split_shell_chain("a && b; c", operators=("&&", "&")) == ["a", "b; c"]
    assert split_shell_chain("a && b; c", operators=(";",)) == ["a && b", "c"]


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


def test_builtin_allowlist_exact_matches():
    interpreter = _interpreter(auto_run_mode="allowlist")

    assert is_execution_allowlisted(interpreter, "bash", "ls")
    assert is_execution_allowlisted(interpreter, "shell", "ls")
    assert is_execution_allowlisted(interpreter, "cmd", "dir")
    assert is_execution_allowlisted(interpreter, "python", "help(os)")

    assert not is_execution_allowlisted(interpreter, "bash", "ls -la")
    assert not is_execution_allowlisted(interpreter, "bash", "ls; rm -rf /")
    assert not is_execution_allowlisted(interpreter, "python", "help(json)")
    assert not is_execution_allowlisted(interpreter, "python", "print(1)")


def test_allowlist_disabled_outside_allowlist_mode():
    interpreter = _interpreter(auto_run_mode="prompt")
    assert not is_execution_allowlisted(interpreter, "bash", "ls")


def test_should_require_execution_confirmation_for_code():
    interpreter = _interpreter(auto_run_mode="all")
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "rm -rf /") is False

    interpreter.auto_run_mode = "allowlist"
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "ls") is False
    assert should_require_execution_confirmation_for_code(interpreter, "bash", "pwd") is True

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
        "content": {"type": "code", "format": "bash", "content": "pwd"},
    }
    edit = {
        "type": "confirmation",
        "format": "edit",
        "content": {"format": "python", "content": "x = 1", "target": "a.py"},
    }

    assert should_require_execution_confirmation(interpreter, allowlisted) is False
    assert should_require_execution_confirmation(interpreter, blocked) is True
    assert should_require_execution_confirmation(interpreter, edit) is True


def test_persist_allowlist_rule(tmp_path):
    allowlist_file = tmp_path / "allowlist.yaml"
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_file=str(allowlist_file),
    )

    rule, added = persist_allowlist_rule(interpreter, "bash", "pwd")
    assert added is True
    assert rule["pattern"] == "pwd"

    assert is_execution_allowlisted(interpreter, "bash", "pwd")

    _, added_again = persist_allowlist_rule(interpreter, "bash", "pwd")
    assert added_again is False

    assert allowlist_file.is_file()


def test_profile_rule_merge():
    interpreter = _interpreter(
        auto_run_mode="allowlist",
        auto_run_allowlist_rules=[
            {"language": "bash", "match": "exact", "pattern": "pwd"},
        ],
    )
    assert is_execution_allowlisted(interpreter, "bash", "pwd")
    assert is_execution_allowlisted(interpreter, "bash", "ls")
