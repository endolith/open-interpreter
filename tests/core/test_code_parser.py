import ast

from interpreter.core.utils.code_parser import (
    parse_python_statements,
    parse_shell_commands,
    python_dotted_name,
    python_import_roots,
    python_is_safe_expr,
)


def test_parse_shell_command_name_and_args():
    """A simple command yields its word and literal arguments."""
    (command,) = parse_shell_commands("git status --short")
    assert command.name == "git"
    assert command.args == ("status", "--short")
    assert command.has_substitution is False
    assert command.redirects == ()


def test_parse_shell_chain_and_pipeline_enumerate_every_command():
    """Commands chained with && / ; and piped with | are each returned.

    The allowlist must sign off on every command that will run, not just the
    first, so a parser that only saw the leading command would be unsound.
    """
    chained = parse_shell_commands("cd /x && ls; rm -rf /")
    assert [command.name for command in chained] == ["cd", "ls", "rm"]

    piped = parse_shell_commands("cat file | wc -l")
    assert [command.name for command in piped] == ["cat", "wc"]


def test_parse_shell_command_substitution_is_flagged():
    """A command that embeds ``$(...)``/backticks is marked, including nested ones."""
    commands = {command.name: command for command in parse_shell_commands("echo $(date)")}
    assert commands["echo"].has_substitution is True
    assert commands["date"].has_substitution is False  # the nested command itself is plain

    backticked = next(command for command in parse_shell_commands("echo `date`") if command.name == "echo")
    assert backticked.has_substitution is True


def test_parse_shell_redirects_captured():
    """Redirections are reported with their operator and target."""
    (command,) = parse_shell_commands("echo hi > out.txt")
    assert command.writes_a_file is True
    assert command.redirects[0].operator == ">"
    assert command.redirects[0].target == "out.txt"

    (devnull,) = parse_shell_commands("ls 2>/dev/null")
    assert devnull.redirects[0].operator == ">"
    assert devnull.redirects[0].target == "/dev/null"


def test_parse_shell_dynamic_command_name_is_none():
    """``$cmd foo`` has no static word, so it can never match a name rule."""
    (command,) = parse_shell_commands("$cmd foo")
    assert command.name is None


def test_python_dotted_name():
    """Dotted call targets resolve; non-chains return None."""
    (expr,) = parse_python_statements("toolbox.web.search('x')")
    assert python_dotted_name(expr.value.func) == "toolbox.web.search"
    assert python_dotted_name(ast.Name(id="help")) == "help"
    (chained,) = parse_python_statements("a().b")
    assert python_dotted_name(chained.value) is None


def test_python_is_safe_expr_allows_reads():
    """Names, attributes, constants and operators over them are safe."""
    for source in ["os", "os.path", "1", '"x"', "[os, 2]", "{'a': os}", "os.path + os.sep", "f'{os.sep}'"]:
        (expr,) = parse_python_statements(source)
        assert python_is_safe_expr(expr.value), source


def test_python_is_safe_expr_rejects_code_execution():
    """Calls, lambdas, comprehensions, dunder access and await/yield are unsafe."""
    for source in ["os.system('x')", "lambda: 1", "[x for x in y]", "os.__class__", "f(os.system('x'))"]:
        (expr,) = parse_python_statements(source)
        assert not python_is_safe_expr(expr.value), source


def test_python_import_roots():
    """Imports report their top-level module; relative imports report None."""
    (plain,) = parse_python_statements("import os")
    assert python_import_roots(plain) == ("os",)

    (dotted,) = parse_python_statements("import os.path")
    assert python_import_roots(dotted) == ("os",)

    (from_import,) = parse_python_statements("from collections.abc import Mapping")
    assert python_import_roots(from_import) == ("collections",)

    (relative,) = parse_python_statements("from . import sibling")
    assert python_import_roots(relative) is None

    (assignment,) = parse_python_statements("x = 1")
    assert python_import_roots(assignment) is None


def test_parse_python_statements_syntax_error():
    """Unparseable python yields None so the caller can refuse to auto-run it."""
    assert parse_python_statements("def (") is None
    assert parse_python_statements("") == []
