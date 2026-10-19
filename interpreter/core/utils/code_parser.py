"""Parse bash and python into the shapes the allowlist reasons about.

The allowlist must decide what a snippet *does*, not what it looks like, so it
matches against a parse tree rather than the raw text. This module is the only
place that knows about tree-sitter (bash) and the stdlib ``ast`` (python); the
allowlist consumes normalised shapes and never touches a parse node.

Both languages are parsed for real. tree-sitter-bash is a declared dependency,
so there is no textual fallback: a snippet that cannot be parsed simply yields
no units, and the caller treats "nothing recognised" as "not allowed".
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import tree_sitter_bash as tsbash
from tree_sitter import Language, Parser

_BASH_LANGUAGE = Language(tsbash.language())
_BASH_PARSER = Parser(_BASH_LANGUAGE)

# A literal command word: an unquoted token without expansions. Anything else
# (``$cmd``, ``"ls"``, a concatenation) is dynamic and never matched by name.
_LITERAL_WORD = re.compile(r"[A-Za-z0-9_./:@+-]+")

# Node types that spawn a nested process (``$(...)``, backticks, ``<(...)``).
_SUBSTITUTION_NODES = frozenset({"command_substitution", "process_substitution"})

# Redirection node types and the operators that write to a file.
_REDIRECT_NODES = frozenset({"file_redirect", "heredoc_redirect", "herestring_redirect"})
_WRITE_OPERATORS = frozenset({">", ">>", "&>", ">&", ">|", "<>"})
_REDIRECT_OPERATORS = _WRITE_OPERATORS | {"<"}

_QUOTE_CHARS = "\"'`"


@dataclass(frozen=True)
class ShellRedirect:
    """A redirection attached to a command, e.g. ``> out.txt`` or ``2>/dev/null``."""

    operator: str
    target: str


@dataclass(frozen=True)
class ShellCommand:
    """One simple command that will run (a pipeline or list holds several)."""

    text: str
    name: str | None
    args: tuple[str, ...]
    has_substitution: bool
    redirects: tuple[ShellRedirect, ...]

    @property
    def writes_a_file(self) -> bool:
        """True when the command redirects output somewhere other than the null device."""
        return any(r.operator in _WRITE_OPERATORS for r in self.redirects)


def _walk(node):
    """Yield a node and its descendants in document order."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _unquoted(node) -> str:
    text = node.text.decode("utf-8", "replace")
    if node.type in ("string", "raw_string") and len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTE_CHARS:
        return text[1:-1]
    return text


def _collect_redirects(node) -> tuple[ShellRedirect, ...]:
    statement = node
    while statement is not None and statement.type != "redirected_statement":
        statement = statement.parent
    if statement is None:
        return ()
    redirects = []
    for descendant in _walk(statement):
        if descendant.type == "file_redirect":
            operator = ""
            target = ""
            for child in descendant.children:
                if child.type == "file_descriptor":
                    continue
                if child.type in _REDIRECT_OPERATORS:
                    operator = child.type
                else:
                    target = _unquoted(child)
            redirects.append(ShellRedirect(operator, target))
        elif descendant.type in ("heredoc_redirect", "herestring_redirect"):
            redirects.append(ShellRedirect("<<" if descendant.type == "heredoc_redirect" else "<<<", ""))
    return tuple(redirects)


def _build_shell_command(node) -> ShellCommand:
    name_node = node.child_by_field_name("name")
    name = None
    if name_node is not None:
        candidate = name_node.text.decode("utf-8", "replace")
        if _LITERAL_WORD.fullmatch(candidate):
            name = candidate

    # tree-sitter hands back a fresh Node object per access, so the name node
    # cannot be filtered by identity — match on its type instead.
    args = []
    for child in node.children:
        if child.type in ("command_name", "variable_assignment"):
            continue
        args.append(_unquoted(child))

    has_substitution = any(descendant.type in _SUBSTITUTION_NODES for descendant in _walk(node))

    return ShellCommand(
        text=node.text.decode("utf-8", "replace"),
        name=name,
        args=tuple(args),
        has_substitution=has_substitution,
        redirects=_collect_redirects(node),
    )


def parse_shell_commands(code: str) -> list[ShellCommand]:
    """Every simple command in ``code``, including those nested in pipelines,
    lists (``&&``, ``||``, ``;``) and substitutions. A snippet that means to run
    a command is only safe when *all* of these are allowed, so returning them
    all — rather than just the top-level line — is what makes chain handling
    structural rather than textual."""
    if not code or not code.strip():
        return []
    tree = _BASH_PARSER.parse(bytes(code, "utf-8"))
    return [_build_shell_command(node) for node in _walk(tree.root_node) if node.type == "command"]


def python_dotted_name(node) -> str | None:
    """The dotted name of a ``Name``/``Attribute`` chain, e.g. ``toolbox.web.search``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = python_dotted_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def python_is_safe_expr(node) -> bool:
    """Whether an expression can be evaluated without running user code.

    Safe means "reads a value": constants, names, non-dunder attributes, and
    containers/operators over safe expressions. Anything that *calls* or
    *defers* code — a ``Call``, lambda, comprehension, ``await``/``yield``,
    dunder access — is unsafe, so ``help(os)`` is allowed but
    ``help(os.system("rm -rf /"))`` is not.
    """
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            return False
        return python_is_safe_expr(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(python_is_safe_expr(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(python_is_safe_expr(key) for key in node.keys if key is not None) and all(
            python_is_safe_expr(value) for value in node.values
        )
    if isinstance(node, ast.UnaryOp):
        return python_is_safe_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return python_is_safe_expr(node.left) and python_is_safe_expr(node.right)
    if isinstance(node, ast.BoolOp):
        return all(python_is_safe_expr(value) for value in node.values)
    if isinstance(node, ast.Compare):
        return python_is_safe_expr(node.left) and all(python_is_safe_expr(c) for c in node.comparators)
    if isinstance(node, ast.IfExp):
        return all(python_is_safe_expr(part) for part in (node.test, node.body, node.orelse))
    if isinstance(node, ast.Subscript):
        return python_is_safe_expr(node.value) and python_is_safe_expr(node.slice)
    if isinstance(node, ast.Slice):
        return all(python_is_safe_expr(part) for part in (node.lower, node.upper, node.step) if part is not None)
    if isinstance(node, ast.JoinedStr):
        return all(python_is_safe_expr(value) for value in node.values)
    if isinstance(node, ast.FormattedValue):
        return python_is_safe_expr(node.value)
    if isinstance(node, ast.Starred):
        return python_is_safe_expr(node.value)
    return False


def python_import_roots(stmt) -> tuple[str, ...] | None:
    """Top-level module names imported by an ``import``/``from`` statement.

    ``None`` when ``stmt`` is not an import, or is a relative import (which
    cannot run from the kernel anyway). ``from a.b import c`` reports ``a``.
    """
    if isinstance(stmt, ast.Import):
        return tuple(alias.name.split(".")[0] for alias in stmt.names)
    if isinstance(stmt, ast.ImportFrom):
        if stmt.level:  # relative import — never treated as an import we can allow
            return None
        if not stmt.module:
            return None
        return (stmt.module.split(".")[0],)
    return None


def parse_python_statements(code: str) -> list[ast.stmt] | None:
    """Top-level statements of ``code``, or ``None`` when it does not parse."""
    if not code or not code.strip():
        return []
    try:
        return ast.parse(code).body
    except SyntaxError:
        return None
