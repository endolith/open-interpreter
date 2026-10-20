"""Decide which code may run unattended in ``auto_run == "allowlist"`` mode.

Rules are *structural*: bash rules name a command (and optionally its leading
arguments), python rules name a call or an import. Code is parsed (see
``code_parser``) and matched against those shapes, so a rule can never be
fooled by quoting, reflowing or an embedded ``$(...)`` the way a text pattern
can. A block runs without asking only when *every* command/statement in it is
covered and passes the structural guards.

The threat model is accidental destruction, not a determined attacker: rules
may be broad (``git``, ``ls``) because the guards still stop the destructive
*forms* (command substitution, writes outside the null device). Path and
command policy belongs in a separate denylist, not here.
"""

import ast
import os

import yaml

from ...terminal_interface.utils.oi_dir import oi_dir
from .code_parser import (
    parse_python_statements,
    parse_shell_commands,
    python_dotted_name,
    python_import_roots,
    python_is_safe_expr,
)
from .safe_imports import SIDE_EFFECT_FREE_IMPORTS

AUTO_RUN_MODES = ("prompt", "all", "allowlist")

SHELL_LANGUAGES = frozenset({"bash", "shell", "sh"})

DEFAULT_ALLOWLIST_FILE = os.path.join(oi_dir, "allowlist.yaml")

_WRITE_OPERATORS = frozenset({">", ">>", "&>", ">&", ">|", "<>"})
_NULL_TARGETS = frozenset({"/dev/null", "nul", "$null"})

_SAFE_SHELL_COMMANDS = (
    "cd", "pwd", "ls", "echo", "date", "whoami", "which", "true", "false",
)
_READ_ONLY_GIT_SUBCOMMANDS = ("status", "log", "diff", "show")
_SAFE_PYTHON_CALLS = ("print", "help", "dir", "type", "len", "vars", "repr", "id", "introspect")

BUILTIN_ALLOWLIST_RULES = (
    [{"language": "bash", "command": name} for name in _SAFE_SHELL_COMMANDS]
    + [{"language": "bash", "command": "git", "args": [sub]} for sub in _READ_ONLY_GIT_SUBCOMMANDS]
    + [{"language": "python", "call": name} for name in _SAFE_PYTHON_CALLS]
    + [{"language": "python", "import": "*"}]
)

_PYTHON_ARG_KINDS = ("object", "string", "number", "literal")
_PYTHON_KIND_LABELS = {"object": "<object>", "string": "<string>", "number": "<number>", "literal": "<value>"}


def describe_python_args(kinds):
    """Render argument kinds for a human-facing rule label."""
    return ", ".join(_PYTHON_KIND_LABELS.get(kind, kind) for kind in kinds)


def normalize_auto_run_mode(value):
    if value is True or value == "all" or value == "true":
        return "all"
    if value is False or value == "prompt" or value == "false" or value is None:
        return "prompt"
    if value == "allowlist":
        return "allowlist"
    raise ValueError(
        f"Invalid auto_run mode: {value!r}. Expected prompt, all, allowlist, or a boolean."
    )


def _expand_path(path):
    if not path:
        return path
    return os.path.expanduser(path)


def _languages_compatible(rule_language, code_language):
    rule_language = (rule_language or "").lower()
    code_language = (code_language or "").lower()
    if rule_language == code_language:
        return True
    return rule_language in SHELL_LANGUAGES and code_language in SHELL_LANGUAGES


def _rule_key(rule):
    return (
        (rule.get("language") or "").lower(),
        rule.get("command"),
        tuple(rule.get("args") or ()) if "args" in rule else None,
        rule.get("call"),
        rule.get("import"),
    )


class AllowlistConfigError(ValueError):
    """A user-authored allowlist rule or file could not be understood."""


def _validate_rule(rule, source="allowlist"):
    if not isinstance(rule, dict):
        raise AllowlistConfigError(
            f"Invalid rule in {source}: expected a mapping, got {type(rule).__name__}: {rule!r}"
        )
    language_value = rule.get("language")
    if not isinstance(language_value, str) or not language_value:
        raise AllowlistConfigError(f"Rule in {source} is missing 'language': {rule!r}")
    language = language_value.lower()
    if language in SHELL_LANGUAGES:
        if not isinstance(rule.get("command"), str) or not rule["command"]:
            raise AllowlistConfigError(f"Shell rule in {source} needs a 'command': {rule!r}")
        if "args" in rule and not isinstance(rule["args"], list):
            raise AllowlistConfigError(f"'args' in {source} must be a list: {rule!r}")
        return
    if language == "python":
        keys = [key for key in ("call", "import") if key in rule]
        if len(keys) != 1:
            raise AllowlistConfigError(
                f"Python rule in {source} needs exactly one of 'call' or 'import': {rule!r}"
            )
        if "args" in rule and "call" not in rule:
            raise AllowlistConfigError(
                f"'args' in {source} only applies to a python 'call' rule: {rule!r}"
            )
        return
    raise AllowlistConfigError(f"Unsupported rule language {language!r} in {source}: {rule!r}")


def _load_rules_from_file(path):
    path = _expand_path(path)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except OSError as error:
        raise AllowlistConfigError(f"Could not read allowlist file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise AllowlistConfigError(f"Allowlist file is not valid YAML: {path}: {error}") from error
    if not isinstance(data, dict):
        raise AllowlistConfigError(f"Allowlist file must be a YAML mapping: {path}")
    rules = data.get("rules") or []
    if not isinstance(rules, list):
        raise AllowlistConfigError(f"'rules' in allowlist file must be a list: {path}")
    return rules


def load_allowlist_rules(interpreter):
    rules = []
    seen = set()

    def add_rules(rule_list, source):
        for rule in rule_list:
            _validate_rule(rule, source)
            key = _rule_key(rule)
            if key in seen:
                continue
            seen.add(key)
            rules.append(dict(rule))

    if interpreter.auto_run_mode != "allowlist":
        return rules

    if not getattr(interpreter, "auto_run_allowlist_replace_builtin", False):
        add_rules(BUILTIN_ALLOWLIST_RULES, "builtin preset")

    add_rules(getattr(interpreter, "auto_run_allowlist_rules", None) or [], "profile")

    allowlist_file = _expand_path(
        getattr(interpreter, "auto_run_allowlist_file", None) or DEFAULT_ALLOWLIST_FILE
    )
    add_rules(_load_rules_from_file(allowlist_file), f"allowlist file {allowlist_file}")

    add_rules(getattr(interpreter, "_session_allowlist_rules", None) or [], "session")

    return rules


def _is_fd_duplication(redirect):
    """`2>&1`, `>&2`, `>&-` only retarget a descriptor; no file is written."""
    if redirect.operator != ">&":
        return False
    target = redirect.target.strip()
    return target == "-" or target.isdigit()


def _shell_redirect_ok(redirect):
    if _is_fd_duplication(redirect):
        return True
    target = redirect.target.strip().strip("'\"")
    if redirect.operator in _WRITE_OPERATORS:
        return target in _NULL_TARGETS
    return True  # here-doc / here-string: inert input


def _shell_command_is_matchable(command):
    if command.name is None:
        return False
    if command.has_substitution:
        return False
    if any(not _shell_redirect_ok(redirect) for redirect in command.redirects):
        return False
    return True


def _shell_command_matches(command, rule):
    if not _shell_command_is_matchable(command):
        return False
    if command.name != rule.get("command"):
        return False
    if "args" in rule:
        required = tuple(rule["args"])
        if not required:
            # An explicit empty list means "the bare command": unlike a missing
            # `args` key (any arguments), `git` alone matches but `git status`
            # does not. This mirrors how python `args: []` matches zero args.
            return not command.args
        return command.args[: len(required)] == required
    return True


def _python_arg_kind(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return "string"
        if isinstance(node.value, bool) or node.value is None:
            return "literal"
        if isinstance(node.value, (int, float, complex)):
            return "number"
        return "literal"
    if isinstance(node, (ast.Name, ast.Attribute)):
        return "object"
    return "literal"


def _python_call_matches(call, rule):
    if python_dotted_name(call.func) != rule.get("call"):
        return False
    if not all(python_is_safe_expr(arg) for arg in call.args):
        return False
    if not all(python_is_safe_expr(keyword.value) for keyword in call.keywords):
        return False
    kinds = rule.get("args")
    if kinds is not None:
        if call.keywords or len(call.args) != len(kinds):
            return False
        if any(_python_arg_kind(arg) != kind for arg, kind in zip(call.args, kinds)):
            return False
    return True


def _python_statement_matches(stmt, rule):
    if "call" in rule:
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and _python_call_matches(stmt.value, rule)
    return False


def _python_import_covered(stmt, applicable):
    """An import is covered when every module in it has its own rule.

    ``import os, sys`` may be spread across ``{import: os}`` and
    ``{import: sys}``; a single rule never claims a module it does not name.
    ``import: "*"`` covers any set of side-effect-free modules at once.
    """
    roots = python_import_roots(stmt)
    if roots is None:
        return False
    if any(
        rule.get("import") == "*" and all(root in SIDE_EFFECT_FREE_IMPORTS for root in roots)
        for rule in applicable
    ):
        return True
    return all(any(rule.get("import") == root for rule in applicable) for root in roots)


def _python_statement_covered(stmt, applicable):
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return _python_import_covered(stmt, applicable)
    return any(_python_statement_matches(stmt, rule) for rule in applicable)


def _shell_unit_label(command):
    return " ".join([command.name or "<dynamic>", *command.args])


def code_is_covered(rules, language, code):
    """``(covered, uncovered_labels)`` for a parsed block against ``rules``.

    Bash needs every parsed command matched; python needs every top-level
    statement covered, with imports matched per module.
    """
    lang = (language or "").lower()

    if lang in SHELL_LANGUAGES:
        commands = parse_shell_commands(code)
        if not commands:
            return False, []
        applicable = [rule for rule in rules if _languages_compatible(rule["language"], lang) and "command" in rule]
        uncovered = [
            command
            for command in commands
            if not any(_shell_command_matches(command, rule) for rule in applicable)
        ]
        return not uncovered, [_shell_unit_label(command) for command in uncovered]

    if lang == "python":
        statements = parse_python_statements(code)
        if not statements:
            return False, []
        applicable = [
            rule
            for rule in rules
            if _languages_compatible(rule["language"], lang) and ("call" in rule or "import" in rule)
        ]
        uncovered = [stmt for stmt in statements if not _python_statement_covered(stmt, applicable)]
        return not uncovered, [ast.unparse(stmt) for stmt in uncovered]

    return False, []


def is_execution_allowlisted(interpreter, language, code):
    if interpreter.auto_run_mode != "allowlist":
        return False
    covered, _ = code_is_covered(load_allowlist_rules(interpreter), language, code)
    return covered


def should_require_execution_confirmation_for_code(interpreter, language, code):
    if interpreter.auto_run_mode == "all":
        return False
    if is_execution_allowlisted(interpreter, language, code):
        return False
    return True


def _execution_target_from_confirmation_chunk(chunk):
    if chunk.get("type") != "confirmation":
        return None, None
    if chunk.get("format") == "edit":
        return None, None
    content = chunk.get("content") or {}
    if chunk.get("format") == "execution" or content.get("type") == "code":
        return content.get("format"), content.get("content")
    return content.get("format"), content.get("content")


def should_require_execution_confirmation(interpreter, chunk):
    if chunk.get("type") != "confirmation":
        return True
    if chunk.get("format") == "edit":
        return True
    if interpreter.auto_run_mode == "all":
        return False
    language, code = _execution_target_from_confirmation_chunk(chunk)
    if language is None:
        return True
    return should_require_execution_confirmation_for_code(interpreter, language, code)


def _broad_warning(command_name):
    return f"also covers every other `{command_name}` invocation; only the structural guards still apply"


def suggest_allowlist_rules(language, code):
    """Candidate rules that would cover ``code``, narrowest first.

    Each candidate is ``{"rule", "label", "warning"}``: ``rule`` is the
    structural rule to persist, ``label`` is the human-readable shape shown to
    the user, and ``warning`` (or None) says what else a broad candidate would
    allow. Only units that could ever be matched are offered, so a suggestion
    is never a dead rule.
    """
    lang = (language or "").lower()
    candidates = []
    seen = set()

    def add(rule, label, warning=None):
        key = _rule_key(rule)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"rule": rule, "label": label, "warning": warning})

    if lang in SHELL_LANGUAGES:
        for command in parse_shell_commands(code):
            if not _shell_command_is_matchable(command):
                continue
            name = command.name
            if command.args:
                add({"language": "bash", "command": name, "args": list(command.args)}, f"{name} {' '.join(command.args)}")
                if not command.args[0].startswith("-"):
                    add({"language": "bash", "command": name, "args": [command.args[0]]}, f"{name} {command.args[0]}")
            add({"language": "bash", "command": name}, f"{name} (any arguments)", warning=_broad_warning(name))
        return candidates

    if lang == "python":
        for stmt in parse_python_statements(code) or []:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                name = python_dotted_name(call.func)
                if name is None:
                    continue
                if not all(python_is_safe_expr(arg) for arg in call.args):
                    continue
                if not all(python_is_safe_expr(keyword.value) for keyword in call.keywords):
                    continue
                if call.args and not call.keywords:
                    kinds = [_python_arg_kind(arg) for arg in call.args]
                    label = f"{name}({describe_python_args(kinds)})"
                    add({"language": "python", "call": name, "args": kinds}, label)
                add({"language": "python", "call": name}, f"{name}(any arguments)", warning=_broad_warning(name))
            else:
                for root in python_import_roots(stmt) or ():
                    add({"language": "python", "import": root}, f"import {root}")

    return candidates


def persist_allowlist_rule(interpreter, rule):
    """Add a structural rule to the session and the allowlist file.

    Returns ``(rule, added)``; ``added`` is False when an equivalent rule is
    already present in either place. Existing file entries are validated too,
    so a hand-edited file with a malformed rule raises rather than being
    silently rewritten without it.
    """
    _validate_rule(rule, "session")
    rule = dict(rule)

    allowlist_file = _expand_path(
        getattr(interpreter, "auto_run_allowlist_file", None) or DEFAULT_ALLOWLIST_FILE
    )
    file_rules = _load_rules_from_file(allowlist_file)
    for existing in file_rules:
        _validate_rule(existing, f"allowlist file {allowlist_file}")

    session_rules = getattr(interpreter, "_session_allowlist_rules", None)
    if session_rules is None:
        interpreter._session_allowlist_rules = []
        session_rules = interpreter._session_allowlist_rules

    key = _rule_key(rule)
    if any(_rule_key(existing) == key for existing in session_rules):
        return rule, False
    session_rules.append(dict(rule))

    if any(_rule_key(existing) == key for existing in file_rules):
        return rule, False

    directory = os.path.dirname(allowlist_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_rules.append(dict(rule))
    with open(allowlist_file, "w", encoding="utf-8") as file:
        yaml.safe_dump({"rules": file_rules}, file, default_flow_style=False, sort_keys=False)

    return rule, True
