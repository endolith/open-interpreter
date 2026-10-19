"""Decide which code may run unattended in ``auto_run == "allowlist"`` mode.

Rules are *structural*: bash rules name a command (and optionally its leading
arguments), python rules name a call or an import. Code is parsed (see
``code_parser``) and matched against those shapes, so a rule can never be
fooled by quoting, reflowing or an embedded ``$(...)`` the way a text pattern
can. A block runs without asking only when *every* command/statement in it is
covered and passes the built-in safety guards.

The threat model is accidental destruction, not a determined attacker: rules
may be broad (``git``, ``ls``) because the guards still stop the destructive
*forms* (command substitution, writes outside the null device, sensitive
paths, interpreter one-liners).
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

# Commands a structural rule can never cover: language runtimes and wrappers
# that would let a rule smuggle in arbitrary code (`python -c`, `xargs`), shells
# and evaluators that re-parse their input, network fetchers, privilege tools,
# and primitives whose whole job is to overwrite or destroy. They always ask.
NEVER_ALLOW_COMMANDS = frozenset(
    {
        ".", "source", "eval", "exec", "command", "builtin", "trap",
        "sh", "bash", "zsh", "dash", "ash", "ksh", "fish", "csh", "tcsh", "busybox",
        "python", "python2", "python3", "pypy", "perl", "ruby", "node", "nodejs",
        "php", "lua", "r", "rscript", "julia", "deno", "bun",
        "awk", "gawk", "mawk", "sed",
        "env", "xargs", "sudo", "su", "doas", "pkexec",
        "timeout", "nohup", "nice", "time", "stdbuf",
        "export", "unset", "set", "shopt", "alias", "unalias", "declare", "typeset", "readonly", "ulimit",
        "dd", "mkfs", "chmod", "chown", "chgrp", "mount", "umount", "fdisk", "parted",
        "kill", "pkill", "killall", "reboot", "shutdown", "halt", "poweroff", "systemctl", "service",
        "curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "rsync", "telnet", "ftp",
        "osascript", "powershell", "pwsh", "cmd",
    }
)

# Paths a command argument may never name: system/secret locations whose
# reading or writing is the classic accidental-destruction case.
_SENSITIVE_PATH_PREFIXES = (
    "/etc", "/root", "/proc", "/sys", "/boot",
    "~/.ssh", "~/.aws", "~/.gnupg", "~/.kube", "~/.docker", "~/.netrc", "~/.git-credentials",
)

# Argument tokens that are unsafe for any command (a filesystem root, a home
# directory or a bare traversal), independent of which command they belong to.
_DANGEROUS_ARG_LITERALS = frozenset({"/", "/*", "~", "~/", "~/*"})

# Flags that turn a read command into a file writer (``sort -o``, ``git
# --output=``). Narrow on purpose: per-binary flag vetting is out of scope.
_DANGEROUS_FLAGS = ("--output", "--outfile", "--out=")

_WRITE_OPERATORS = frozenset({">", ">>", "&>", ">&", ">|", "<>"})
_NULL_TARGETS = frozenset({"/dev/null", "nul", "$null"})

_READ_ONLY_SHELL_COMMANDS = (
    "ls", "pwd", "cat", "head", "tail", "wc", "grep", "which", "whoami", "date",
    "echo", "file", "stat", "du", "df", "uniq", "cut", "tr", "basename", "dirname",
    "realpath", "true", "false",
)
_READ_ONLY_GIT_SUBCOMMANDS = ("status", "log", "diff", "show")
_SAFE_PYTHON_CALLS = ("print", "help", "dir", "type", "len", "vars", "repr", "id")

BUILTIN_ALLOWLIST_RULES = (
    [{"language": "bash", "command": name} for name in _READ_ONLY_SHELL_COMMANDS]
    + [{"language": "bash", "command": "git", "args": [sub]} for sub in _READ_ONLY_GIT_SUBCOMMANDS]
    + [{"language": "python", "call": name} for name in _SAFE_PYTHON_CALLS]
    + [{"language": "python", "import": "*"}]
)

_PYTHON_ARG_KINDS = ("object", "string", "number", "literal")
_PYTHON_KIND_LABELS = {"object": "<object>", "string": "<string>", "number": "<number>", "literal": "<value>"}


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


def _validate_rule(rule, source="allowlist"):
    if not isinstance(rule, dict):
        raise ValueError(f"Invalid rule in {source}: expected mapping, got {type(rule).__name__}")
    language = (rule.get("language") or "").lower()
    if not language:
        raise ValueError(f"Rule in {source} missing 'language'")
    if language in SHELL_LANGUAGES:
        if not isinstance(rule.get("command"), str) or not rule["command"]:
            raise ValueError(f"Shell rule in {source} needs a 'command'")
        if "args" in rule and not isinstance(rule["args"], list):
            raise ValueError(f"'args' in {source} must be a list")
        return
    if language == "python":
        keys = [key for key in ("call", "import") if key in rule]
        if len(keys) != 1:
            raise ValueError(f"Python rule in {source} needs exactly one of 'call' or 'import'")
        if "args" in rule and "call" not in rule:
            raise ValueError(f"'args' in {source} only applies to a python 'call' rule")
        return
    raise ValueError(f"Unsupported rule language {language!r} in {source}")


def _load_rules_from_file(path):
    path = _expand_path(path)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Allowlist file must be a YAML mapping: {path}")
    rules = data.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError(f"'rules' in allowlist file must be a list: {path}")
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


def _is_sensitive_path(value):
    candidate = value.strip().strip("'\"")
    if ".." in candidate.split("/"):
        return True
    if candidate in _DANGEROUS_ARG_LITERALS:
        return True
    expanded = os.path.expanduser(candidate)
    for prefix in _SENSITIVE_PATH_PREFIXES:
        expanded_prefix = os.path.expanduser(prefix)
        if candidate == prefix or candidate.startswith(prefix + "/"):
            return True
        if expanded == expanded_prefix or expanded.startswith(expanded_prefix + "/"):
            return True
    return False


def _shell_redirect_ok(redirect):
    target = redirect.target.strip().strip("'\"")
    if redirect.operator in _WRITE_OPERATORS:
        return target in _NULL_TARGETS
    if redirect.operator == "<":
        return not _is_sensitive_path(target)
    return True  # here-doc / here-string: inert input


def _shell_arg_ok(arg):
    if any(arg.startswith(flag) for flag in _DANGEROUS_FLAGS):
        return False
    return not _is_sensitive_path(arg)


def _shell_command_is_matchable(command):
    if command.name is None:
        return False
    if command.name.lower() in NEVER_ALLOW_COMMANDS:
        return False
    if command.has_substitution:
        return False
    if any(not _shell_redirect_ok(redirect) for redirect in command.redirects):
        return False
    return all(_shell_arg_ok(arg) for arg in command.args)


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


def _python_import_matches(stmt, rule):
    roots = python_import_roots(stmt)
    if roots is None:
        return False
    module = rule.get("import")
    if module == "*":
        return all(root in SIDE_EFFECT_FREE_IMPORTS for root in roots)
    return all(root == module for root in roots)


def _python_statement_matches(stmt, rule):
    if "call" in rule:
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and _python_call_matches(stmt.value, rule)
    if "import" in rule:
        return _python_import_matches(stmt, rule)
    return False


def is_execution_allowlisted(interpreter, language, code):
    if interpreter.auto_run_mode != "allowlist":
        return False
    rules = load_allowlist_rules(interpreter)
    lang = (language or "").lower()

    if lang in SHELL_LANGUAGES:
        commands = parse_shell_commands(code)
        if not commands:
            return False
        applicable = [rule for rule in rules if _languages_compatible(rule["language"], lang) and "command" in rule]
        return all(any(_shell_command_matches(command, rule) for rule in applicable) for command in commands)

    if lang == "python":
        statements = parse_python_statements(code)
        if not statements:
            return False
        applicable = [
            rule for rule in rules if _languages_compatible(rule["language"], lang) and ("call" in rule or "import" in rule)
        ]
        return all(any(_python_statement_matches(stmt, rule) for rule in applicable) for stmt in statements)

    return False


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
    return f"also covers every other `{command_name}` invocation (destructive arguments still ask)"


def suggest_allowlist_rules(language, code):
    """Candidate rules that would cover ``code``, narrowest first.

    Each candidate is ``{"rule", "label", "warning"}``: ``rule`` is the
    structural rule to persist, ``label`` is the human-readable shape shown to
    the user, and ``warning`` (or None) says what else a broad candidate would
    allow.     Only units that could ever be matched are offered, so a suggestion
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
                    label = f"{name}({', '.join(_PYTHON_KIND_LABELS[kind] for kind in kinds)})"
                    add({"language": "python", "call": name, "args": kinds}, label)
                add({"language": "python", "call": name}, f"{name}(...)")
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
