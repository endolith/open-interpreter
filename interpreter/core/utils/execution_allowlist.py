import os

import yaml

from ...terminal_interface.utils.oi_dir import oi_dir

AUTO_RUN_MODES = ("prompt", "all", "allowlist")

SHELL_LANGUAGES = frozenset({"bash", "shell", "sh"})

DEFAULT_ALLOWLIST_FILE = os.path.join(oi_dir, "allowlist.yaml")

# PoC builtin preset — exact match only (see plan: v2 AST, v3 safe-chains).
BUILTIN_STRICT_RULES = [
    {"language": "bash", "match": "exact", "pattern": "ls"},
    {"language": "shell", "match": "exact", "pattern": "ls"},
    {"language": "cmd", "match": "exact", "pattern": "dir"},
    {"language": "python", "match": "exact", "pattern": "help(os)"},
]


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


def _rule_key(rule):
    return (
        rule.get("language", "").lower(),
        rule.get("match", ""),
        rule.get("pattern", ""),
    )


def _languages_compatible(rule_language, code_language):
    rule_language = (rule_language or "").lower()
    code_language = (code_language or "").lower()
    if rule_language == code_language:
        return True
    if rule_language in SHELL_LANGUAGES and code_language in SHELL_LANGUAGES:
        return True
    return False


def match_exact(code, pattern):
    return code.strip() == pattern


def _validate_rule(rule, source="allowlist"):
    if not isinstance(rule, dict):
        raise ValueError(f"Invalid rule in {source}: expected mapping, got {type(rule)}")
    match_type = rule.get("match")
    if match_type != "exact":
        raise ValueError(
            f"Unsupported match type {match_type!r} in {source}. PoC supports 'exact' only."
        )
    if not rule.get("language"):
        raise ValueError(f"Rule in {source} missing 'language'")
    if "pattern" not in rule:
        raise ValueError(f"Rule in {source} missing 'pattern'")


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
    for rule in rules:
        _validate_rule(rule, source=path)
    return rules


def load_allowlist_rules(interpreter):
    rules = []
    seen = set()

    def add_rules(rule_list):
        for rule in rule_list:
            _validate_rule(rule, "allowlist")
            key = _rule_key(rule)
            if key in seen:
                continue
            seen.add(key)
            rules.append(dict(rule))

    if interpreter.auto_run_mode != "allowlist":
        return rules

    if not getattr(interpreter, "auto_run_allowlist_replace_builtin", False):
        add_rules(BUILTIN_STRICT_RULES)

    profile_rules = getattr(interpreter, "auto_run_allowlist_rules", None) or []
    add_rules(profile_rules)

    allowlist_file = getattr(interpreter, "auto_run_allowlist_file", None) or DEFAULT_ALLOWLIST_FILE
    add_rules(_load_rules_from_file(allowlist_file))

    session_rules = getattr(interpreter, "_session_allowlist_rules", None) or []
    add_rules(session_rules)

    return rules


def is_execution_allowlisted(interpreter, language, code):
    if interpreter.auto_run_mode != "allowlist":
        return False
    rules = load_allowlist_rules(interpreter)
    for rule in rules:
        if not _languages_compatible(rule["language"], language):
            continue
        if rule["match"] == "exact" and match_exact(code, rule["pattern"]):
            return True
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


def persist_allowlist_rule(interpreter, language, code):
    rule = {
        "language": language,
        "match": "exact",
        "pattern": code.strip(),
    }
    _validate_rule(rule, "session")

    session_rules = getattr(interpreter, "_session_allowlist_rules", None)
    if session_rules is None:
        interpreter._session_allowlist_rules = []
        session_rules = interpreter._session_allowlist_rules

    key = _rule_key(rule)
    if any(_rule_key(existing) == key for existing in session_rules):
        return rule, False

    session_rules.append(dict(rule))

    allowlist_file = _expand_path(
        getattr(interpreter, "auto_run_allowlist_file", None) or DEFAULT_ALLOWLIST_FILE
    )
    os.makedirs(os.path.dirname(allowlist_file), exist_ok=True)

    file_rules = _load_rules_from_file(allowlist_file)
    if any(_rule_key(existing) == key for existing in file_rules):
        return rule, False

    file_rules.append(dict(rule))
    with open(allowlist_file, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "rules": file_rules,
            },
            file,
            default_flow_style=False,
            sort_keys=False,
        )

    return rule, True
