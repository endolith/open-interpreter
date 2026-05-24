"""
file_edit.py — backend runners for the `edit` tool.

Each runner is a thin wrapper that:
  - Validates the target path (absolute, exists/doesn't-exist as required)
  - Delegates to the right binary or pure-Python I/O
  - Returns a short success string, or raises on failure (caller gets traceback/message)

Binary resolution mirrors resolve_bash.py: env-var override → PATH → Git usr/bin on Windows.
The model never constructs shell command strings; flags and temp-file hygiene live here.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..terminal.languages.resolve_bash import resolve_bash_executable

EDIT_LANGUAGES = frozenset({
    "sed", "gawk", "jq", "write",
    "yq", "poke",
    "comby", "patch",
})


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------

def _resolve_binary(env_var, candidates):
    """Return path to a binary. env_var overrides; then PATH; then Git usr/bin on Windows."""
    env = os.environ.get(env_var, "").strip()
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"{env_var} is set but not a file: {env!r}")
        return env

    for name in candidates:
        found = shutil.which(name)
        if found:
            return found

    if platform.system() == "Windows":
        # Try Git Bash's usr/bin alongside the bash executable
        try:
            bash = resolve_bash_executable()
            usr_bin = os.path.normpath(
                os.path.join(os.path.dirname(bash), "..", "usr", "bin")
            )
            for name in candidates:
                candidate = os.path.join(usr_bin, name + ".exe")
                if os.path.isfile(candidate):
                    return candidate
        except FileNotFoundError:
            pass

    raise FileNotFoundError(
        f"Could not find {candidates[0]!r}. "
        f"Install it, add to PATH, or set {env_var} to the full path."
    )


def _resolve_sed():
    return _resolve_binary("INTERPRETER_SED", ["sed"])


def _resolve_gawk():
    return _resolve_binary("INTERPRETER_GAWK", ["gawk", "awk"])


def _resolve_jq():
    return _resolve_binary("INTERPRETER_JQ", ["jq"])


def _resolve_yq():
    return _resolve_binary("INTERPRETER_YQ", ["yq"])


def _resolve_poke():
    return _resolve_binary("INTERPRETER_POKE", ["poke"])


def _resolve_comby():
    return _resolve_binary("INTERPRETER_COMBY", ["comby"])


def _comby_json_flag(comby):
    """comby 1.7+ uses -json-lines; older builds accept -json."""
    help_result = subprocess.run(
        [comby, "-help"],
        capture_output=True,
        text=True,
    )
    help_text = (help_result.stdout or "") + (help_result.stderr or "")
    if "-json-lines" in help_text:
        return "-json-lines"
    return "-json"


def _resolve_patch():
    return _resolve_binary("INTERPRETER_PATCH", ["patch"])


def _is_gnu_sed(sed_path):
    result = subprocess.run(
        [sed_path, "--version"], capture_output=True, text=True
    )
    return "GNU" in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _validate_target(target, *, must_exist):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target is required and must be a non-empty string")
    if not os.path.isabs(target):
        raise ValueError(
            "target must be an absolute path "
            "(e.g. C:\\Users\\... on Windows, /home/... on Linux/Mac)"
        )
    path = Path(target)
    if must_exist:
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {target}")
    else:
        if path.exists():
            raise FileExistsError(
                f"file already exists — use another edit language (not write): {target}"
            )


def _run_failed(lang, result):
    raise RuntimeError(
        (result.stderr or result.stdout or "").strip()
        or f"{lang} exited with code {result.returncode}"
    )


def _atomic_replace_from_stdout(target, stdout_bytes):
    path = Path(target)
    fd, tmp = tempfile.mkstemp(
        suffix=path.suffix, prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(fd)
    try:
        Path(tmp).write_bytes(stdout_bytes)
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_write(target, code):
    """Create a new file verbatim. Errors if target already exists."""
    _validate_target(target, must_exist=False)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(code.encode("utf-8"))
    byte_count = len(code.encode("utf-8"))
    return f"Wrote {byte_count} bytes to {target}"


def _write_temp_script(code, suffix, prefix="oi-edit-"):
    """Write edit code to a temp script file; preserves multi-line content verbatim."""
    script = code.replace("\r\n", "\n").replace("\r", "\n")
    if not script.strip():
        raise ValueError("empty script")
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, text=True)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as script_file:
        script_file.write(script)
        if not script.endswith("\n"):
            script_file.write("\n")
    return path


def run_sed(target, code):
    """Apply a sed script in-place (-f file). One command per line or multi-line sed scripts."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("sed: no commands in code")

    sed = _resolve_sed()
    script_path = _write_temp_script(code, ".sed")
    try:
        args = [sed]
        if _is_gnu_sed(sed):
            args.extend(["-i", "-f", script_path, target])
        else:
            args.extend(["-i", "", "-f", script_path, target])
        result = subprocess.run(args, capture_output=True, text=True)
    finally:
        if os.path.isfile(script_path):
            os.remove(script_path)

    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip()
            or f"sed exited with code {result.returncode}"
        )
    return "sed: OK"


def run_gawk(target, code):
    """Apply a gawk program in-place. Requires GNU awk (-i inplace)."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("gawk: no program in code")

    gawk = _resolve_gawk()
    prog_path = _write_temp_script(code, ".awk")
    try:
        result = subprocess.run(
            [gawk, "-i", "inplace", "-f", prog_path, target],
            capture_output=True,
            text=True,
        )
    finally:
        if os.path.isfile(prog_path):
            os.remove(prog_path)

    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip()
            or f"gawk exited with code {result.returncode}"
        )
    out = (result.stdout or "").strip()
    return out if out else "gawk: OK"


def run_jq(target, code):
    """Apply a jq filter to a JSON file, replacing it atomically."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("jq: no filter in code")

    jq = _resolve_jq()
    path = Path(target)

    # Write to a sibling temp file so os.replace() is atomic on the same filesystem
    filter_path = _write_temp_script(code, ".jq")
    fd, tmp = tempfile.mkstemp(
        suffix=path.suffix, prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(fd)
    try:
        result = subprocess.run(
            [jq, "-f", filter_path, target],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "").strip()
                or f"jq exited with code {result.returncode}"
            )
        Path(tmp).write_text(result.stdout, encoding="utf-8", newline="")
        os.replace(tmp, target)
        tmp = None  # consumed; don't delete in finally
    finally:
        if os.path.isfile(filter_path):
            os.remove(filter_path)
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)

    return "jq: OK"


def run_yq(target, code):
    """Apply a yq (mikefarah) expression in-place (YAML/JSON/TOML/XML/CSV)."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("yq: no expression in code")

    yq = _resolve_yq()
    path = Path(target).as_posix()
    expr_path = _write_temp_script(code, ".yq")
    try:
        for args in (
            [yq, "eval", "-i", "-f", expr_path, path],
            [yq, "eval", "-i", "--from-file", expr_path, path],
            [yq, "-i", "-f", expr_path, path],
        ):
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode == 0:
                return "yq: OK"
        _run_failed("yq", result)
    finally:
        if os.path.isfile(expr_path):
            os.remove(expr_path)


def _poke_dot_file_arg(path):
    """Path token for .file (quote only when the path contains whitespace)."""
    if " " in path or path[:1].isspace() or path[-1:].isspace():
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return path


def _poke_prepare_script(body, path):
    """Build a command file for poke -s; prepends .file unless user already opens/switches IOS."""
    lines = []
    if not re.search(r"^\s*\.(?:ios|file)\b", body, re.MULTILINE):
        lines.append(f".file {_poke_dot_file_arg(path)}")
    lines.append(body)
    if ".quit" not in body.lower() and ".exit" not in body.lower():
        lines.append(".quit")
    return "\n".join(lines) + "\n"


def run_poke(target, code):
    """Run GNU poke dot-commands / statements against a binary file."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("poke: no commands in code")

    poke = _resolve_poke()
    path = str(Path(target).resolve())
    body = code.replace("\r\n", "\n").replace("\r", "\n")
    script = _poke_prepare_script(body, path)

    fd, cmd_path = tempfile.mkstemp(suffix=".poke", prefix="oi-edit-", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as cmd_file:
            cmd_file.write(script)
        # poke -s loads the script then enters the REPL; .quit exits non-interactively.
        result = subprocess.run(
            [
                poke,
                "-q",
                "--no-init-file",
                "--no-hserver",
                "-s",
                cmd_path,
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
    finally:
        if os.path.isfile(cmd_path):
            os.remove(cmd_path)

    if result.returncode != 0:
        _run_failed("poke", result)
    out = (result.stdout or "").strip()
    return out if out else "poke: OK"


def _split_comby_templates(code):
    """Match and rewrite templates separated by a line containing only ---."""
    stripped = code.strip()
    if "\n---\n" in stripped:
        match, rewrite = stripped.split("\n---\n", 1)
        return match.strip(), rewrite.strip()
    lines = stripped.splitlines()
    if len(lines) < 2:
        raise ValueError(
            "comby: code must be match template and rewrite template "
            "(two lines, or multiline blocks separated by a --- line)"
        )
    return lines[0].strip(), "\n".join(lines[1:]).strip()


def _comby_rewritten_source(stdout_bytes):
    """Parse comby -json / -json-lines stdout for rewritten_source."""
    raw = (stdout_bytes or b"").decode("utf-8", errors="replace").strip()
    if not raw:
        raise RuntimeError("comby: empty output")

    payloads = []
    if raw.startswith("{"):
        try:
            payloads = [json.loads(raw)]
        except json.JSONDecodeError:
            payloads = []
    if not payloads:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                payloads.append(json.loads(line))

    for data in payloads:
        if isinstance(data, list):
            if not data:
                continue
            data = data[0]
        rewritten = data.get("rewritten_source")
        if rewritten is not None:
            return rewritten.encode("utf-8")

    raise RuntimeError(
        f"comby: no rewritten_source in JSON output: {raw[:200]!r}"
    )


def run_comby(target, code):
    """Structural search/replace via comby -stdin -json-lines (single file, atomic write)."""
    _validate_target(target, must_exist=True)
    match, rewrite = _split_comby_templates(code)
    if not match or not rewrite:
        raise ValueError("comby: both match and rewrite templates are required")

    comby = _resolve_comby()
    json_flag = _comby_json_flag(comby)
    source = Path(target).read_bytes()
    result = subprocess.run(
        [comby, "-stdin", json_flag, match, rewrite],
        input=source,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            (stderr or stdout).strip() or f"comby exited with code {result.returncode}"
        )
    _atomic_replace_from_stdout(target, _comby_rewritten_source(result.stdout))
    return "comby: OK"


def run_patch(target, code):
    """Apply a unified diff (patch format) to an existing file."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("patch: diff body is empty")

    patch_bin = _resolve_patch()
    path = Path(target)
    diff = code.replace("\r\n", "\n").replace("\r", "\n")
    if not diff.endswith("\n"):
        diff += "\n"

    result = subprocess.run(
        [patch_bin, "-p0", "--forward", path.name],
        input=diff.encode("utf-8"),
        capture_output=True,
        cwd=str(path.parent),
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            (stderr or stdout).strip() or f"patch exited with code {result.returncode}"
        )
    out = ((result.stdout or b"") + (result.stderr or b"")).decode("utf-8", errors="replace").strip()
    return out if out else "patch: OK"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_edit(language, code, target):
    language = language.lower().strip()
    if language not in EDIT_LANGUAGES:
        raise ValueError(
            f"unsupported edit language: {language!r}. "
            f"Choose one of: {', '.join(sorted(EDIT_LANGUAGES))}"
        )
    if not isinstance(code, str):
        raise ValueError("code must be a string")

    if language == "write":
        return run_write(target, code)
    if language == "sed":
        return run_sed(target, code)
    if language == "gawk":
        return run_gawk(target, code)
    if language == "jq":
        return run_jq(target, code)
    if language == "yq":
        return run_yq(target, code)
    if language == "poke":
        return run_poke(target, code)
    if language == "comby":
        return run_comby(target, code)
    if language == "patch":
        return run_patch(target, code)
    # unreachable given the set-membership check above
    raise ValueError(f"unsupported edit language: {language!r}")
