"""
file_edit.py — backend runners for the `edit` tool.

Each runner is a thin wrapper that:
  - Validates the target path (absolute, exists/doesn't-exist as required)
  - Delegates to the right binary or pure-Python I/O
  - Returns a short success string, or raises on failure (caller gets traceback/message)

Binary resolution mirrors resolve_bash.py: env-var override → PATH → Git usr/bin on Windows.
The model never constructs shell command strings; flags and temp-file hygiene live here.
"""

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..terminal.languages.resolve_bash import resolve_bash_executable

EDIT_LANGUAGES = frozenset({"sed", "ed", "gawk", "jq", "write"})


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


def _resolve_ed():
    return _resolve_binary("INTERPRETER_ED", ["ed"])


def _resolve_jq():
    return _resolve_binary("INTERPRETER_JQ", ["jq"])


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
                f"file already exists — use sed/ed/gawk/jq to edit it: {target}"
            )


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


def run_sed(target, code):
    """Apply sed commands (one per line) in-place."""
    _validate_target(target, must_exist=True)
    commands = [line for line in code.splitlines() if line.strip()]
    if not commands:
        raise ValueError("sed: no commands in code")

    sed = _resolve_sed()
    args = [sed]
    if _is_gnu_sed(sed):
        # GNU sed: -i with no suffix argument
        args.append("-i")
        for cmd in commands:
            args.extend(["-e", cmd])
    else:
        # BSD sed (macOS): -i requires a suffix argument (empty string = no backup)
        for cmd in commands:
            args.extend(["-i", "", "-e", cmd])
    args.append(target)

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip()
            or f"sed exited with code {result.returncode}"
        )
    return "sed: OK"


def run_ed(target, code):
    """Feed an ed script to the file. Script must end with wq (or w then q) to save."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("ed: no commands in code")

    ed = _resolve_ed()
    script = code if code.endswith("\n") else code + "\n"
    result = subprocess.run(
        [ed, "-s", target],
        input=script,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip()
            or f"ed exited with code {result.returncode}"
        )
    out = (result.stdout or "").strip()
    return out if out else "ed: OK"


def run_gawk(target, code):
    """Apply a gawk program in-place. Requires GNU awk (-i inplace)."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("gawk: no program in code")

    gawk = _resolve_gawk()
    result = subprocess.run(
        [gawk, "-i", "inplace", code, target],
        capture_output=True,
        text=True,
    )
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
    fd, tmp = tempfile.mkstemp(
        suffix=path.suffix, prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(fd)
    try:
        result = subprocess.run(
            [jq, code, target],
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
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)

    return "jq: OK"


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
    if language == "ed":
        return run_ed(target, code)
    if language == "gawk":
        return run_gawk(target, code)
    if language == "jq":
        return run_jq(target, code)
    # unreachable given the set-membership check above
    raise ValueError(f"unsupported edit language: {language!r}")
