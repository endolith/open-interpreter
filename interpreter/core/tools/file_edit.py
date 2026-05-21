import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..terminal.languages.resolve_bash import resolve_bash_executable

EDIT_LANGUAGES = frozenset({"sed", "ed", "gawk", "jq", "write"})


def _resolve_executable(env_var, unix_names, windows_relative_paths=()):
    env = os.environ.get(env_var, "").strip()
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"{env_var} is set but not a file: {env!r}")
        return env

    for name in unix_names:
        path = shutil.which(name)
        if path:
            return path

    if platform.system() == "Windows":
        bash = resolve_bash_executable()
        bash_dir = os.path.dirname(bash)
        for rel in windows_relative_paths:
            candidate = os.path.normpath(os.path.join(bash_dir, rel))
            if os.path.isfile(candidate):
                return candidate
        usr_bin = os.path.normpath(os.path.join(bash_dir, "..", "usr", "bin"))
        for name in unix_names:
            candidate = os.path.join(usr_bin, name + ".exe")
            if os.path.isfile(candidate):
                return candidate

    raise FileNotFoundError(
        f"{'/'.join(unix_names)} not found. Install it, add to PATH, or set {env_var}."
    )


def _resolve_sed():
    return _resolve_executable(
        "INTERPRETER_SED",
        ("sed",),
        (os.path.join("..", "usr", "bin", "sed.exe"),),
    )


def _resolve_gawk():
    return _resolve_executable(
        "INTERPRETER_GAWK",
        ("gawk", "awk"),
        (os.path.join("..", "usr", "bin", "gawk.exe"),),
    )


def _resolve_ed():
    return _resolve_executable(
        "INTERPRETER_ED",
        ("ed",),
        (os.path.join("..", "usr", "bin", "ed.exe"),),
    )


def _resolve_jq():
    return _resolve_executable(
        "INTERPRETER_JQ",
        ("jq",),
        (os.path.join("..", "usr", "bin", "jq.exe"),),
    )


def _is_gnu_sed(sed_path):
    result = subprocess.run(
        [sed_path, "--version"],
        capture_output=True,
        text=True,
    )
    return "GNU" in (result.stdout + result.stderr)


def validate_target(target, *, must_exist):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target is required and must be a non-empty string")
    if not os.path.isabs(target):
        raise ValueError(
            "target must be an absolute path (e.g. C:\\Users\\... on Windows, /home/... on Linux/Mac)"
        )
    path = Path(target)
    if must_exist:
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {target}")
    elif path.exists():
        raise FileExistsError(f"file already exists (use sed/ed/gawk/jq to edit): {target}")


def run_write(target, code):
    validate_target(target, must_exist=False)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(code.encode("utf-8"))
    return f"Wrote {len(code.encode('utf-8'))} bytes to {target}"


def run_sed(target, code):
    validate_target(target, must_exist=True)
    commands = [line for line in code.splitlines() if line.strip()]
    if not commands:
        raise ValueError("sed: code is empty")

    sed = _resolve_sed()
    args = [sed]
    if _is_gnu_sed(sed):
        args.append("-i")
        for cmd in commands:
            args.extend(["-e", cmd])
    else:
        # BSD sed (macOS): edit via temp backup extension
        for cmd in commands:
            args.extend(["-i", "", "-e", cmd])
    args.append(target)

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or f"sed exited {result.returncode}"
        raise RuntimeError(err)
    return "sed: OK"


def run_ed(target, code):
    validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("ed: code is empty")

    ed = _resolve_ed()
    result = subprocess.run(
        [ed, "-s", target],
        input=code if code.endswith("\n") else code + "\n",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or f"ed exited {result.returncode}"
        raise RuntimeError(err)
    out = (result.stdout or "").strip()
    return out if out else "ed: OK"


def run_gawk(target, code):
    validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("gawk: code is empty")

    gawk = _resolve_gawk()
    result = subprocess.run(
        [gawk, "-i", "inplace", code, target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or f"gawk exited {result.returncode}"
        raise RuntimeError(err)
    out = (result.stdout or "").strip()
    return out if out else "gawk: OK"


def run_jq(target, code):
    validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("jq: code is empty")

    jq = _resolve_jq()
    path = Path(target)
    fd, tmp_name = tempfile.mkstemp(
        suffix=path.suffix,
        prefix=path.name + ".",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        result = subprocess.run(
            [jq, code, target],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip() or f"jq exited {result.returncode}"
            raise RuntimeError(err)
        Path(tmp_name).write_text(result.stdout, encoding="utf-8", newline="")
        os.replace(tmp_name, target)
        tmp_name = None
    finally:
        if tmp_name and os.path.isfile(tmp_name):
            os.remove(tmp_name)

    return "jq: OK"


def run_edit(language, code, target):
    language = language.lower().strip()
    if language not in EDIT_LANGUAGES:
        raise ValueError(
            f"unsupported edit language: {language!r}. "
            f"Use one of: {', '.join(sorted(EDIT_LANGUAGES))}"
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
    raise ValueError(f"unsupported edit language: {language!r}")
