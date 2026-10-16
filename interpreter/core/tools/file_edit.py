"""
file_edit.py — backend runners for the `edit` tool.

Each runner is a thin wrapper that:
  - Validates the target path (absolute, exists/doesn't-exist as required)
  - Delegates to the right binary or pure-Python I/O
  - Returns a short success string, or raises on failure (caller gets traceback/message)

Binary resolution mirrors resolve_bash.py: env-var override → PATH → Git usr/bin on Windows.
The model never constructs shell command strings; flags and temp-file hygiene live here.

In-place edit strategies (Windows cross-drive safety):
  - stdout + atomic replace: sed, jq, yq — tool emits the new file; we write a
    temp sibling in the target's parent directory and os.replace().
  - cwd in target's parent: gawk, patch — tool must edit in place (e.g. gawk without
    print); run with cwd set to the target directory so tool temps stay on that drive.
  - direct open: poke (binary), write (create or replace), append (append to existing).
Line-ending policy (CRLF vs LF):
  - Text tools (sed, gawk, jq, yq) emit LF even when the target uses CRLF
    (e.g. .bat files), which used to silently convert the whole file. To avoid
    that, stdout-based rewrites and write-replacements are normalized back to
    the target's original newline style before writing, and append converts its
    block to match. Poke is binary and never normalized. Patch is applied
    against an LF-normalized copy when the target is CRLF, then converted back,
    so LF context lines still match.
"""

import difflib
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
    "append", "patch",
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


def _assert_mikefarah_yq(yq):
    """edit/yq requires https://github.com/mikefarah/yq, not the Python jq-wrapper yq."""
    result = subprocess.run(
        [yq, "--version"], capture_output=True, text=True
    )
    version_text = ((result.stdout or "") + (result.stderr or "")).lower()
    if "mikefarah" not in version_text and "github.com/mikefarah/yq" not in version_text:
        raise RuntimeError(
            "yq edit language requires mikefarah/yq "
            "(https://github.com/mikefarah/yq). "
            f"INTERPRETER_YQ or PATH resolved to a different program: "
            f"{(result.stdout or result.stderr or '').strip()!r}"
        )


def _normalize_yq_expression(code):
    """LF-normalize; keep internal newlines for multi-line expressions."""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def _resolve_poke():
    return _resolve_binary("INTERPRETER_POKE", ["poke"])


def _resolve_patch():
    return _resolve_binary("INTERPRETER_PATCH", ["patch"])


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def _reject_wrong_os_path(target):
    """Reject paths shaped for a different OS before os.path misreads them.

    On Windows, os.path.isabs('/c/Users/...') is True (rooted on the current
    drive), so a Git Bash style drive path sails through validation and lands
    at e.g. C:\\c\\Users — a literal 'c' folder. Catch that here and tell the
    model the right shape instead.
    """
    if platform.system() == "Windows":
        if re.match(r"^[\\/][a-zA-Z][\\/]", target):
            raise ValueError(
                f"target looks like a Git Bash style path ({target!r}); "
                "on Windows use a drive-letter path (e.g. 'C:/Users/...')"
            )
        if re.match(r"^[\\/]", target) and not re.match(r"^[\\/]{2}", target):
            raise ValueError(
                f"target {target!r} has no drive letter and resolves against "
                "the current drive; use a full path with a drive letter "
                "(e.g. 'C:/Users/...')"
            )
    else:
        if re.match(r"^[a-zA-Z]:[\\/]", target):
            raise ValueError(
                f"target looks like a Windows path ({target!r}), but this "
                "system is not Windows; use a POSIX absolute path "
                "(e.g. '/home/...')"
            )


def _validate_target(target, *, must_exist):
    """Validate the target path shape, and optionally its existence.

    must_exist True means the file must already be there (in-place edits);
    False means it must not (kept for callers that truly create); None skips
    the existence check -- used when creating and replacing share one runner
    and the confirmation gates upstream have already approved either outcome.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target is required and must be a non-empty string")
    _reject_wrong_os_path(target)
    if not os.path.isabs(target):
        raise ValueError(
            "target must be an absolute path "
            "(e.g. C:\\Users\\... on Windows, /home/... on Linux/Mac)"
        )
    if must_exist is None:
        return
    path = Path(target)
    if must_exist:
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {target}")
    else:
        if path.exists():
            raise FileExistsError(
                f"file already exists — write a new file name or use another edit language to modify the file: {target}"
            )


def _run_failed(lang, result):
    raise RuntimeError(
        _subprocess_text(result)
        or f"{lang} exited with code {result.returncode}"
    )


def _subprocess_text(result):
    """Decode captured subprocess output as UTF-8 (tools emit UTF-8; avoid text=True on Windows)."""
    if isinstance(result.stdout, bytes) or isinstance(result.stderr, bytes):
        out = (result.stdout or b"") + (result.stderr or b"")
        return out.decode("utf-8", errors="replace").strip()
    return ((result.stdout or "") + (result.stderr or "")).strip()


def _target_parent_dir(target):
    """Parent directory of target; use as subprocess cwd for in-place tools on Windows."""
    return str(Path(target).parent)


def _detect_newline_style(data):
    """Return b'\\r\\n' if the file uses CRLF, b'\\n' if LF-only, else None.

    A file counts as CRLF when it contains at least one CRLF pair. Mixed files
    (some CRLF, some lone LF) are treated as CRLF so normalization converges to
    the dominant Windows style instead of flipping back and forth. Returns None
    for empty files or files with no newlines, where there is no style to keep.
    """
    if not data:
        return None
    if b"\r\n" in data:
        return b"\r\n"
    if b"\n" in data:
        return b"\n"
    return None


def _normalize_to_style(data, style):
    """Convert line breaks in data to the given style (CRLF or LF).

    Only real line breaks are translated: bare LF becomes CRLF for CRLF
    targets, CRLF pairs become LF for LF targets. A lone CR is data, not a
    line break — real tools write it verbatim (e.g. sed s/x/y\\ry/ inserts a
    literal CR), so promoting it to a full break would corrupt output that
    re-emits fields containing CRs (notably gawk print $1 on CRLF files).
    """
    if style is None:
        return data
    if style == b"\r\n":
        return re.sub(rb"(?<!\r)\n", b"\r\n", data)
    return data.replace(b"\r\n", b"\n")


def _match_target_newlines(target, new_bytes):
    """Normalize new_bytes to the newline style of the file at target.

    Text tools emit LF even for CRLF targets (notably .bat files), which used
    to silently convert the whole file. When the original uses CRLF, lone LFs
    become CRLF; when it uses LF, stray CRLF pairs become LF. A lone CR is
    left alone on both styles — it is data, not a line break. Files with no
    newlines (or missing files, e.g. new writes) are left untouched.
    """
    try:
        original = Path(target).read_bytes()
    except OSError:
        return new_bytes
    style = _detect_newline_style(original)
    if style is None:
        return new_bytes
    return _normalize_to_style(new_bytes, style)


def _atomic_replace_from_stdout(target, stdout_bytes):
    """Write tool stdout into target atomically via a temp file in the target's directory.

    Standard pattern for edit runners whose tool emits the full new file on stdout.
    Keeps temp files on the same drive as the target (Windows cannot rename across drives).
    Preserves the original file's permission bits (e.g. executable bit), which a
    fresh mkstemp file would otherwise lose.
    Restores the target's original newline style (CRLF vs LF) so LF-emitting
    tools do not silently convert e.g. .bat files.
    """
    path = Path(target)
    # Read mode before replacing; target is validated to exist by callers.
    try:
        original_mode = path.stat().st_mode
    except FileNotFoundError:
        original_mode = None
    stdout_bytes = _match_target_newlines(target, stdout_bytes)
    fd, tmp = tempfile.mkstemp(
        suffix=path.suffix, prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(fd)
    try:
        Path(tmp).write_bytes(stdout_bytes)
        if original_mode is not None:
            os.chmod(tmp, original_mode)
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_write(target, code):
    """Create a new file verbatim, or replace an existing one verbatim.

    Overwriting used to be refused, which taught models a worse habit: write,
    get blocked, delete the file, write again. A write that would replace a
    file now previews as a diff first, so by the time this runs the user has
    confirmed the replacement.
    When replacing, the new body is normalized to the existing file's newline
    style (CRLF vs LF) so rewriting a CRLF file with LF text does not silently
    convert it. New files are written as given.
    """
    _validate_target(target, must_exist=None)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = code.encode("utf-8")
    if path.is_file():
        try:
            original = path.read_bytes()
        except OSError:
            original = b""
        style = _detect_newline_style(original)
        if style is not None:
            data = _normalize_to_style(data, style)
    path.write_bytes(data)
    return f"Wrote {len(data)} bytes to {target}"


def run_append(target, code):
    """Append a block of text to the end of an existing file.

    The block is appended verbatim — no separator newline is added. If the file
    does not end with a newline (or blank line) and one is wanted (e.g. before
    a markdown header), the code itself must start with one. The block's
    newlines are converted to match the file's existing style so appending LF
    text to a CRLF file does not mix styles.
    """
    _validate_target(target, must_exist=True)
    if not isinstance(code, str) or not code:
        raise ValueError("append: no text in code")
    path = Path(target)
    try:
        original = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"file not found: {target}") from exc
    data = code.encode("utf-8")
    style = _detect_newline_style(original)
    if style is not None:
        data = _normalize_to_style(data, style)
    with open(path, "ab") as handle:
        handle.write(data)
    return f"Appended {len(data)} bytes to {target}"


def _write_temp_script(code, suffix, prefix="oi-edit-"):
    """Write edit code to a temp script file; preserves multi-line content verbatim."""
    script = code.replace("\r\n", "\n").replace("\r", "\n")
    if not script.strip():
        raise ValueError("empty script")
    if not script.endswith("\n"):
        script += "\n"
    # Binary UTF-8 write: mkstemp(text=True) uses the locale encoding on Windows (e.g.
    # cp1252), which mojibakes non-ASCII jq/sed/gawk filters when tools read the file.
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    with os.fdopen(fd, "wb") as script_file:
        script_file.write(script.encode("utf-8"))
    return path


def run_sed(target, code):
    """Apply a sed script (-f file), replacing the file atomically.

    Avoid sed -i: GNU sed creates its backup next to the process cwd, so on
    Windows a target on another drive (e.g. D:\\) fails with "Invalid cross-device
    link" when cwd is on C:\\. Write stdout to a sibling temp file instead.
    """
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("sed: no commands in code")

    sed = _resolve_sed()
    script_path = _write_temp_script(code, ".sed")
    try:
        result = subprocess.run(
            [sed, "-f", script_path, target],
            capture_output=True,
        )
    finally:
        if os.path.isfile(script_path):
            os.remove(script_path)

    if result.returncode != 0:
        _run_failed("sed", result)
    new_bytes = result.stdout or b""
    # Warn instead of silently reporting OK when nothing changed. sed exits 0
    # even when the pattern matched zero lines, which previously looked like
    # success while leaving the file untouched.
    try:
        before_bytes = Path(target).read_bytes()
    except FileNotFoundError:
        before_bytes = None
    if before_bytes is not None and before_bytes == new_bytes:
        return "sed: OK (no changes — pattern did not match)"
    _atomic_replace_from_stdout(target, new_bytes)
    return "sed: OK"


def run_gawk(target, code):
    """Apply a gawk program, replacing the file atomically from stdout.

    The program must emit the new file content on stdout (filter model, e.g.
    ``{ gsub(...); print }``), like sed. Runs without ``-i inplace`` so behavior
    is deterministic across platforms and matches dry_run_edit; ``-i inplace``
    sends END-block output to stdout while truncating the file, which silently
    wiped files. Empty output for a non-empty file is refused.
    """
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("gawk: no program in code")

    gawk = _resolve_gawk()
    prog_path = _write_temp_script(code, ".awk")
    try:
        result = subprocess.run(
            [gawk, "-f", prog_path, target],
            capture_output=True,
        )
    finally:
        if os.path.isfile(prog_path):
            os.remove(prog_path)

    if result.returncode != 0:
        _run_failed("gawk", result)
    new_bytes = result.stdout or b""
    try:
        before_bytes = Path(target).read_bytes()
    except FileNotFoundError:
        before_bytes = None
    if before_bytes:
        if not new_bytes.strip():
            raise RuntimeError(
                "gawk produced no output for a non-empty file "
                "(file was not modified; END-only programs print to stdout — "
                "include a per-line print when transforming)"
            )
        if before_bytes == new_bytes:
            return "gawk: OK (no changes — program output matches file)"
        before_lines = before_bytes.count(b"\n")
        new_lines = new_bytes.count(b"\n")
        if before_lines >= 10 and new_lines <= max(2, before_lines // 10):
            _atomic_replace_from_stdout(target, new_bytes)
            return (
                f"gawk: OK (warning: {before_lines} lines -> {new_lines} lines — "
                "END-only output replaces the whole file; include per-line "
                "print when transforming)"
            )
    _atomic_replace_from_stdout(target, new_bytes)
    return "gawk: OK"


def run_jq(target, code):
    """Apply a jq filter to a JSON file, replacing it atomically."""
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("jq: no filter in code")

    jq = _resolve_jq()
    filter_path = _write_temp_script(code, ".jq")
    try:
        result = subprocess.run(
            [jq, "-f", filter_path, target],
            capture_output=True,
        )
        if result.returncode != 0:
            _run_failed("jq", result)
        _atomic_replace_from_stdout(target, result.stdout or b"")
    finally:
        if os.path.isfile(filter_path):
            os.remove(filter_path)

    return "jq: OK"


def _yq_eval_argv_candidates(yq, expr, data_path):
    """Argv lists for mikefarah yq eval (stdout mode — no -i).

    Pass the expression as a subprocess argument (no shell). Multi-line expressions
    work in argv; a temp file is not required.

    Do NOT use ``-f`` / ``--from-file`` for the expression: with mikefarah yq 4.53.x,
    ``eval -f <exprfile> <datafile>`` often exits 0 and prints the input unchanged,
    which made the edit tool report "yq: OK" while leaving the file untouched.
    The working invocation is ``eval '<expression>' <datafile>`` (see 2026-05-23 report).
    """
    path = Path(data_path).as_posix()
    return (
        [yq, "eval", expr, path],
        [yq, expr, path],
    )


def _run_yq_eval(yq, expr, target):
    """Run yq eval; return the first subprocess result with returncode 0."""
    _assert_mikefarah_yq(yq)
    result = None
    before = Path(target).read_text(encoding="utf-8")
    for args in _yq_eval_argv_candidates(yq, expr, target):
        result = subprocess.run(args, capture_output=True)
        if result.returncode == 0:
            out = (result.stdout or b"").decode("utf-8")
            # Catch silent no-ops (exit 0, stdout equals input) on obvious assignments.
            if out == before and "=" in expr and before.strip():
                continue
            return result
    _run_failed("yq", result)


def run_yq(target, code):
    """Apply a yq (mikefarah) expression, replacing the file atomically."""
    _validate_target(target, must_exist=True)
    expr = _normalize_yq_expression(code)
    if not expr.strip():
        raise ValueError("yq: no expression in code")

    yq = _resolve_yq()
    path = Path(target)
    had_content = path.stat().st_size > 0

    result = _run_yq_eval(yq, expr, target)
    out = result.stdout or b""
    if had_content and not out.strip():
        raise RuntimeError(
            "yq produced no output for a non-empty file "
            "(file was not modified; check the expression)"
        )
    _atomic_replace_from_stdout(target, out)

    return "yq: OK"


def _poke_prepare_script(body, path):
    """Build a command file for poke -s; prepends .file unless user already opens/switches IOS.

    The path goes into the `.file` line raw, never quoted: poke's `.file` takes
    the rest of the line literally, so surrounding quotes become part of the
    filename and any target containing a space fails with 'error: opening
    "\"/tmp/a b/f.bin\""'. Unquoted, spaces, quotes and backslashes in the
    name all work -- verified against GNU poke 4.3.
    """
    lines = []
    if not re.search(r"^\s*\.(?:ios|file)\b", body, re.MULTILINE):
        lines.append(f".file {path}")
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

    cmd_path = _write_temp_script(script, ".poke")
    try:
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


def _hunks_missing_trailing_context(code):
    """Return the 1-based hunk numbers whose last change has <2 context lines after it.

    GNU patch will not apply a hunk that has fewer than two unchanged context
    lines following its final +/- line, unless the hunk truly ends at
    end-of-file. Verified against patch 2.8: with 0 or 1 trailing context lines
    the hunk is rejected at any amount of leading context, and with 2 it applies.
    The rejection is reported only as "Hunk #N FAILED", which does not suggest
    the context depth is at fault, so the model lowers the fuzz or renumbers the
    header instead of adding the lines. Named here so the message can say what
    to change.
    """
    thin = []
    hunk_no = 0
    trailing = None  # context lines seen since the last change in this hunk
    for line in code.split("\n"):
        if line.startswith("@@"):
            if trailing is not None and trailing < 2:
                thin.append(hunk_no)
            hunk_no += 1
            trailing = None
        elif line.startswith(("---", "+++")):
            continue
        elif line[:1] == " ":
            if trailing is not None:
                trailing += 1
        elif line[:1] in ("-", "+"):
            trailing = 0
    if trailing is not None and trailing < 2:
        thin.append(hunk_no)
    return thin


def _read_patch_reject(target):
    """The reject file's contents, read before it is cleaned up.

    This is the most useful diagnostic available on a failed patch: it shows the
    hunk as patch understood it next to the file's real lines, so the caller can
    see exactly which context did not match. Deleting the file without reading
    it first throws that away and leaves the error message as the only account.
    """
    reject = Path(target).with_name(Path(target).name + ".rej")
    try:
        return reject.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _context_lines_present(target, diff):
    """True if the hunk's unchanged context lines are found verbatim in the target.

    Distinguishes the two reasons a hunk fails. If the context is present but
    there is too little of it after the change, the trailing-context rule is the
    plausible cause and can be named. If the context is not in the file at all,
    the edit was aimed at the wrong text, and suggesting anything about context
    depth would send the caller off fixing the wrong thing.
    """
    original = _read_text_for_diff(target)
    if original is None:
        return None
    lines = original.splitlines()
    for line in diff.split("\n"):
        if not line.startswith(" ") or line.strip() == "":
            continue
        if line[1:] not in lines:
            return False
    return True


def _cleanup_patch_artifacts(target):
    """Remove the .orig/.rej files patch leaves beside the target.

    patch writes .orig whenever it modifies a file -- success as well as
    failure -- and .rej when a hunk fails. Both land in the user's directory
    next to the file they were editing, which is litter here, and a stale .rej
    reads as an unresolved edit on the next visit.

    Removing the backup unconditionally is a deliberate choice, not an
    oversight: every other language in this module (sed, gawk, jq, yq)
    rewrites the target atomically and leaves no backup, so keeping patch's
    would make it the only language that silently deposits .orig files. Callers
    that want the original should copy it themselves before editing.
    """
    path = Path(target)
    for suffix in (".orig", ".rej"):
        artifact = path.with_name(path.name + suffix)
        try:
            artifact.unlink()
        except OSError:
            pass


def run_patch(target, code):
    """Apply a unified diff (patch format) to an existing file.

    When the target uses CRLF, the file is normalized to LF before patch runs
    (diff context lines are LF) and converted back afterwards, so LF hunks
    still match CRLF files without converting the file. On failure the original
    bytes are restored so the LF normalization never leaks.
    """
    _validate_target(target, must_exist=True)
    if not code.strip():
        raise ValueError("patch: diff body is empty")

    patch_bin = _resolve_patch()
    path = Path(target)
    diff = code.replace("\r\n", "\n").replace("\r", "\n")
    if not diff.endswith("\n"):
        diff += "\n"

    original_bytes = path.read_bytes()
    original_style = _detect_newline_style(original_bytes)
    normalized_for_patch = False
    if original_style == b"\r\n":
        lf_bytes = original_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if lf_bytes != original_bytes:
            path.write_bytes(lf_bytes)
            normalized_for_patch = True

    try:
        result = subprocess.run(
            [patch_bin, "-p0", "--forward", path.name],
            input=diff.encode("utf-8"),
            capture_output=True,
            cwd=_target_parent_dir(target),
        )
    except Exception:
        if normalized_for_patch:
            path.write_bytes(original_bytes)
        raise
    # Read the reject before cleaning up: it is the only side-by-side of the
    # hunk against the file's real lines, and it is removed on both paths.
    reject = _read_patch_reject(target)
    _cleanup_patch_artifacts(target)
    if result.returncode != 0:
        if normalized_for_patch:
            path.write_bytes(original_bytes)
        _raise_patch_failure(result, diff, target, reject)
    if normalized_for_patch:
        try:
            patched = path.read_bytes()
        except OSError:
            patched = original_bytes
        path.write_bytes(_normalize_to_style(patched, b"\r\n"))
    else:
        # Non-CRLF targets: still normalize tool-introduced CRLF back to the
        # original style when one exists, mirroring the atomic-replace path.
        try:
            patched = path.read_bytes()
        except OSError:
            patched = None
        if patched is not None and original_style is not None:
            normalized = _normalize_to_style(patched, original_style)
            if normalized != patched:
                path.write_bytes(normalized)
    out = _subprocess_text(result)
    return out if out else "patch: OK"


def _raise_patch_failure(result, diff, target, reject=""):
    """Turn a patch failure into a message that names the likely mistake."""
    output = _subprocess_text(result)
    # patch always says it is "saving rejects to file X.rej", but the reject is
    # removed immediately afterwards. Left in place, the message points at a
    # file that does not exist.
    output = re.sub(r"\s*-- saving rejects to file \S+", "", output).strip()

    # The `*** Begin Patch` / `*** Update File:` envelope is a different
    # tool's format. patch cannot read it and says only "Only garbage was
    # found", which reads like the diff was corrupt rather than wrapped.
    if "*** Begin Patch" in diff or "*** Update File:" in diff:
        raise RuntimeError(
            "patch: this looks like an OpenAI-style '*** Begin Patch' envelope, "
            "which patch cannot read (it reports 'Only garbage was found in the "
            "patch input').\n"
            "Send a bare unified diff instead: '--- <file>' and '+++ <file>' header "
            "lines, then @@ hunks. No Begin/End Patch wrapper."
        )
    if "FAILED" in output or "malformed patch" in output:
        thin = _hunks_missing_trailing_context(diff)
        if thin and _context_lines_present(target, diff):
            listed = ", ".join(str(n) for n in thin)
            raise RuntimeError(
                f"{output}\n"
                f"Likely cause: hunk(s) {listed} have fewer than two unchanged "
                "context lines after their last change. patch requires two, unless "
                "the hunk truly ends at end-of-file; with only one or none it "
                "cannot confirm where the change belongs and rejects the hunk.\n"
                "Add two or three unchanged context lines after the final + or - "
                "line of each hunk (matching lines already in the file, each "
                "prefixed with a single space)."
            )
    if reject:
        # The context is wrong, or the counts are: show the caller what the file
        # actually holds against what the hunk expected, so the next attempt can
        # be aimed at the real text instead of guessed at.
        raise _reject_report(output, diff, reject)
    raise RuntimeError(output or f"patch exited with code {result.returncode}")


def _reject_report(output, diff, reject):
    """Message that includes the reject, since the .rej file no longer exists."""
    return RuntimeError(
        f"{output}\n"
        "The context lines did not match the file. patch's reject (normally "
        "written to a .rej file, which is cleaned up here) is below -- it shows "
        "the hunk as patch read it, so you can compare against the real file:\n"
        f"{reject}\n"
        "Re-read the target and rebuild the hunk from its exact current lines; "
        "context that is not in the file verbatim can never match."
    )


# ---------------------------------------------------------------------------
# Dry-run previews (no file modifications)
# ---------------------------------------------------------------------------

def _read_text_for_diff(target):
    """Read the target as text for diffing, or None if it is not decodable.

    Binary targets (poke) never reach here, but a file can still hold bytes that
    are not valid UTF-8, and difflib would either raise or produce mojibake. In
    that case the caller falls back to showing the tool's raw output, which is
    no worse than the previous behaviour.
    """
    try:
        return Path(target).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def unified_edit_diff(target, new_text, *, context=3):
    """Unified diff between the target's current content and `new_text`.

    The dry run for most languages runs the tool *without* its in-place flag, so
    what it prints is the entire edited file. Showing that verbatim means
    re-reading a whole file to find the one line that changed, and it grows
    without bound: a 4000-line file edited in one place buries the change. A
    diff with a few lines of surrounding context shows the change and enough to
    locate it, and shrinks to the same size whether the file is ten lines or ten
    thousand.

    Returns None when the content is undecodable or identical to what is
    already on disk, so the caller can fall back to the raw tool output.

    Comparison is in LF space: line-ending style is managed (and reported)
    separately by the runners, so a CRLF file versus LF tool output must not
    read as every line changed. Both sides get the same universal-newline
    treatment (CRLF -> LF, lone CR -> LF), matching how the file is read.
    """
    original = _read_text_for_diff(target)
    if original is None:
        return None
    original_lf = original.replace("\r\n", "\n").replace("\r", "\n")
    new_lf = new_text.replace("\r\n", "\n").replace("\r", "\n")
    if original_lf == new_lf:
        return None
    diff = difflib.unified_diff(
        original_lf.splitlines(keepends=True),
        new_lf.splitlines(keepends=True),
        fromfile=Path(target).name,
        tofile=Path(target).name,
        n=context,
    )
    # splitlines(keepends=True) leaves the last line without a newline; without
    # this the final line of the diff would run into the closing fence.
    body = "".join(line if line.endswith("\n") else line + "\n" for line in diff)
    return body or None


def dry_run_edit(language, code, target):
    """Run the edit without modifying the file.

    Returns None if this language has no dry-run preview, else
    {"output": str, "ok": bool} where ok is False for tool/validation failures.
    """
    language = language.lower().strip()
    if language == "poke":
        return None
    if language == "write":
        # A write that would replace an existing file previews the diff it
        # would produce, like every other edit. A brand-new file has nothing
        # to diff against, and an undecodable (binary) target has no text
        # diff to show.
        if os.path.isfile(target):
            preview = unified_edit_diff(target, code)
            if preview is not None:
                return {"output": preview, "ok": True}
            if _read_text_for_diff(target) is not None:
                return {
                    "output": "write: no changes (content is identical to the current file)",
                    "ok": True,
                }
        return None

    def _preview(result, *, append_diff=None, diff_against=None):
        out = _subprocess_text(result)
        ok = result.returncode == 0
        # Prefer the diff; fall through to raw output when it cannot be built.
        #
        # Diff the *unstripped* stdout. _subprocess_text() strips, which would
        # drop the file's trailing newline and make the last line look changed
        # even when the edit never touched it.
        if ok and diff_against is not None:
            raw = result.stdout
            if isinstance(raw, bytes):
                new_text = raw.decode("utf-8", errors="replace")
            else:
                new_text = raw or ""
            preview = unified_edit_diff(diff_against, new_text)
            if preview is not None:
                return {"output": preview, "ok": True}
            # An empty diff with a decodable target means the edit changed
            # nothing. Say so in one line rather than falling through to the
            # raw output, which is the entire file and hides that fact.
            if _read_text_for_diff(diff_against) is not None:
                return {
                    "output": f"{language}: no changes (result is identical to the current file)",
                    "ok": True,
                }
            # Not text at all. The raw output would decode with replacement
            # characters, which looks like corruption rather than like a
            # preview; say the preview is unavailable instead. The edit itself
            # is unaffected -- the user still gets the y/n confirmation.
            return {
                "output": f"{language}: preview unavailable ({Path(diff_against).name} is not a text file)",
                "ok": True,
            }
        if not out:
            out = f"{language} exited with code {result.returncode}"
        # GNU patch --dry-run on success often only prints "checking file …"; include the diff.
        if ok and append_diff is not None and "@@" not in out:
            out = f"{out}\n\n{append_diff.strip()}" if out else append_diff.strip()
        return {"output": out, "ok": ok}

    if language == "patch":
        _validate_target(target, must_exist=True)
        if not code.strip():
            raise ValueError("patch: diff body is empty")
        patch_bin = _resolve_patch()
        path = Path(target)
        diff = code.replace("\r\n", "\n").replace("\r", "\n")
        if not diff.endswith("\n"):
            diff += "\n"
        result = subprocess.run(
            [patch_bin, "-p0", "--forward", "--dry-run", path.name],
            input=diff.encode("utf-8"),
            capture_output=True,
            cwd=_target_parent_dir(target),
        )
        return _preview(result, append_diff=diff)

    if language == "sed":
        _validate_target(target, must_exist=True)
        if not code.strip():
            raise ValueError("sed: no commands in code")
        sed = _resolve_sed()
        script_path = _write_temp_script(code, ".sed")
        try:
            result = subprocess.run(
                [sed, "-f", script_path, target],
                capture_output=True,
            )
        finally:
            if os.path.isfile(script_path):
                os.remove(script_path)
        return _preview(result, diff_against=target)

    if language == "gawk":
        _validate_target(target, must_exist=True)
        if not code.strip():
            raise ValueError("gawk: no program in code")
        gawk = _resolve_gawk()
        prog_path = _write_temp_script(code, ".awk")
        try:
            result = subprocess.run(
                [gawk, "-f", prog_path, target],
                capture_output=True,
            )
        finally:
            if os.path.isfile(prog_path):
                os.remove(prog_path)
        return _preview(result, diff_against=target)

    if language == "jq":
        _validate_target(target, must_exist=True)
        if not code.strip():
            raise ValueError("jq: no filter in code")
        jq = _resolve_jq()
        filter_path = _write_temp_script(code, ".jq")
        try:
            result = subprocess.run(
                [jq, "-f", filter_path, target],
                capture_output=True,
            )
        finally:
            if os.path.isfile(filter_path):
                os.remove(filter_path)
        return _preview(result, diff_against=target)

    if language == "yq":
        _validate_target(target, must_exist=True)
        expr = _normalize_yq_expression(code)
        if not expr.strip():
            raise ValueError("yq: no expression in code")
        yq = _resolve_yq()
        # Same stdout eval as run_yq; dry-run only shows output, never writes the file.
        result = _run_yq_eval(yq, expr, target)
        return _preview(result, diff_against=target)

    if language == "append":
        _validate_target(target, must_exist=True)
        if not isinstance(code, str) or not code:
            raise ValueError("append: no text in code")
        try:
            original_bytes = Path(target).read_bytes()
        except OSError as exc:
            raise FileNotFoundError(f"file not found: {target}") from exc
        style = _detect_newline_style(original_bytes)
        data = code.encode("utf-8")
        if style is not None:
            data = _normalize_to_style(data, style)
        try:
            original_text = original_bytes.decode("utf-8")
            appended_text = data.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "output": f"append: preview unavailable ({Path(target).name} is not a text file)",
                "ok": True,
            }
        preview = unified_edit_diff(target, original_text + appended_text)
        if preview is not None:
            return {"output": preview, "ok": True}
        return {"output": "append: no changes (empty block)", "ok": True}

    return None


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
    if language == "append":
        return run_append(target, code)
    if language == "patch":
        return run_patch(target, code)
    # unreachable given the set-membership check above
    raise ValueError(f"unsupported edit language: {language!r}")
