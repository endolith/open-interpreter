"""Optional pytest hooks for interactive code-execution approval during integration tests.

When installed (see tests/conftest.py), wraps ``Computer.run`` / ``Terminal.run`` so
tests can print an approval banner before executing model-generated code.  On
Windows, pytest often closes stderr handles; writes must fail soft or auto-approve.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

_original_terminal_run: Callable[..., Any] | None = None
_patch_installed = False


def _safe_write(msg: str) -> None:
    """Write to stderr without raising when the handle is invalid (Windows/pytest)."""
    text = msg if msg.endswith("\n") else msg + "\n"
    for stream in (getattr(sys, "__stderr__", None), sys.stderr, sys.stdout):
        if stream is None:
            continue
        try:
            if hasattr(stream, "closed") and stream.closed:
                continue
            stream.write(text)
            if hasattr(stream, "flush"):
                stream.flush()
            return
        except OSError:
            continue


def _tty_print(msg: str) -> None:
    _safe_write(msg)


def _running_under_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _should_auto_approve() -> bool:
    if os.environ.get("OI_TEST_AUTO_APPROVE", "").lower() in ("1", "true", "yes"):
        return True
    return _running_under_pytest()


def prompt_for_code_execution(
    *,
    test_name: str | None = None,
    language: str | None = None,
    code: str | None = None,
) -> bool:
    if _should_auto_approve():
        return True

    title = test_name or "integration test"
    lines = [
        "",
        f"--- {title}: approve code execution? ---",
        f"Language: {language or '?'}",
    ]
    if code:
        preview = code if len(code) <= 400 else code[:400] + "..."
        lines.append(preview)
    lines.append("(auto-approved under pytest / auto_run)")
    _tty_print("\n".join(lines))
    return True


def _approving_run(self, language, code, stream=False, display=False, *args, **kwargs):
    if not prompt_for_code_execution(
        test_name=getattr(self, "_oi_test_name", None),
        language=language,
        code=code,
    ):
        return [] if stream else ""
    assert _original_terminal_run is not None
    return _original_terminal_run(
        self, language, code, stream=stream, display=display, *args, **kwargs
    )


def install_approval_patch() -> None:
    """Wrap Terminal.run once so integration tests can gate execution safely."""
    global _original_terminal_run, _patch_installed
    if _patch_installed:
        return

    from interpreter.core.terminal.terminal import Terminal

    _original_terminal_run = Terminal.run
    Terminal.run = _approving_run  # type: ignore[method-assign]
    _patch_installed = True


def uninstall_approval_patch() -> None:
    global _patch_installed
    if not _patch_installed or _original_terminal_run is None:
        return
    from interpreter.core.terminal.terminal import Terminal

    Terminal.run = _original_terminal_run  # type: ignore[method-assign]
    _patch_installed = False
