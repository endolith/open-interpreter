"""Tests for ComputerTool pure functions that don't need pyautogui interaction.

These tests exercise free functions in computer.py that don't touch the real
input device, so they run headless. computer.py imports pyautogui at module
load; stub it in a scoped patch.dict so the import succeeds without a display
and the stub is restored afterward. The imported computer module is kept only
for the extracted pure function; the stub does not leak to later tests.
"""

import sys
from unittest import mock

_pyautogui = mock.MagicMock()
_pyautogui.size.return_value = (1920, 1080)
_original_computer = sys.modules.get("interpreter.computer_use.tools.computer")
with mock.patch.dict(sys.modules, {"pyautogui": _pyautogui}):
    # Ensure a fresh import so the stub is actually used even if the module
    # was already cached from a prior test.
    sys.modules.pop("interpreter.computer_use.tools.computer", None)
    from interpreter.computer_use.tools.computer import chunks as _chunks
# Restore the original cached module (or remove the fresh one) so later tests
# don't see a computer module whose pyautogui global is still the stub.
if _original_computer is not None:
    sys.modules["interpreter.computer_use.tools.computer"] = _original_computer
else:
    sys.modules.pop("interpreter.computer_use.tools.computer", None)
# Also keep the package's attribute in sync if the package is already loaded.
_pkg = sys.modules.get("interpreter.computer_use.tools")
if _pkg is not None and _original_computer is not None:
    setattr(_pkg, "computer", _original_computer)
elif _pkg is not None and _original_computer is None:
    try:
        delattr(_pkg, "computer")
    except AttributeError:
        pass

chunks = _chunks


def test_chunks_larger_than_string():
    """chunks() with size > len(s) returns the whole string.

    A chunk size larger than the input must not drop or pad the text; callers
    pass the full string through untouched in that case.
    """
    assert chunks("hello", 100) == ["hello"]


def test_chunks_exact_multiple():
    """chunks() with an exact multiple returns equal-sized pieces.

    The exact-boundary case must split evenly with no trailing empty piece,
    so callers can rely on chunk boundaries aligning with the requested size.
    """
    assert chunks("abcdef", 2) == ["ab", "cd", "ef"]


def test_chunks_remainder():
    """chunks() with a non-multiple length returns a shorter final piece.

    A remainder must be preserved as a final partial chunk rather than
    silently dropped, so no text is lost for inputs that don't divide evenly.
    """
    assert chunks("abcde", 2) == ["ab", "cd", "e"]


def test_chunks_size_one():
    """chunks() with size 1 returns individual characters.

    The smallest legal chunk size must still work, yielding each character as
    its own chunk, so callers can request the finest-grained split.
    """
    assert chunks("abc", 1) == ["a", "b", "c"]
