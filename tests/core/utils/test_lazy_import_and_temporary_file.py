import importlib
import os
from unittest import mock

import pytest

from interpreter.core.utils.lazy_import import lazy_import
from interpreter.core.utils.temporary_file import (
    cleanup_temporary_file,
    create_temporary_file,
)


def test_create_temporary_file_writes_contents(tmp_path, monkeypatch):
    """create_temporary_file writes the given contents and cleanup_temporary_file removes the file."""
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("hello", extension="txt")
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read() == "hello"
    cleanup_temporary_file(path)
    assert not os.path.exists(path)


def test_create_temporary_file_verbose(capsys, tmp_path, monkeypatch):
    """verbose=True prints creation and cleanup messages for temporary files."""
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("data", verbose=True)
    captured = capsys.readouterr()
    assert "Created temporary file" in captured.out
    cleanup_temporary_file(path, verbose=True)
    captured = capsys.readouterr()
    assert "Cleaning up temporary file" in captured.out


def test_cleanup_missing_file_does_not_raise(capsys):
    """cleanup_temporary_file on a missing path logs a warning instead of raising."""
    cleanup_temporary_file("/nonexistent/path/file.txt")
    captured = capsys.readouterr()
    assert "Could not clean up temporary file" in captured.out


def test_lazy_import_returns_existing_module():
    """lazy_import returns the already-imported module object for a valid module name."""
    import json as json_module

    assert lazy_import("json") is json_module


def test_lazy_import_optional_missing_returns_none():
    """lazy_import with optional=True returns None when the module cannot be imported."""
    assert lazy_import("this_module_definitely_does_not_exist_xyz", optional=True) is None


def test_lazy_import_required_missing_raises():
    """lazy_import with optional=False raises ImportError when the module cannot be found."""
    with pytest.raises(ImportError, match="cannot be found"):
        lazy_import("this_module_definitely_does_not_exist_xyz",
                    optional=False)


def test_lazy_import_default_is_optional():
    """Omitting optional defaults to True, returning None for a missing module.

    The default must stay True: a False default would turn every optional import
    into a hard failure, breaking startup when an optional dependency is absent.
    """
    assert lazy_import("this_module_definitely_does_not_exist_xyz") is None


def test_lazy_import_registers_the_module_in_sys_modules(monkeypatch):
    """The imported module is cached under its name in sys.modules.

    lazy_import's whole point is a single lookup; a mutation that stored None
    (module -> None, sys.modules[name] -> None) would make the second call
    return the wrong object and defeat the cache.

    Uses a module that is not already in sys.modules, so the call has to take
    the registration branch instead of returning an existing entry. "json" is
    imported by pytest itself, which left that branch unreached here.
    """
    import sys

    name = "fractions"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = lazy_import(name)

    assert sys.modules[name] is module
    assert lazy_import(name) is module


def test_create_temporary_file_verbose_defaults_to_silent(capsys, tmp_path, monkeypatch):
    """Omitting verbose leaves create and cleanup silent.

    A True default would print to stdout from every temp-file use, so the
    default is pinned by asserting nothing is printed.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("quiet", extension="txt")
    assert capsys.readouterr().out == ""
    cleanup_temporary_file(path)
    assert capsys.readouterr().out == ""


def test_extension_defaults_to_no_suffix(tmp_path, monkeypatch):
    """With no extension the file has no ".None" or trailing-dot suffix.

    The suffix is f".{extension}" when extension is truthy and "" otherwise; a
    mutation that appended ".None" or dropped the conditional would create
    misnamed files. Asserting the extracted suffix is empty rather than the
    absence of two specific endings, so an unrelated appended suffix is caught
    too.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("x")
    assert os.path.splitext(path)[1] == ""
    cleanup_temporary_file(path)


def test_extension_is_appended_as_a_dot_suffix(tmp_path, monkeypatch):
    """A provided extension becomes a ".py"-style suffix on the file name.

    The suffix must use the extension verbatim; a wrapped or dropped extension
    would produce files the language tooling cannot recognise.
    """
    monkeypatch.chdir(tmp_path)
    path = create_temporary_file("print(1)", extension="py")
    assert path.endswith(".py")
    cleanup_temporary_file(path)


def test_lazy_import_defers_loading(tmp_path, monkeypatch):
    """lazy_import does not execute the module body until an attribute is read.

    The earlier version of this test only checked the module's contents work,
    which passes identically for eager and deferred loading and so pinned
    nothing about deferral. Here the module body has an observable side effect
    (it writes a marker file), which lets the test assert the body has *not*
    run after lazy_import and *has* run once an attribute is touched.
    """
    import sys

    marker = tmp_path / "executed.marker"
    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_probe.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('executed')\n"
        "VALUE = 42\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))

    name = "oi_lazy_probe"
    monkeypatch.delitem(sys.modules, name, raising=False)

    module = lazy_import(name)
    assert not marker.exists(), "module body ran during lazy_import; loading was eager"

    assert module.VALUE == 42
    assert marker.exists(), "reading an attribute did not execute the module body"



def test_create_temporary_file_raises_when_creation_fails(capsys, monkeypatch):
    """A failed write raises after printing, rather than returning None.

    Fixed in #433 for #399: the `except` used to print and fall off the end,
    so the caller got `None` — indistinguishable from success at the call
    site, and `scan_code` passed it straight to `os.path.dirname`. Now the
    error is re-raised after printing, so callers see the real failure.
    """
    monkeypatch.setattr(
        "tempfile.NamedTemporaryFile",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        create_temporary_file("x = 1", "py")

    captured = capsys.readouterr().out
    assert "Could not create temporary file." in captured
    assert "disk full" in captured, "the underlying reason is printed for the user"


def test_create_temporary_file_failure_does_not_leave_a_file_behind(tmp_path, monkeypatch, capsys):
    """A failed creation leaves nothing on disk to clean up later.

    `cleanup_temporary_file` is called on the same path, so a file that existed
    despite the failure would be orphaned rather than cleaned.
    """
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(
        "tempfile.NamedTemporaryFile",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        create_temporary_file("x = 1", "py")

    assert list(tmp_path.iterdir()) == [], "a failed creation must not leave files"



def _install_broken_lazy_module(tmp_path, monkeypatch, request):
    """Create a module whose body raises, and return its lazy placeholder.

    The module counts how often its body runs via a builtins attribute, so a
    test can tell "was executed" from "was only inspected" without relying on
    a marker file (which would survive a failed import).
    """
    import builtins
    import sys

    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_broken.py").write_text(
        "import builtins\n"
        "builtins.OI_LAZY_PROBE_RUNS = getattr(builtins, 'OI_LAZY_PROBE_RUNS', 0) + 1\n"
        "raise RuntimeError('lazy probe import failed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))
    monkeypatch.setattr(builtins, "OI_LAZY_PROBE_RUNS", 0, raising=False)
    monkeypatch.delitem(sys.modules, "oi_lazy_broken", raising=False)
    # lazy_import registers a placeholder that outlives monkeypatch's undo
    # (delitem only removes an entry that existed). Drop the placeholder so a
    # later test cannot import a module whose file the tmp_path has since
    # deleted.
    request.addfinalizer(lambda: sys.modules.pop("oi_lazy_broken", None))

    return lazy_import("oi_lazy_broken")


def test_lazy_import_introspection_does_not_execute_the_module(tmp_path, monkeypatch, request):
    """Inspecting a lazy module must not run its body, even when importing it fails.

    Traceback rendering reads the module's __file__, __name__ and repr. If
    those trigger the real import, a module that cannot load (pynput without a
    display is the motivating case) raises inside the printer and buries the
    original error. The placeholder answers them from the spec instead.
    """
    import builtins

    module = _install_broken_lazy_module(tmp_path, monkeypatch, request)
    assert module is not None

    assert module.__name__ == "oi_lazy_broken"
    assert module.__file__.endswith("oi_lazy_broken.py")
    assert module.__spec__ is not None
    assert "oi_lazy_broken" in repr(module)
    assert builtins.OI_LAZY_PROBE_RUNS == 0, "introspection executed the module body"


def test_lazy_import_package_runs_init_before_submodule(tmp_path, monkeypatch):
    """Importing a submodule of a lazy package runs the package body first.

    The import machinery asks the parent package for __path__ before it
    executes the child module. Answering that from the placeholder would let
    the child initialise against a package whose body never ran, which breaks
    circular-import bookkeeping: numpy hit exactly that when matplotlib
    imported numpy.linalg, which pulled numpy._core and numpy.dtypes ahead of
    numpy/__init__.py, and the partially initialised _core then could not
    export busday_count. The fixture mirrors that shape — the package body
    imports a name from a child that is still executing — so a safe-__path__
    placeholder raises the same "partially initialized module" ImportError
    instead of loading the package.
    """
    import builtins
    import sys

    package_dir = tmp_path / "pkg_probe"
    package_dir.mkdir()
    package = package_dir / "oi_lazy_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import builtins\n"
        "builtins.OI_LAZY_PKG_INIT_RUNS = (\n"
        "    getattr(builtins, 'OI_LAZY_PKG_INIT_RUNS', 0) + 1\n"
        ")\n"
        "VALUE = 'ready'\n"
        "from .mid import MID_EXPORT\n",
        encoding="utf-8",
    )
    (package / "mid.py").write_text(
        "from . import leaf\n"
        "MID_EXPORT = 'exported'\n",
        encoding="utf-8",
    )
    (package / "leaf.py").write_text(
        "from oi_lazy_pkg import VALUE\n"
        "LEAF = VALUE + '-leaf'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(package_dir))
    monkeypatch.setattr(builtins, "OI_LAZY_PKG_INIT_RUNS", 0, raising=False)
    for name in ("oi_lazy_pkg", "oi_lazy_pkg.mid", "oi_lazy_pkg.leaf"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = lazy_import("oi_lazy_pkg")
    assert module is not None, "the fixture package must be importable"

    mid = importlib.import_module("oi_lazy_pkg.mid")
    assert mid.MID_EXPORT == "exported"
    assert builtins.OI_LAZY_PKG_INIT_RUNS == 1, (
        "the package body must run exactly once, before the submodule it imports"
    )


def test_lazy_import_reports_broken_parent_package_instead_of_absent(tmp_path, monkeypatch):
    """A package whose own import fails must not look like a missing optional module.

    find_spec imports the parent package of a dotted name, so a parent that
    raises because one of *its* dependencies is missing surfaces as
    ModuleNotFoundError naming that dependency. Treating every such error as
    "not installed" makes lazy_import hand back None for an optional reference
    and hides the real error until an unrelated feature breaks on it. Only a
    missing requested name, or one of its parents, means the module is absent.
    """
    import sys

    package_dir = tmp_path / "pkg_probe"
    package_dir.mkdir()
    package = package_dir / "oi_lazy_brokenpkg"
    package.mkdir()
    (package / "__init__.py").write_text("import oi_lazy_missing_dep\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 1", encoding="utf-8")
    monkeypatch.syspath_prepend(str(package_dir))
    for missing in (
        "oi_lazy_brokenpkg",
        "oi_lazy_brokenpkg.child",
        "oi_lazy_missing_dep",
        "oi_lazy_absent_pkg",
        "oi_lazy_absent_pkg.child",
    ):
        monkeypatch.delitem(sys.modules, missing, raising=False)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        lazy_import("oi_lazy_brokenpkg.child", optional=True)
    assert excinfo.value.name == "oi_lazy_missing_dep", (
        "the dependency that the package needs must be reported, not swallowed"
    )

    assert lazy_import("oi_lazy_absent_pkg.child", optional=True) is None, (
        "a genuinely absent parent is still optional"
    )


def test_lazy_import_failed_use_raises_original_error_and_caches_it(tmp_path, monkeypatch, request):
    """Real use raises the import's own error once, keeps the placeholder, and does not retry.

    Restoring the placeholder in sys.modules keeps other importers from seeing
    a half-built module, and caching the exception makes later real use raise
    the same error instead of re-running a body that already failed.
    """
    import builtins
    import sys

    module = _install_broken_lazy_module(tmp_path, monkeypatch, request)
    assert "oi_lazy_broken" in repr(module)

    with pytest.raises(RuntimeError, match="lazy probe import failed"):
        module.VALUE
    assert builtins.OI_LAZY_PROBE_RUNS == 1
    assert sys.modules["oi_lazy_broken"] is module, "the placeholder must be restored"

    with pytest.raises(RuntimeError, match="lazy probe import failed"):
        module.VALUE
    assert builtins.OI_LAZY_PROBE_RUNS == 1, "a failed import must not run a second time"
    assert "oi_lazy_broken" in repr(module), "introspection stays safe after a failure"


def test_lazy_import_ignores_stub_installed_after_creation(tmp_path, monkeypatch):
    """A stub parked in sys.modules must not hijack a lazy placeholder.

    tests/computer_use/tools/test_computer_tool.py stubs sys.modules
    ["pyautogui"] with a MagicMock so computer.py can be imported without a
    display, and that stub outlives the module. The display tool is imported
    later, so its lazy pyautogui placeholder sees the stub in sys.modules
    before its first real use. A placeholder that re-read sys.modules then
    handed the mock to callers: display.height unpacked MagicMock.size() (an
    empty iterator) and collection died with "not enough values to unpack
    (expected 2, got 0)" on Windows and macOS, although the real pyautogui
    loads fine. The placeholder owns its load and keeps answering from the
    module it imported itself.
    """
    import sys

    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_hijack.py").write_text(
        "SIZE = (1920, 1080)\n"
        "\n"
        "def size():\n"
        "    return SIZE\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))
    name = "oi_lazy_hijack"
    monkeypatch.delitem(sys.modules, name, raising=False)

    module = lazy_import(name)
    stub = mock.MagicMock()
    monkeypatch.setitem(sys.modules, name, stub)

    assert not isinstance(module.SIZE, mock.Mock), "the sys.modules stub answered instead"
    assert module.SIZE == (1920, 1080)
    assert module.size() == (1920, 1080)


def test_lazy_import_forwards_writes_and_deletes_to_the_real_module(tmp_path, monkeypatch):
    """Assigning or deleting an attribute edits the real module, not the placeholder.

    Reads are served from the real module, so a write kept on the placeholder
    would never be visible; a delete of a name that only exists on the real
    module would raise AttributeError. Both must load the module and apply to
    it, so monkeypatch.setattr on a lazy reference behaves like editing the
    imported module.
    """
    import sys

    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_writes.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(module_dir))
    name = "oi_lazy_writes"
    monkeypatch.delitem(sys.modules, name, raising=False)

    module = lazy_import(name)
    assert module.VALUE == 1

    module.VALUE = 99
    assert module.VALUE == 99, "the assignment did not reach the real module"

    del module.VALUE
    assert not hasattr(module, "VALUE"), "the deletion did not reach the real module"


def test_lazy_import_repeated_failure_does_not_grow_the_traceback(tmp_path, monkeypatch, request):
    """Every retry of a cached import error raises the original traceback, not a longer one.

    The placeholder caches the exception so a failed module body is not run
    again. Re-raising that same instance appends the caller's frames to its
    traceback on each attempt, so a long-running process that catches the
    error and retries accumulates frames for every retry. Restoring the
    original traceback keeps the report bounded and identical each time.
    """
    import traceback

    module = _install_broken_lazy_module(tmp_path, monkeypatch, request)

    attempts = []
    for _ in range(3):
        with pytest.raises(RuntimeError, match="lazy probe import failed") as excinfo:
            module.VALUE
        attempts.append(len(traceback.extract_tb(excinfo.value.__traceback__)))

    assert attempts[1] == attempts[2], f"traceback grew on retry: {attempts}"


def test_lazy_import_evicts_placeholder_before_delegating_to_importlib():
    """The placeholder is unparked before the delegated import runs.

    importlib returns whatever sys.modules already holds instead of loading
    it, and a single-phase extension's create_module hands back the module
    registered under its name. Leaving the placeholder parked therefore made
    the delegated import hand the placeholder back, so the real module never
    ran and its first attribute read recursed. The fake finder below mimics
    that single-phase behaviour: create_module returns whatever is registered
    under the name, which is the placeholder unless _load_real evicted it
    first.
    """
    import importlib.abc
    import importlib.machinery
    import sys

    name = "oi_lazy_single_phase_probe"
    sys.modules.pop(name, None)

    class SinglePhaseLoader(importlib.abc.Loader):
        def create_module(self, spec):
            # Single-phase initialisation reuses a module already registered
            # under the name, so looking it up here reproduces that exactly.
            return sys.modules.get(spec.name)

        def exec_module(self, module):
            module.VALUE = 42

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != name:
                return None
            return importlib.machinery.ModuleSpec(name, SinglePhaseLoader(), origin="oi")

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        placeholder = lazy_import(name)
        assert placeholder is not None
        assert placeholder.VALUE == 42
        assert placeholder._real is not placeholder, "the placeholder was handed back as the real module"
        assert placeholder._real is sys.modules[name]
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop(name, None)


def test_lazy_import_does_not_cache_a_reentrant_partial_module():
    """A module read reentrantly while still initialising is never cached as loaded.

    The loader lock is reentrant, so a module body that touches its own
    placeholder during exec_module gets the partially built module back.
    Publishing that module as the loaded result would make every later access
    return half-built state without consulting the cached error, and a body
    that failed afterwards would leave the placeholder looking loaded. The
    body below reads itself and then fails; the placeholder must keep no
    cached module and raise the stored error on the next real access.
    """
    import importlib.abc
    import importlib.machinery
    import sys

    name = "oi_lazy_reentrant_probe"
    sys.modules.pop(name, None)
    holder = {}

    class ReentrantLoader(importlib.abc.Loader):
        def exec_module(self, module):
            module.VALUE = 1
            module.PLACEHOLDER = holder["placeholder"]
            # Reentrant read while this module is still initialising: it must
            # return the partial module instead of recursing or blocking.
            assert module.PLACEHOLDER.VALUE == 1
            raise RuntimeError("lazy reentrant probe failed")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != name:
                return None
            return importlib.machinery.ModuleSpec(name, ReentrantLoader(), origin="oi")

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        placeholder = lazy_import(name)
        holder["placeholder"] = placeholder
        with pytest.raises(RuntimeError, match="lazy reentrant probe failed"):
            placeholder.VALUE
        assert placeholder._real is None, "the partial module was cached as the real one"
        assert sys.modules[name] is placeholder, "the placeholder was not restored"
        with pytest.raises(RuntimeError, match="lazy reentrant probe failed"):
            placeholder.VALUE
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop(name, None)


def test_lazy_import_restores_placeholder_after_abnormal_exit(tmp_path, monkeypatch, request):
    """An abnormal exit from the module body restores the placeholder and is not cached.

    Only ordinary import failures are cached because they describe the
    module; a BaseException says nothing about it, so a later access must
    retry instead of replaying a cached error or treating a half-built module
    as loaded. The probe raises a BaseException subclass rather than
    KeyboardInterrupt because pytest aborts the whole session on
    KeyboardInterrupt even when the test catches it.
    """
    import builtins
    import sys

    class ProbeAbort(BaseException):
        """Stands in for an interrupt so a regression fails a test, not the session."""

    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_abort.py").write_text(
        "import builtins\n"
        "builtins.OI_LAZY_ABORT_RUNS = getattr(builtins, 'OI_LAZY_ABORT_RUNS', 0) + 1\n"
        "raise builtins.OI_LAZY_ABORT('aborted during import')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))
    monkeypatch.setattr(builtins, "OI_LAZY_ABORT", ProbeAbort, raising=False)
    monkeypatch.setattr(builtins, "OI_LAZY_ABORT_RUNS", 0, raising=False)
    monkeypatch.delitem(sys.modules, "oi_lazy_abort", raising=False)
    request.addfinalizer(lambda: sys.modules.pop("oi_lazy_abort", None))

    module = lazy_import("oi_lazy_abort")

    with pytest.raises(ProbeAbort):
        module.VALUE
    assert builtins.OI_LAZY_ABORT_RUNS == 1
    assert sys.modules["oi_lazy_abort"] is module, "the placeholder must be restored"

    with pytest.raises(ProbeAbort):
        module.VALUE
    assert builtins.OI_LAZY_ABORT_RUNS == 2, "an abnormal exit must not be cached"


def test_lazy_import_plain_import_waits_for_the_load(tmp_path, monkeypatch, request):
    """A plain import of a lazy module waits for its initialisation to finish.

    The real load runs importlib and holds the per-module import lock, so
    another thread that imports the same name blocks in importlib until the
    body completes. Driving the spec by hand registered the module before
    exec_module, and a second thread read sys.modules directly, so it could
    receive a partially initialised module.
    """
    import builtins
    import importlib
    import sys
    import threading
    import time

    release = threading.Event()
    module_dir = tmp_path / "probe"
    module_dir.mkdir()
    (module_dir / "oi_lazy_lock.py").write_text(
        "import builtins\n"
        "builtins.OI_LAZY_LOCK_STARTED = True\n"
        "builtins.OI_LAZY_LOCK_RELEASE.wait(5)\n"
        "VALUE = 42\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(module_dir))
    monkeypatch.setattr(builtins, "OI_LAZY_LOCK_STARTED", False, raising=False)
    monkeypatch.setattr(builtins, "OI_LAZY_LOCK_RELEASE", release, raising=False)
    monkeypatch.delitem(sys.modules, "oi_lazy_lock", raising=False)
    request.addfinalizer(lambda: sys.modules.pop("oi_lazy_lock", None))

    module = lazy_import("oi_lazy_lock")
    loaded = []
    loader = threading.Thread(target=lambda: loaded.append(module.VALUE))
    loader.start()
    deadline = time.monotonic() + 5
    while not builtins.OI_LAZY_LOCK_STARTED and time.monotonic() < deadline:
        time.sleep(0.01)
    assert builtins.OI_LAZY_LOCK_STARTED, "the lazy load never reached the module body"

    entered = threading.Event()
    imported = []

    def plain_import():
        entered.set()
        imported.append(importlib.import_module("oi_lazy_lock"))

    importer = threading.Thread(target=plain_import)
    importer.start()
    assert entered.wait(5), "the plain import thread did not start"
    time.sleep(0.2)
    assert imported == [], "a plain import returned before the body finished"

    release.set()
    loader.join(5)
    importer.join(5)
    assert not loader.is_alive() and not importer.is_alive(), "a thread did not finish"
    assert loaded == [42]
    assert imported and imported[0].VALUE == 42, "the returned module was incomplete"
