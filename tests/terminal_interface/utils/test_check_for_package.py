import sys
from unittest import mock

from interpreter.terminal_interface.utils.check_for_package import check_for_package


def test_check_for_package_returns_true_for_installed():
    """check_for_package returns True when the package is importable."""
    assert check_for_package("json") is True


def test_check_for_package_returns_false_for_missing():
    """check_for_package returns False when the package cannot be imported."""
    assert check_for_package("definitely_not_a_real_package_xyz") is False


def test_already_imported_package_short_circuits_before_find_spec():
    """A package already in sys.modules returns True without probing for a spec.

    The sys.modules check is the first branch, so a mutated order would call
    find_spec (or exec_module) for something that is already loaded. Patching
    find_spec to raise makes that ordering visible: if the early return is
    removed or moved, this test errors instead of passing.
    """
    with mock.patch(
        "interpreter.terminal_interface.utils.check_for_package.importlib.util.find_spec",
        side_effect=AssertionError("find_spec must not be called"),
    ):
        sys.modules["json"] = mock.MagicMock()
        try:
            assert check_for_package("json") is True
        finally:
            sys.modules.pop("json", None)


def test_missing_spec_returns_false_without_exec():
    """When find_spec yields None, the function returns False and never execs.

    module_from_spec(None) would raise, so this also pins that the None spec is
    short-circuited by the `else` rather than falling into the try block.
    """
    with mock.patch(
        "interpreter.terminal_interface.utils.check_for_package.importlib.util.find_spec",
        return_value=None,
    ):
        with mock.patch(
            "interpreter.terminal_interface.utils.check_for_package.importlib.util.module_from_spec"
        ) as mfs:
            assert check_for_package("some_absent_package") is False
    mfs.assert_not_called()


def test_import_error_during_module_execution_returns_false():
    """A module whose loader raises ImportError yields False, not an exception.

    check_for_package exists so callers can test optional dependencies without
    a try/except of their own, so an ImportError escaping here would break every
    caller the moment an optional dependency is half-installed.
    """
    spec = mock.MagicMock()
    spec.loader.exec_module.side_effect = ImportError("no compiled module")
    # check_for_package assigns sys.modules[package] *before* exec_module runs, so
    # a failing exec still leaves an entry behind. Clear it on the way in as well
    # as the way out: the helper's first branch returns True for anything already
    # in sys.modules, which would make this test pass once and fail on any re-run
    # in the same process (mutmut does exactly that).
    sys.modules.pop("broken_optional_dep", None)
    with mock.patch(
        "interpreter.terminal_interface.utils.check_for_package.importlib.util.find_spec",
        return_value=spec,
    ):
        try:
            assert check_for_package("broken_optional_dep") is False
        finally:
            sys.modules.pop("broken_optional_dep", None)


def test_successful_load_registers_the_module_in_sys_modules():
    """A package that execs cleanly returns True and is left in sys.modules.

    The helper populates sys.modules as a side effect; dropping that line would
    still return True here, so the registration has to be asserted directly.
    """
    spec = mock.MagicMock()
    with mock.patch(
        "interpreter.terminal_interface.utils.check_for_package.importlib.util.find_spec",
        return_value=spec,
    ):
        sys.modules.pop("fake_optional_dep", None)
        assert check_for_package("fake_optional_dep") is True
        try:
            assert "fake_optional_dep" in sys.modules
            # Registering the name is not the same as executing it: with the
            # spec fully mocked, a helper that skipped exec_module would still
            # register the name and still return True, leaving a package that
            # cannot be imported afterwards.
            spec.loader.exec_module.assert_called_once()
        finally:
            sys.modules.pop("fake_optional_dep", None)


def test_the_built_module_is_registered_and_executed():
    """The object from module_from_spec is both stored and handed to the loader.

    Returning True is not enough to show the module was built correctly: a
    mutant that passes None to exec_module, or stores None in sys.modules, still
    returns True while leaving a package that cannot be imported afterwards.
    Asserting object identity is what distinguishes a real module from a
    placeholder.
    """
    spec = mock.MagicMock()
    module = mock.MagicMock(name="built_module")
    sys.modules.pop("identity_optional_dep", None)
    with mock.patch(
        "interpreter.terminal_interface.utils.check_for_package.importlib.util.find_spec",
        return_value=spec,
    ):
        with mock.patch(
            "interpreter.terminal_interface.utils.check_for_package.importlib.util.module_from_spec",
            return_value=module,
        ):
            assert check_for_package("identity_optional_dep") is True
            try:
                assert sys.modules["identity_optional_dep"] is module
                spec.loader.exec_module.assert_called_once_with(module)
            finally:
                sys.modules.pop("identity_optional_dep", None)
