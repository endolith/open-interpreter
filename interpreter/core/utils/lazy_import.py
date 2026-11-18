import importlib.util
import sys
import threading
import types

# Attributes answered from the placeholder itself, without executing the real
# import. Traceback printers and other introspection touch exactly these
# (__file__ via inspect.getmodule, once inspect needs the module object's
# repr). If answering them triggered the real import, a module that cannot
# load (e.g. pynput with no display) would raise inside the printer and bury
# the original error. Everything else performs the real import; internal
# helpers must bypass the load as well or they would recurse.
#
# __path__ is deliberately NOT safe: importing a submodule asks the parent
# package for __path__ before running its __init__. Answering that from the
# placeholder would let the submodule initialise against a package that never
# ran, which breaks circular-import bookkeeping (numpy: matplotlib imports
# numpy.linalg, which pulls numpy._core and numpy.dtypes ahead of
# numpy/__init__.py, and the partially initialised _core then fails to export
# busday_count). The import machinery is the intended consumer of __path__, so
# loading there is correct.
_LAZY_SAFE_ATTRS = frozenset({
    "__file__", "__name__", "__spec__", "__loader__",
    "__package__", "_load_real", "_real", "_real_error", "_load_lock",
})


class _LazyPlaceholder(types.ModuleType):
    """Placeholder that imports the real module on first real use.

    Introspection attributes (see _LAZY_SAFE_ATTRS) are served from the
    already-known spec without executing anything, so incidental touches
    never trigger — or choke on — the real import. The first real attribute
    access performs the import; if it fails, the placeholder is restored in
    sys.modules, the error is cached, and later real use re-raises the
    original error instead of a poisoned module.
    """

    def _load_real(self):
        """Execute the real import and return the real module.

        The result is cached on the placeholder. After this the placeholder
        never defers to whatever sys.modules happens to hold: tests and other
        tools routinely install a stub under the module name (pyautogui gets
        a MagicMock so headless imports work), and a placeholder that re-read
        sys.modules on every use hands that stub to its caller instead of the
        module it was created for. That turned the display tool's height
        property into "not enough values to unpack (expected 2, got 0)" from
        an empty MagicMock while the real pyautogui had loaded fine.
        """
        real = object.__getattribute__(self, "_real")
        if real is not None:
            return real
        name = self.__name__
        with self._load_lock:
            real = object.__getattribute__(self, "_real")
            if real is not None:
                return real
            if self._real_error is not None:
                raise self._real_error
            try:
                real = importlib.util.module_from_spec(self.__spec__)
                sys.modules[name] = real
                self.__spec__.loader.exec_module(real)
            except Exception as exc:
                # Restore the placeholder so introspection stays safe and the
                # next real use re-raises this same error.
                sys.modules[name] = self
                self._real_error = exc
                raise
            self._real = real
        return real

    def __getattribute__(self, attr):
        if attr in _LAZY_SAFE_ATTRS:
            return object.__getattribute__(self, attr)
        return getattr(object.__getattribute__(self, "_load_real")(), attr)


def lazy_import(name, optional=True):
    """Lazily import a module, specified by the name. Useful for optional packages, to speed up startup times.

    Returns a placeholder that performs the real import on first real use.
    Merely inspecting it (e.g. accessing __file__ while rendering a traceback)
    never executes the import, so a module that fails to load cannot break
    unrelated error reporting. Using it raises the real import error.
    """
    # Check if module is already imported
    if name in sys.modules:
        return sys.modules[name]

    # Find the module specification from the module name
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        # find_spec raises for a dotted name whose parent is missing, which is
        # still just "this module cannot be found" for an optional dependency.
        spec = None
    if spec is None:
        if optional:
            return None  # Do not raise an error if the module is optional
        else:
            raise ImportError(f"Module '{name}' cannot be found")

    # Create a placeholder with correct metadata; the real import happens on
    # the first real attribute use. Stock LazyLoader is not used because it
    # also executes the import for introspection attributes like __file__.
    module = importlib.util.module_from_spec(spec)
    module.__class__ = _LazyPlaceholder
    module._real = None
    module._real_error = None
    module._load_lock = threading.RLock()
    sys.modules[name] = module

    return module
