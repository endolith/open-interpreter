import importlib.util
import sys
import threading
import types

# Attributes answered from the placeholder itself, without executing the real
# import. Traceback printers and other introspection touch exactly these
# (__file__ via inspect.getmodule, etc.). If answering them triggered the
# real import, a module that cannot load (e.g. pynput with no display) would
# detonate inside the printer and bury the original error. Everything else
# performs the real import. Internal helpers must also bypass the load.
_LAZY_SAFE_ATTRS = frozenset({
    "__file__", "__path__", "__name__", "__spec__", "__loader__",
    "__package__", "_load_real", "_real_error", "_load_lock",
})


class _LazyPlaceholder(types.ModuleType):
    """Module placeholder that imports the real module on first real use.

    Introspection attributes (see _LAZY_SAFE_ATTRS) are served from the
    already-known spec without executing anything, so incidental touches
    never trigger — or choke on — the real import. The first real attribute
    access executes the import; if that fails, the placeholder is restored
    in sys.modules, the error is cached, and every later real use re-raises
    the original error.
    """

    def _load_real(self):
        """Execute the real import and return the real module."""
        name = self.__name__
        with self._load_lock:
            current = sys.modules.get(name)
            if current is not None and current is not self:
                return current  # Someone loaded it for real meanwhile.
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
            return real

    def __getattribute__(self, attr):
        if attr in _LAZY_SAFE_ATTRS:
            return object.__getattribute__(self, attr)
        return getattr(object.__getattribute__(self, "_load_real")(), attr)


def lazy_import(name, optional=True):
    """Lazily import a module, specified by the name. Useful for optional packages, to speed up startup times.

    Returns a placeholder that performs the real import on first real use.
    Merely inspecting it (e.g. __file__ during traceback rendering) never
    executes the import, so a module that fails to load cannot break
    unrelated error reporting. Using it raises the real import error.
    """
    # Check if module is already imported
    if name in sys.modules:
        return sys.modules[name]

    # Find the module specification from the module name
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        if optional:
            return None  # Do not raise an error if the module is optional
        else:
            raise ImportError(f"Module '{name}' cannot be found")

    # Placeholder with correct metadata; real import happens on first real use.
    module = importlib.util.module_from_spec(spec)
    module.__class__ = _LazyPlaceholder
    module._real_error = None
    module._load_lock = threading.RLock()
    sys.modules[name] = module

    return module
