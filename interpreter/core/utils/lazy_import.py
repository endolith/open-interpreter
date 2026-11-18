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
    "__package__", "_load_real", "_real", "_real_error", "_real_error_tb",
    "_load_lock",
})


def _is_initializing(module):
    """Whether the module's body is still executing (or about to start).

    importlib sets ``spec._initializing`` on the module it registers in
    sys.modules for the duration of ``exec_module`` and clears it in
    ``_load``'s finally block, so it is the authoritative signal that this
    module has not finished loading. Reading it off the spec touches no
    module attribute and imports nothing.
    """
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


class _LazyPlaceholder(types.ModuleType):
    """Placeholder that imports the real module on first real use.

    Introspection attributes (see _LAZY_SAFE_ATTRS) are served from the
    already-known spec without executing anything, so incidental touches
    never trigger — or choke on — the real import. The first real attribute
    access performs the import; if it fails, the placeholder is restored in
    sys.modules, the error is cached, and later real use re-raises the
    original error instead of a poisoned module.

    Reads, writes and deletions of everything else are forwarded to the real
    module, so ``monkeypatch.setattr`` on a lazy reference edits the module
    callers actually use instead of a placeholder copy that nothing reads.
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

        The import itself is delegated to importlib rather than driven by
        hand. importlib creates a fresh module object (a single-phase
        extension's create_module may hand back the module already registered
        under the name, which here is this placeholder), takes the per-module
        import lock so ordinary ``import name`` calls wait for the same
        initialisation, and cleans sys.modules up itself when the body fails.
        Anything parked under the name is evicted first, so a stub installed
        after this placeholder was created cannot win; a real module that
        arrived in the meantime is adopted instead of re-executed.

        A module parked under the name while its body is still executing is
        never adopted: the body may yet fail, and caching it would pin a
        partially built module. That case is handed to importlib instead,
        whose per-module lock makes a load on another thread wait for the
        body to finish, while a reentrant load on this thread gets the
        partial module back. Only a module that finished initialising is
        cached on the placeholder.

        The placeholder's own lock keeps the ordinary first use serialised,
        but it is not part of Python's import lock graph: two threads that
        each pull a mutually importing lazy module through a placeholder they
        already hold can still deadlock. importlib's own LazyLoader has the
        same limitation, since it holds its loader lock across exec_module.
        """
        real = object.__getattribute__(self, "_real")
        if real is not None:
            return real
        name = self.__name__
        with self._load_lock:
            real = object.__getattribute__(self, "_real")
            if real is not None:
                return real
            error = object.__getattribute__(self, "_real_error")
            if error is not None:
                # Re-raising the cached instance directly would keep appending
                # this caller's frames to the traceback it was first raised
                # with. Reset to the original traceback on every use instead.
                raise error.with_traceback(
                    object.__getattribute__(self, "_real_error_tb")
                )
            try:
                parked = sys.modules.get(name)
                if parked is not self and isinstance(parked, types.ModuleType):
                    if _is_initializing(parked):
                        # The body is running elsewhere, or this thread is
                        # reentering it. Hand off to importlib so its module
                        # lock serialises cross-thread loads and a reentrant
                        # load gets the partial module back.
                        real = importlib.import_module(name)
                    else:
                        real = parked
                else:
                    sys.modules.pop(name, None)
                    real = importlib.import_module(name)
            except BaseException as exc:
                # Restore the placeholder on every abnormal exit, so
                # introspection stays safe. An interrupt is not a property of
                # the module, so it is not cached: the next use may retry.
                sys.modules[name] = self
                if isinstance(exc, Exception):
                    self._real_error = exc
                    self._real_error_tb = exc.__traceback__
                raise
            if not _is_initializing(real):
                self._real = real
        return real

    def __getattribute__(self, attr):
        if attr in _LAZY_SAFE_ATTRS:
            return object.__getattribute__(self, attr)
        return getattr(object.__getattribute__(self, "_load_real")(), attr)

    def __setattr__(self, attr, value):
        if attr in _LAZY_SAFE_ATTRS:
            object.__setattr__(self, attr, value)
        else:
            setattr(object.__getattribute__(self, "_load_real")(), attr, value)

    def __delattr__(self, attr):
        if attr in _LAZY_SAFE_ATTRS:
            object.__delattr__(self, attr)
        else:
            delattr(object.__getattribute__(self, "_load_real")(), attr)


def lazy_import(name, optional=True):
    """Lazily import a module, specified by the name. Useful for optional packages, to speed up startup times.

    Returns a placeholder that performs the real import on first real use.
    Touching only the metadata in _LAZY_SAFE_ATTRS (e.g. __file__ while
    rendering a traceback) never executes the import, so a module that fails
    to load cannot break unrelated error reporting. Anything else — including
    dir(), which reads __dict__ — performs the real import and raises its
    error.
    """
    # Check if module is already imported
    if name in sys.modules:
        return sys.modules[name]

    # Find the module specification from the module name
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError as exc:
        # find_spec imports the parent package of a dotted name. A missing
        # module or parent means "not found" for an optional dependency, but a
        # parent that failed because one of its own dependencies is missing
        # must not be reported as absent: the caller would get None (or a
        # misleading "cannot be found") instead of the real error.
        missing = exc.name or ""
        if not missing or not (name == missing or name.startswith(missing + ".")):
            raise
        spec = None
    except ValueError:
        # A relative name has no package to resolve against.
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
