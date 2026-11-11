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

    def __setattr__(self, attr, value):
        # Writing through to the real module keeps `lazy_import("litellm")` from
        # silently configuring a placeholder: `litellm.suppress_debug_info = True`
        # has to land on the module that is actually used. It also makes
        # monkeypatching a lazily imported module behave like patching a normal
        # one, because reads already delegate to the real module.
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


class _DeferredProxy:
    """Delegation shared by the deferred stand-ins below.

    Attribute reads, assignments and deletions reach the object the proxy
    stands in for, so configuration and monkeypatching behave as they would on
    a real import.
    """

    __slots__ = ()

    def _resolve(self):
        raise NotImplementedError

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)

    def __setattr__(self, attr, value):
        setattr(self._resolve(), attr, value)

    def __delattr__(self, attr):
        delattr(self._resolve(), attr)


class _DeferredModule(_DeferredProxy):
    """Stand-in for a module that imports itself on first real use.

    lazy_import() builds its placeholder from importlib.util.find_spec, and
    resolving the spec of a dotted name imports its parent package. That makes
    `lazy_import("matplotlib.pyplot")` pay for the whole matplotlib import just
    to answer where pyplot lives. This proxy keeps only the name and runs
    importlib.import_module on first use, so a dotted submodule costs nothing
    until it is actually needed.
    """

    __slots__ = ("_name",)

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def __repr__(self):
        return f"<deferred module {self._name!r}>"

    def _resolve(self):
        return importlib.import_module(self._name)


class _DeferredAttribute(_DeferredProxy):
    """Stand-in for one attribute of a module, resolved on first real use.

    Used for `from package.module import name` where the module is only needed
    at call time. Keeps the module-global name patchable without importing
    anything while the Interpreter is constructed.
    """

    __slots__ = ("_module_name", "_attribute_name")

    def __init__(self, module_name, attribute_name):
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_attribute_name", attribute_name)

    def __repr__(self):
        return f"<deferred attribute {self._module_name}.{self._attribute_name}>"

    def _resolve(self):
        return getattr(importlib.import_module(self._module_name), self._attribute_name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


def lazy_import_deferred(name):
    """Lazily import module `name` without resolving its import spec now.

    Unlike lazy_import(), this works for dotted names without importing their
    parent packages while the placeholder is created. The module is imported
    for real on first use, and a missing module raises its real import error
    then rather than at creation time.
    """
    if name in sys.modules:
        return sys.modules[name]
    return _DeferredModule(name)


def lazy_import_attribute(module_name, attribute_name):
    """Lazily provide `attribute_name` from `module_name`, resolved on first use.

    The module-global keeps its normal name and stays patchable, but nothing is
    imported until the attribute is read or called.
    """
    return _DeferredAttribute(module_name, attribute_name)
