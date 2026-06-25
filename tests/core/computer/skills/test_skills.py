"""Regression tests for issue #3: lazy pynput import masking AttributeError."""

import importlib
import importlib.abc
import importlib.util
import sys

import pytest

SKILLS_MODULE = "interpreter.core.computer.skills.skills"


def _install_headless_lazy_module(module_name):
    """Register a PEP 562 lazy module that fails like pynput on headless SSH."""

    class HeadlessLoader(importlib.abc.Loader):
        def exec_module(self, module):
            raise ImportError(
                'this platform is not supported: ("failed to acquire X connection: '
                'Bad display name \\"\\"", DisplayNameError(""))'
            )

    loader = HeadlessLoader()
    spec = importlib.util.spec_from_loader(module_name, loader)
    lazy_loader = importlib.util.LazyLoader(loader)
    spec.loader = lazy_loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    lazy_loader.exec_module(module)
    return module


def _reload_skills_module():
    if SKILLS_MODULE in sys.modules:
        return importlib.reload(sys.modules[SKILLS_MODULE])
    return importlib.import_module(SKILLS_MODULE)


def test_skills_import_does_not_register_pynput():
    """skills.py must not register pynput in sys.modules at import time (issue #3)."""
    saved_modules = {}
    for name in ("pynput", SKILLS_MODULE):
        if name in sys.modules:
            saved_modules[name] = sys.modules.pop(name)

    try:
        _reload_skills_module()
        assert "pynput" not in sys.modules
    finally:
        for name, mod in saved_modules.items():
            sys.modules[name] = mod


def test_headless_lazy_pynput_masks_unrelated_attribute_error():
    """Simulate headless SSH: lazy pynput in sys.modules can mask AttributeError."""
    module_name = "_test_headless_pynput_fake"
    saved = sys.modules.pop(module_name, None)

    try:
        _install_headless_lazy_module(module_name)

        with pytest.raises(ImportError, match="not supported"):
            try:
                raise AttributeError(
                    "'AnswerResult' object has no attribute 'answer'"
                )
            except AttributeError:
                # IPython traceback formatting can touch modules in sys.modules
                getattr(sys.modules[module_name], "keyboard")
    finally:
        sys.modules.pop(module_name, None)
        if saved is not None:
            sys.modules[module_name] = saved


def test_attribute_error_unmasked_when_skills_does_not_preload_pynput():
    """Importing skills leaves pynput out of sys.modules so errors stay visible."""
    saved = {}
    for name in ("pynput", SKILLS_MODULE):
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    try:
        _reload_skills_module()
        assert "pynput" not in sys.modules

        with pytest.raises(AttributeError, match="answer"):
            raise AttributeError(
                "'AnswerResult' object has no attribute 'answer'"
            )
    finally:
        for name, mod in saved.items():
            sys.modules[name] = mod
