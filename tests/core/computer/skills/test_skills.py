import importlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from interpreter.core.computer.skills.skills import Skills

SKILLS_MODULE = "interpreter.core.computer.skills.skills"


# 3 regression tests for issue #3: lazy pynput import masking AttributeError.
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


# Other tests

def test_list_returns_empty_when_skills_disabled(capsys):
    """Skills.list() returns an empty list when import_skills is disabled."""
    computer = SimpleNamespace(import_skills=False, _has_imported_skills=False)
    skills = Skills(computer)
    assert skills.list() == []


def test_list_returns_skill_names(tmp_path):
    """Skills.list() returns callable signatures for each .py file in the skills directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "demo_skill.py").write_text("def demo_skill(): pass")
    computer = SimpleNamespace(
        import_skills=True,
        _has_imported_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    skills = Skills(computer)
    skills.path = str(skills_dir)
    result = skills.list()
    assert result == ["demo_skill()"]


def test_import_skills_runs_python_files(tmp_path):
    """import_skills() executes each skill file via computer.run and enables save_skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill_file = skills_dir / "skill_a.py"
    skill_file.write_text("x = 1")
    computer = SimpleNamespace(
        import_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    skills = Skills(computer)
    skills.path = str(skills_dir)
    skills.import_skills()
    computer.run.assert_called_once_with("python",
                                         skill_file.read_text() + "\n")
    assert computer.save_skills is True


def test_import_skills_skips_when_disabled():
    """import_skills() does nothing when import_skills is False on the computer."""
    computer = SimpleNamespace(import_skills=False, run=mock.Mock())
    skills = Skills(computer)
    skills.import_skills()
    computer.run.assert_not_called()


def _computer(tmp_path, **overrides):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    computer = SimpleNamespace(
        import_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    computer.__dict__.update(overrides)
    skills = Skills(computer)
    skills.path = str(skills_dir)
    return computer, skills, skills_dir


def test_import_skills_restores_save_skills_after_success(tmp_path):
    """import_skills() leaves save_skills as it found it once skills load (issue #417).

    The flag is disabled only for the duration of the import run so that
    replayed skill code cannot trigger a recursive/side-effecting save, and
    the user's setting is put back afterwards.
    """
    computer, skills, _ = _computer(tmp_path)
    computer.save_skills = True
    skills.import_skills()
    assert computer.save_skills is True


def test_import_skills_restores_save_skills_when_import_raises(tmp_path):
    """A raise during the skill import still restores save_skills (issue #417).

    Before the try/finally the assignment back was skipped on the error path,
    leaving the interpreter permanently in save_skills=False.
    """
    computer, skills, _ = _computer(tmp_path)
    computer.run = mock.Mock(side_effect=RuntimeError("boom"))
    computer.save_skills = True

    with pytest.raises(RuntimeError, match="boom"):
        skills.import_skills()

    assert computer.save_skills is True


def test_import_skills_restores_save_skills_when_paths_too_large(tmp_path):
    """The 100mb cap warning also restores save_skills (issue #417)."""
    computer, skills, skills_dir = _computer(tmp_path)
    big = skills_dir / "big.py"
    big.write_bytes(b"0" * 64)
    computer.save_skills = True

    with mock.patch("interpreter.core.computer.skills.skills.os.path.getsize", return_value=200 * 1024 * 1024):
        with pytest.raises(Warning, match="can't exceed 100mb"):
            skills.import_skills()

    assert computer.save_skills is True


def test_import_skills_restores_save_skills_when_interpreter_raises(tmp_path):
    """A raise from outside computer.run (e.g. interpreter.debug access) restores save_skills."""
    computer, skills, _ = _computer(tmp_path)
    computer.save_skills = True
    computer.run = mock.Mock(side_effect=MemoryError("kaboom"))

    with pytest.raises(MemoryError):
        skills.import_skills()

    assert computer.save_skills is True


def test_import_skills_broken_skill_reports_and_restores(capsys, tmp_path):
    """A skill producing a traceback is reported and save_skills is restored (issue #417)."""
    computer, skills, skills_dir = _computer(tmp_path)
    (skills_dir / "broken.py").write_text("def broken(): pass")
    computer.save_skills = True
    computer.run = mock.Mock(
        side_effect=[
            "Traceback (most recent call last): ...",
            "Traceback (most recent call last): ...",
        ]
    )

    skills.import_skills()

    out = capsys.readouterr().out
    assert "might be broken" in out
    assert computer.save_skills is True


def test_import_skills_broken_skill_debug_names_the_file(tmp_path, capsys):
    """In debug mode the per-file fallback prints each skill's code (covers that branch)."""
    computer, skills, skills_dir = _computer(tmp_path)
    (skills_dir / "broken.py").write_text("def broken(): pass")
    computer.interpreter = SimpleNamespace(debug=True)
    computer.save_skills = True
    computer.run = mock.Mock(
        side_effect=[
            "Traceback (most recent call last): ...",
            "Traceback (most recent call last): ...",
        ]
    )

    skills.import_skills()

    out = capsys.readouterr().out
    assert "IMPORTING SKILL:" in out
    assert "def broken" in out
    assert computer.save_skills is True


def test_import_skills_debug_prints_code(tmp_path, capsys):
    """debug=True prints the concatenated code before running it (covers the debug branch)."""
    computer, skills, skills_dir = _computer(tmp_path)
    (skills_dir / "s.py").write_text("y = 2")
    computer.interpreter = SimpleNamespace(debug=True)
    computer.save_skills = True

    skills.import_skills()

    out = capsys.readouterr().out
    assert "IMPORTING SKILLS" in out
    assert computer.save_skills is True
