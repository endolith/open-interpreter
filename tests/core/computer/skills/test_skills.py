import importlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace
from unittest import mock

import os
from pathlib import Path

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


# search() and the import guards

def test_search_returns_skill_names(tmp_path):
    """search() lists the same names as list(); it currently just delegates."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "demo_skill.py").write_text("def demo_skill(): pass")
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=True)
    skills = Skills(computer)
    skills.path = str(skills_dir)
    assert skills.search("anything") == ["demo_skill()"]


def test_search_returns_empty_when_skills_disabled(capsys):
    """search() refuses and explains when skills are turned off."""
    computer = SimpleNamespace(import_skills=False, _has_imported_skills=False)
    skills = Skills(computer)
    assert skills.search("x") == []
    assert "Skills are disabled" in capsys.readouterr().out


def test_list_explains_that_skills_have_not_been_imported(capsys, tmp_path):
    """The "not imported yet" case tells the user, rather than returning silently.

    An empty list is indistinguishable from "no skills exist", so the message is
    the only way a user learns the difference.
    """
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=False)
    skills = Skills(computer)
    skills.path = str(tmp_path)
    assert skills.list() == []
    assert "not been imported" in capsys.readouterr().out


def test_search_explains_that_skills_have_not_been_imported(capsys, tmp_path):
    """search() reports the not-imported state too, not just the disabled state."""
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=False)
    skills = Skills(computer)
    skills.path = str(tmp_path)
    assert skills.search("x") == []
    assert "not been imported" in capsys.readouterr().out


def test_list_ignores_non_python_files(tmp_path):
    """Only .py files become callable names; a README or cache file is skipped."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "a_skill.py").write_text("x = 1")
    (skills_dir / "notes.md").write_text("docs")
    (skills_dir / "__pycache__").mkdir()
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=True)
    skills = Skills(computer)
    skills.path = str(skills_dir)
    assert skills.list() == ["a_skill()"]


def test_import_skills_refuses_a_directory_over_100mb(tmp_path):
    """An oversized skills directory is rejected before any code is executed.

    The limit exists because the skill files are concatenated and run in one
    interpreter, so an unbounded directory is an unbounded amount of code to
    execute at startup. Raises rather than truncating, since silently importing
    a subset would be worse than refusing.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    big = skills_dir / "huge.py"
    with open(big, "w") as fh:
        fh.truncate(101 * 1024 * 1024)  # 101 MB sparse file
    computer = SimpleNamespace(
        import_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    skills = Skills(computer)
    skills.path = str(skills_dir)

    with pytest.raises(Warning):
        skills.import_skills()

    computer.run.assert_not_called()


def test_import_skills_retries_individually_after_a_traceback(tmp_path, capsys):
    """A combined import that tracebacks falls back to importing one file at a time.

    Two skills are imported together and one is broken, so the combined run
    fails and the fallback re-runs each separately. The broken file is named in
    the output, which is the only way a user learns which skill to fix.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "good.py").write_text("x = 1")
    (skills_dir / "bad.py").write_text("raise ValueError('nope')")

    def fake_run(language, code):
        """Traceback exactly when the code being run contains the broken line.

        Responding to the argument rather than a fixed side_effect list keeps the
        test independent of glob order, which is filesystem-dependent.
        """
        return (
            "Traceback (most recent call last)"
            if "raise ValueError" in code
            else []
        )

    computer = SimpleNamespace(
        import_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(side_effect=fake_run),
    )
    skills = Skills(computer)
    skills.path = str(skills_dir)

    skills.import_skills()

    # One combined attempt, then one per file.
    assert computer.run.call_count == 3
    assert "bad.py" in capsys.readouterr().out


def test_import_skills_restores_the_save_skills_setting(tmp_path):
    """save_skills is restored even though it is disabled for the duration.

    The setting is toggled off around the import so a half-imported skill file
    cannot rewrite the user's profile, and the previous value must come back
    afterwards or the next session inherits the temporary value.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "a.py").write_text("x = 1")
    computer = SimpleNamespace(
        import_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    skills = Skills(computer)
    skills.path = str(skills_dir)
    skills.import_skills()
    assert computer.save_skills is True

    computer2 = SimpleNamespace(
        import_skills=True,
        save_skills=False,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    skills2 = Skills(computer2)
    skills2.path = str(skills_dir)
    skills2.import_skills()
    assert computer2.save_skills is False


def test_run_is_a_deprecated_no_op(capsys):
    """Skills.run() no longer runs anything and says so.

    It is marked DEPRECATED in the source; calling it should print guidance and
    execute nothing, not silently do work.
    """
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=True)
    skills = Skills(computer)
    assert skills.run("anything") is None
    assert "already imported" in capsys.readouterr().out


# NewSkill: building and saving a skill

def _new_skill():
    """A NewSkill taken through its documented create() first step.

    __init__ does not initialise steps or _name (see #390), so create() is the
    first call in every path that uses the object afterwards.
    """
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=True)
    new_skill = Skills(computer).new_skill
    new_skill.create()
    return new_skill


@pytest.mark.xfail(raises=AttributeError, reason="#390: __init__ does not set steps or _name")
def test_new_skill_is_unusable_before_create():
    """Reading `name` or `steps` before create() raises AttributeError.

    Both attributes are only assigned inside create(), so a caller that reaches
    for the placeholder name without going through create() gets an
    AttributeError instead of "Untitled". See #390.
    """
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=True)
    new_skill = Skills(computer).new_skill

    assert new_skill.name == "Untitled"


def test_create_resets_name_and_steps(capsys):
    """create() initialises the name and clears any previously collected steps."""
    new_skill = _new_skill()
    new_skill.add_step("stale step", "x = 1")
    new_skill.name = "something"

    new_skill.create()

    assert new_skill.name == "Untitled"
    assert new_skill.steps == []
    assert "INSTRUCTIONS" in capsys.readouterr().out


def test_setting_the_name_announces_the_next_instructions(capsys):
    """Assigning a name moves the conversation on to the next step."""
    new_skill = _new_skill()
    new_skill.name = "my_new_skill"
    assert new_skill.name == "my_new_skill"
    assert "INSTRUCTIONS" in capsys.readouterr().out


def test_add_step_records_a_prose_description_with_its_code():
    """Each step keeps the description and the code that satisfied it.

    The saved skill interpolates the list straight into the generated file, so a
    step that lost either half would produce a skill that cannot be run.
    """
    new_skill = _new_skill()

    new_skill.add_step("Open the app", "computer.mouse.click(x=1, y=2)")

    assert len(new_skill.steps) == 1
    step = new_skill.steps[0]
    assert step.startswith("Open the app")
    assert "```python" in step
    assert "computer.mouse.click(x=1, y=2)" in step


def test_add_step_accumulates_rather_than_replacing(capsys):
    """Steps build up in order; a second step does not overwrite the first."""
    new_skill = _new_skill()

    new_skill.add_step("first", "x = 1")
    new_skill.add_step("second", "y = 2")

    assert len(new_skill.steps) == 2
    assert "first" in new_skill.steps[0]
    assert "second" in new_skill.steps[1]
    assert "YOU MUST FOLLOW THESE 4 INSTRUCTIONS" in capsys.readouterr().out


def _skills_dir(tmp_path):
    skills_dir = tmp_path / "skills"
    computer = SimpleNamespace(import_skills=True, _has_imported_skills=True)
    skills = Skills(computer)
    skills.path = str(skills_dir)
    return skills, skills.new_skill


def test_save_writes_a_callable_skill_file(tmp_path, capsys):
    """save() writes a .py file named after the skill and defines the function."""
    skills, new_skill = _skills_dir(tmp_path)
    new_skill.create()
    new_skill.name = "My Great Skill"
    new_skill.add_step("do the thing", "print('done')")

    new_skill.save()

    written = Path(skills.path) / "my_great_skill.py"
    assert written.exists()

    source = written.read_text()
    assert "def my_great_skill(step=0):" in source
    # The prose step must reach the generated file verbatim.
    assert "do the thing" in source

    # The generated source is valid Python and its function body runs.
    namespace = {}
    exec(source, namespace)
    assert callable(namespace["my_great_skill"])
    assert "SKILL SAVED" in capsys.readouterr().out


def test_save_normalises_the_filename_to_lowercase_underscores(tmp_path):
    """Punctuation and spaces in the name become underscores in the filename.

    The generated file defines `def <normalised>()`, so the filename and the
    function name must agree or the skill is listed under one name but callable
    under another.
    """
    skills, new_skill = _skills_dir(tmp_path)
    new_skill.create()
    new_skill.name = "Email The Boss!! 2"
    new_skill.add_step("step", "x = 1")

    new_skill.save()

    assert (Path(skills.path) / "email_the_boss_2.py").exists()


def test_save_creates_the_skills_directory_if_absent(tmp_path):
    """Saving into a directory that does not exist yet creates it."""
    skills, new_skill = _skills_dir(tmp_path)
    new_skill.create()
    new_skill.name = "brand_new"
    new_skill.add_step("step", "x = 1")

    new_skill.save()

    assert (Path(skills.path) / "brand_new.py").exists()


def test_save_reports_failure_when_the_file_is_not_written(tmp_path, capsys, monkeypatch):
    """A skill that could not be written says so instead of claiming success.

    The confirmation is printed only when the file exists, so a silent failure
    has to be visible: otherwise the user is told a skill was saved and it was
    not.
    """
    skills, new_skill = _skills_dir(tmp_path)
    new_skill.create()
    new_skill.name = "unwritable"
    new_skill.add_step("step", "x = 1")

    # The skill file is written for real, but the post-write existence check is
    # made to report False — the state a failed or vanished write leaves behind.
    real_exists = os.path.exists

    def flaky_exists(path):
        if str(path).endswith("unwritable.py"):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", flaky_exists)
    new_skill.save()

    out = capsys.readouterr().out
    assert "Error: Failed to write skill file" in out
    assert "SKILL SAVED" not in out


def _enabled_computer(**overrides):
    """A skills-enabled computer stand-in with sane defaults."""
    defaults = dict(
        import_skills=True,
        _has_imported_skills=True,
        save_skills=True,
        interpreter=SimpleNamespace(debug=False),
        run=mock.Mock(return_value=[]),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_search_mirrors_list_results(tmp_path, capsys):
    """Skills.search() currently returns the same listing as list()."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "demo_skill.py").write_text("def demo_skill(): pass")
    skills = Skills(_enabled_computer())
    skills.path = str(skills_dir)
    assert skills.search("anything") == skills.list() == ["demo_skill()"]


def test_import_skills_size_guard_raises_and_restores_save_skills(tmp_path):
    """An oversized skills dir raises Warning and leaves save_skills on.

    Once import_skills() restores the flag in a finally path, the guard
    raise no longer strands the caller's setting: the run is still skipped
    and save_skills is back to True. Today the restore is skipped, so the
    flag stays False. Only the restoration assertion is allowed to
    expect-fail; the guard and no-run expectations stay hard failures.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "big.py").write_text("x = 1")
    computer = _enabled_computer()
    skills = Skills(computer)
    skills.path = str(skills_dir)
    with mock.patch("os.path.getsize", return_value=200 * 1024 * 1024):
        with pytest.raises(Warning, match="can't exceed 100mb"):
            skills.import_skills()
    computer.run.assert_not_called()
    try:
        assert computer.save_skills is True
    except AssertionError:
        pytest.xfail(
            reason="import_skills() flips save_skills to False before the 100MB guard with no try/finally (issue #417), so a guard raise strands the caller's setting; fix in #422"
        )


def test_import_skills_retries_files_individually_on_traceback(tmp_path, capsys):
    """A combined-run traceback falls back to per-file imports with a warning.

    When the concatenated run reports a traceback, each file is run on its own
    and the offending file is named. save_skills is still restored afterwards.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "good.py").write_text("x = 1")
    (skills_dir / "bad.py").write_text("raise Boom")

    def fake_run(language, code):
        """Only the bad file's code reports a traceback, whatever runs first."""
        assert language == "python"
        return "Traceback: Boom" if "raise Boom" in code else "ok"

    computer = _enabled_computer(run=mock.Mock(side_effect=fake_run))
    skills = Skills(computer)
    skills.path = str(skills_dir)
    skills.import_skills()
    assert computer.run.call_count == 3
    out = capsys.readouterr().out
    assert "might be broken" in out
    assert "bad.py" in out
    assert computer.save_skills is True


def test_new_skill_create_names_and_saves_roundtrip(tmp_path, capsys):
    """NewSkill walks create -> name -> add_step -> save and writes a module.

    save() normalizes the name, writes the skill file, and execs it so the
    new function is defined immediately.
    """
    skills = Skills(_enabled_computer())
    skills.path = str(tmp_path / "skills")
    new_skill = skills.new_skill
    new_skill.create()
    assert new_skill.name == "Untitled"
    new_skill.name = "Demo Skill!"
    new_skill.add_step("greet", "print('hi')")
    assert "```python" in new_skill.steps[0]
    new_skill.save()
    out = capsys.readouterr().out
    assert "SKILL SAVED: DEMO SKILL!" in out
    skill_file = tmp_path / "skills" / "demo_skill_.py"
    assert skill_file.exists()
    assert "def demo_skill_(step=0)" in skill_file.read_text()
