from interpreter.core.core import OpenInterpreter
from interpreter.terminal_interface.profiles import profiles


def test_profile_truncation_step_migrates_to_retention_ratio(monkeypatch):
    """Deprecated llm.truncation_step becomes llm.retention_ratio, not ignored.

    Profiles written before the rename set llm.truncation_step, which no longer
    exists on the Llm class. Without a mapping the validation warning fires and
    the setting is silently dropped, so the user's intent (enable cache-aware
    truncation) is lost. The migration removes truncation_step and sets the
    standard retention_ratio (0.8) instead, keeping trimming active.
    """
    interpreter = OpenInterpreter()
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "gpt-4.1",
            "truncation_step": 15000,
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")

    assert "truncation_step" not in profile["llm"]
    assert profile["llm"]["retention_ratio"] == 0.8
    assert interpreter.llm.retention_ratio == 0.8


def test_profile_retention_ratio_preserved(monkeypatch):
    """A modern profile's explicit retention_ratio is left untouched.

    If the user already set llm.retention_ratio, the deprecation mapping must not
    clobber it with the 0.8 default.
    """
    interpreter = OpenInterpreter()
    profile = {
        "version": profiles.OI_VERSION,
        "llm": {
            "model": "gpt-4.1",
            "retention_ratio": 0.9,
        },
    }

    profiles.apply_profile(interpreter, profile, profile_path="/tmp/fake.yaml")

    assert profile["llm"]["retention_ratio"] == 0.9
    assert interpreter.llm.retention_ratio == 0.9
