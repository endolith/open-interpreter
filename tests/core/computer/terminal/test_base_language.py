from interpreter.core.computer.terminal.base_language import BaseLanguage


def test_base_language_run_returns_lmc_output():
    """BaseLanguage.run returns an LMC console output dict echoing the code."""
    lang = BaseLanguage()
    result = lang.run("print('hi')")
    assert result == {
        "type": "console",
        "format": "output",
        "content": "print('hi')",
    }


def test_base_language_stop_and_terminate_are_noops():
    """stop() and terminate() do not raise and have no observable effect."""
    lang = BaseLanguage()
    lang.stop()
    lang.terminate()
