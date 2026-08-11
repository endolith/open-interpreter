import platform
import re
from types import SimpleNamespace
from unittest import mock

from interpreter.core.utils import system_debug_info
from tests.helpers import TEST_LLM_MODEL


def test_get_python_version_matches_current_interpreter():
    """get_python_version reports the same version string as the running interpreter."""
    version = system_debug_info.get_python_version()
    assert version == platform.python_version()


def test_get_os_version_includes_platform_name():
    """get_os_version includes the platform.system() name (e.g. Linux, Darwin)."""
    os_version = system_debug_info.get_os_version()
    assert platform.system() in os_version


def test_get_ram_info_format():
    """get_ram_info returns a human-readable string with GB totals and used/free breakdown."""
    ram = system_debug_info.get_ram_info()
    assert "GB" in ram
    assert "used:" in ram
    assert "free:" in ram


def test_get_pip_version_parses_successful_output():
    """get_pip_version extracts the version number from a successful pip --version subprocess call."""
    with mock.patch(
        "interpreter.core.utils.system_debug_info.subprocess.check_output",
        return_value=b"pip 24.0 from /usr/local/lib/python3.12/site-packages",
    ):
        assert system_debug_info.get_pip_version() == "24.0"


def test_get_pip_version_stringifies_errors():
    """On failure, get_pip_version returns str(exception) instead of raising."""
    with mock.patch(
        "interpreter.core.utils.system_debug_info.subprocess.check_output",
        side_effect=FileNotFoundError("pip not found"),
    ):
        assert system_debug_info.get_pip_version() == "pip not found"


def test_system_info_runs_without_error(capsys):
    """system_info prints a debug summary including Python version and model without raising."""
    interpreter = SimpleNamespace(
        offline=False,
        llm=SimpleNamespace(
            api_base=None,
            supports_vision=False,
            model=TEST_LLM_MODEL,
            supports_functions=True,
            context_window=8000,
            max_tokens=1000,
        ),
        messages=[],
        system_message="test",
        auto_run=True,
        computer=SimpleNamespace(import_computer_api=False),
    )
    system_debug_info.system_info(interpreter)
    captured = capsys.readouterr().out
    assert "Python Version" in captured
    assert TEST_LLM_MODEL in captured
