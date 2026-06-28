import platform
import re
from types import SimpleNamespace
from unittest import mock

from interpreter.core.utils import system_debug_info

from tests.conftest import TEST_LLM_MODEL


def test_get_python_version_matches_current_interpreter():
    version = system_debug_info.get_python_version()
    assert version == platform.python_version()


def test_get_os_version_includes_platform_name():
    os_version = system_debug_info.get_os_version()
    assert platform.system() in os_version


def test_get_ram_info_format():
    ram = system_debug_info.get_ram_info()
    assert "GB" in ram
    assert "used:" in ram
    assert "free:" in ram


def test_get_pip_version_parses_successful_output():
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
    with mock.patch(
        "interpreter.core.utils.system_debug_info.get_package_mismatches",
        return_value="",
    ):
        system_debug_info.system_info(interpreter)
    captured = capsys.readouterr().out
    assert "Python Version" in captured
    assert TEST_LLM_MODEL in captured
