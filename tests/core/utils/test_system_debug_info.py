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
    """get_os_version names the OS it is running on.

    platform.system() is the internal name, which is "Darwin" on macOS, while
    platform.platform() -- what this function returns, and what a human reading
    a bug report wants -- leads with "macOS". Accept either token: the contract
    is that the OS is identifiable, not that one particular spelling is used.
    """
    os_version = system_debug_info.get_os_version()
    assert platform.system() in os_version or "macOS" in os_version


def test_get_ram_info_format():
    """get_ram_info returns a human-readable string with GB totals and used/free breakdown."""
    ram = system_debug_info.get_ram_info()
    assert "GB" in ram
    assert "used:" in ram
    assert "free:" in ram


def test_get_ram_info_converts_each_quantity_by_the_gigabyte_divisor():
    """Total, used and free are each the matching psutil field divided by 1GiB.

    The label-only assertion above cannot tell `vm.used` from `vm.free`, or
    1024**3 from a plain division, because the numbers are never compared. This
    pins the values: a swap between used and free, or a wrong divisor, changes
    a digit rather than removing a label.
    """
    gib = 1024**3
    fake_vm = SimpleNamespace(total=8 * gib, used=3 * gib, free=5 * gib)
    with mock.patch.object(
        system_debug_info.psutil, "virtual_memory", return_value=fake_vm
    ):
        ram = system_debug_info.get_ram_info()

    assert ram == "8.00 GB, used: 3.00, free: 5.00"


def test_get_ram_info_keeps_two_decimal_places():
    """Sub-gigabyte remainders are shown to two decimals, not truncated.

    Rounding is what makes the number readable in a bug report; a mutant that
    drops the precision specifier still produces a plausible-looking string.
    """
    gib = 1024**3
    fake_vm = SimpleNamespace(
        total=2 * gib, used=int(1.25 * gib), free=int(0.75 * gib)
    )
    with mock.patch.object(
        system_debug_info.psutil, "virtual_memory", return_value=fake_vm
    ):
        ram = system_debug_info.get_ram_info()

    assert "used: 1.25" in ram
    assert "free: 0.75" in ram


def test_get_pip_version_parses_successful_output():
    """get_pip_version extracts the version number from a successful pip --version subprocess call."""
    with mock.patch(
        "interpreter.core.utils.system_debug_info.subprocess.check_output",
        return_value=b"pip 24.0 from /usr/local/lib/python3.12/site-packages",
    ) as check_output:
        assert system_debug_info.get_pip_version() == "24.0"

    # The argv matters as much as the parsed output: mocking check_output returns
    # the same bytes for any command, so without this the version could be read
    # from any program at all and every test here would still pass.
    assert check_output.call_args.args[0] == ["pip", "--version"]


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


def test_get_cpu_info_returns_the_processor_string():
    """get_cpu_info reports whatever platform.processor() says.

    Kept as an equality against the platform call rather than a shape check: the
    function is a one-line passthrough, and a mutated return value (or a
    hardcoded string) is only visible if the test names the same source.
    """
    with mock.patch.object(
        system_debug_info.platform, "processor", return_value="Test CPU 9000"
    ):
        assert system_debug_info.get_cpu_info() == "Test CPU 9000"


def test_get_oi_version_pairs_command_output_with_the_package_version():
    """A successful `interpreter --version` is paired with the installed package version.

    The function returns a tuple, and callers index both halves (system_info
    renders them as "cmd: ..., pkg: ..."), so a swapped or flattened return
    would show up as mismatched labels rather than a crash.
    """
    with mock.patch.object(
        system_debug_info.subprocess,
        "check_output",
        return_value="Open Interpreter 0.4.3",
    ) as check_output:
        with mock.patch.object(
            system_debug_info, "version", return_value="0.4.3"
        ) as pkg_version:
            assert system_debug_info.get_oi_version() == (
                "Open Interpreter 0.4.3",
                "0.4.3",
            )

    # Both queries are pinned, not just their results. `interpreter --version`
    # reports the CLI's own version and `version()` reports the installed
    # distribution's, so asking either one the wrong question yields a plausible
    # but wrong pair.
    assert check_output.call_args.args[0] == ["interpreter", "--version"]
    assert check_output.call_args.kwargs.get("text") is True
    assert pkg_version.call_args.args[0] == "open-interpreter"


def test_get_oi_version_reports_a_command_failure_without_raising():
    """If `interpreter --version` fails, its error string becomes the cmd half.

    An exception escaping here would take down system_info, which is the debug
    path a user reaches for precisely when something else is already broken.
    """
    with mock.patch.object(
        system_debug_info.subprocess,
        "check_output",
        side_effect=FileNotFoundError("interpreter not on PATH"),
    ):
        with mock.patch.object(
            system_debug_info, "version", return_value="0.4.3"
        ):
            cmd, pkg = system_debug_info.get_oi_version()

    assert cmd == "interpreter not on PATH"
    assert pkg == "0.4.3"


def test_get_oi_version_reports_a_missing_package_as_none():
    """An uninstalled package yields None rather than raising PackageNotFoundError.

    The two halves are queried independently, so one succeeding must not mask
    the other failing — this pins that a missing package does not turn into an
    error string.
    """
    with mock.patch.object(
        system_debug_info.subprocess, "check_output", return_value="Open Interpreter 0.4.3"
    ):
        with mock.patch.object(
            system_debug_info,
            "version",
            side_effect=system_debug_info.PackageNotFoundError("open-interpreter"),
        ):
            cmd, pkg = system_debug_info.get_oi_version()

    assert cmd == "Open Interpreter 0.4.3"
    assert pkg is None


def test_get_oi_version_handles_both_halves_failing():
    """Command failure and missing package are reported independently.

    Both failure paths are caught separately in the implementation, so a mutant
    that drops one of the two try blocks would raise instead of reporting.
    """
    with mock.patch.object(
        system_debug_info.subprocess,
        "check_output",
        side_effect=OSError("no shell"),
    ):
        with mock.patch.object(
            system_debug_info,
            "version",
            side_effect=system_debug_info.PackageNotFoundError("open-interpreter"),
        ):
            cmd, pkg = system_debug_info.get_oi_version()

    assert cmd == "no shell"
    assert pkg is None


def _debug_interpreter(llm_overrides=None, **overrides):
    """A minimal stand-in for the attributes interpreter_info reads.

    `llm_overrides` is merged into the llm namespace rather than replacing it:
    interpreter_info renders every one of those attributes into its f-string, so
    replacing the namespace would drop the others and land in the blanket
    except instead of exercising the branch under test.
    """
    llm = dict(
        api_base=None,
        supports_vision=False,
        model=TEST_LLM_MODEL,
        supports_functions=True,
        context_window=8000,
        max_tokens=1000,
    )
    llm.update(llm_overrides or {})
    base = dict(
        offline=False,
        llm=SimpleNamespace(**llm),
        messages=[],
        system_message="test",
        auto_run=True,
        computer=SimpleNamespace(import_computer_api=False),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_interpreter_info_skips_curl_when_not_offline():
    """A non-offline interpreter is never probed with curl.

    The curl call shells out to whatever api_base is configured, so running it
    when offline is False would contact a hosted API from a debug helper. This
    pins both that curl is skipped and what the output says instead.
    """
    interpreter = _debug_interpreter(
        offline=False, llm_overrides={"api_base": "http://x"}
    )

    with mock.patch.object(system_debug_info.subprocess, "check_output") as co:
        out = system_debug_info.interpreter_info(interpreter)

    co.assert_not_called()
    assert "Curl output: Not local" in out


def test_interpreter_info_probes_the_api_base_when_offline():
    """An offline interpreter with an api_base is probed, and its output is shown.

    Offline mode with a local base (Ollama and friends) is the case this exists
    for, so the curl result is what tells a user whether their local server is
    actually reachable.
    """
    interpreter = _debug_interpreter(
        offline=True, llm_overrides={"api_base": "http://localhost:11434"}
    )

    with mock.patch.object(
        system_debug_info.subprocess, "check_output", return_value=b"pong"
    ) as co:
        out = system_debug_info.interpreter_info(interpreter)

    assert "curl http://localhost:11434" in str(co.call_args)
    assert "Curl output: b'pong'" in out


def test_interpreter_info_reports_a_failed_curl_without_raising():
    """A curl failure is reported as its error string, not propagated.

    A refused connection is the expected case when a local server is not
    running, so raising here would break the debug report exactly when it is
    most needed.
    """
    interpreter = _debug_interpreter(
        offline=True, llm_overrides={"api_base": "http://localhost:11434"}
    )

    with mock.patch.object(
        system_debug_info.subprocess,
        "check_output",
        side_effect=ConnectionRefusedError("nothing listening"),
    ):
        out = system_debug_info.interpreter_info(interpreter)

    assert "nothing listening" in out


def test_interpreter_info_truncates_a_long_message():
    """A message longer than 2000 characters is cut to 1000 before display.

    Chat history can be enormous and this string goes to a terminal, so the
    cap is what keeps the report readable. Asserting the exact cut length also
    pins that the bound is a truncation, not an ellipsis or a skip.
    """

    class LongMessage:
        def copy(self):
            return "y" * 3000

        def __str__(self):
            return "unused"

    interpreter = _debug_interpreter(messages=[LongMessage()])

    out = system_debug_info.interpreter_info(interpreter)

    assert "y" * 1000 in out
    assert "y" * 1001 not in out


def test_interpreter_info_keeps_a_short_message_intact():
    """A message under the cap is rendered whole.

    Guards against a mutant that truncates unconditionally, which the
    long-message test alone would not notice.
    """

    class ShortMessage:
        def copy(self):
            return "z" * 1500

        def __str__(self):
            return "unused"

    interpreter = _debug_interpreter(messages=[ShortMessage()])

    out = system_debug_info.interpreter_info(interpreter)

    assert "z" * 1500 in out


def test_interpreter_info_returns_a_message_when_the_interpreter_shape_is_unexpected():
    """An unreadable interpreter yields a fixed error string rather than raising.

    The blanket except is what keeps system_info usable against a half-built or
    mocked interpreter. Feeding it a message object with no .copy() reaches that
    except through the normal message loop.
    """
    interpreter = _debug_interpreter(messages=[object()])

    assert (
        system_debug_info.interpreter_info(interpreter)
        == "Error, couldn't get interpreter info"
    )


class _FixedLengthMessage:
    """A message whose copy() yields an exact-length string, for boundary tests."""

    def __init__(self, length, char="q"):
        self.length = length
        self.char = char

    def copy(self):
        return self.char * self.length

    def __str__(self):
        return "unused"


def test_a_message_exactly_at_the_cap_is_not_truncated():
    """2000 characters is kept whole: the guard is strictly greater than the cap.

    The boundary is the whole contract here. `> 2000` and `>= 2000` behave
    identically for any length except 2000 itself, so a test using an obviously
    oversized message passes under both and the off-by-one survives.
    """
    interpreter = _debug_interpreter(messages=[_FixedLengthMessage(2000)])

    out = system_debug_info.interpreter_info(interpreter)

    assert "q" * 2000 in out


def test_a_message_one_over_the_cap_is_truncated():
    """2001 characters is cut, so the guard is not off by one in the other direction."""
    interpreter = _debug_interpreter(messages=[_FixedLengthMessage(2001)])

    out = system_debug_info.interpreter_info(interpreter)

    assert "q" * 1000 in out
    assert "q" * 1001 not in out


def test_messages_are_joined_by_a_blank_line():
    """Consecutive messages are separated by exactly two newlines.

    The messages section reads as a transcript, so the separator is part of the
    output contract. Asserting only that both messages appear would let any
    separator — or a mangled one — through.
    """

    class Simple:
        def __init__(self, text):
            self.text = text

        def copy(self):
            return self.text

        def __str__(self):
            return "unused"

    interpreter = _debug_interpreter(messages=[Simple("FIRST"), Simple("SECOND")])

    out = system_debug_info.interpreter_info(interpreter)

    assert "FIRST\n\nSECOND" in out


def test_system_info_labels_the_command_and_package_versions_separately():
    """The report shows the CLI version under "cmd:" and the package under "pkg:".

    Both halves come from a tuple, so a swapped index still renders a
    well-formed line — and a reader would have no way to tell the two values had
    been exchanged.
    """
    interpreter = _debug_interpreter()
    with mock.patch.object(
        system_debug_info, "get_oi_version", return_value=("CMD_VERSION", "PKG_VERSION")
    ):
        with mock.patch("builtins.print") as fake_print:
            system_debug_info.system_info(interpreter)

    report = fake_print.call_args.args[0]
    assert "cmd: CMD_VERSION" in report
    assert "pkg: PKG_VERSION" in report
