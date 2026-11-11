import subprocess
import sys

THIRD_PARTY = ("inquirer", "yaml", "requests", "platformdirs", "send2trash")

_SCRIPT = """
import sys

sys.argv = ["interpreter", {flag!r}]
import interpreter.terminal_interface.start_terminal_interface  # noqa: F401
print(",".join(name for name in {modules!r} if name in sys.modules))
"""


def _loaded_third_party(flag):
    """Return the third-party modules loaded by importing the CLI entry module."""
    script = _SCRIPT.format(flag=flag, modules=THIRD_PARTY)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_help_import_path_loads_no_third_party_packages():
    """Importing the CLI entry module under --help loads no third-party packages.

    --help and --version exit inside argparse before any flag handling runs, so
    the packages only some flags need must stay at their point of use. Loading
    them at import time makes every invocation pay for packages the run may
    never touch.
    """
    assert _loaded_third_party("--help") == ""


def test_version_import_path_loads_no_third_party_packages():
    """--version, the other immediate-exit flag, gets the same guarantee."""
    assert _loaded_third_party("--version") == ""


_CORE_SCRIPT = """
import sys

import interpreter.core.core  # noqa: F401
print(",".join(name for name in {modules!r} if name in sys.modules))
"""


def test_core_import_leaves_the_local_setup_wizard_deferred():
    """Importing the core module must not pull in the local-model wizard.

    `local_setup.py` imports inquirer, wget and psutil at module scope; an
    interpreter that never calls `interpreter.local_setup()` should not pay for
    them. The name stays a module attribute so `Interpreter.local_setup` and its
    patch point keep working.
    """
    script = _CORE_SCRIPT.format(
        modules=("interpreter.terminal_interface.local_setup", "inquirer", "wget")
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_core_import_leaves_pyyaml_deferred():
    """Importing the core module must not pull in PyYAML.

    `execution_allowlist.py` only needs YAML when a rules file is read or
    written, which happens solely in auto-run "allowlist" mode. The default
    "prompt" mode never loads rules, so importing PyYAML at module scope makes
    every run pay ~11 ms for a file it will not open.
    """
    script = _CORE_SCRIPT.format(modules=("yaml", "yaml._yaml"))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


_VERSION_SCRIPT = """
import sys

sys.argv = ["interpreter", "--version"]
from interpreter.terminal_interface.start_terminal_interface import main

try:
    main()
except SystemExit as exc:
    code = exc.code
else:
    code = 0
loaded = [name for name in {modules!r} if name in sys.modules]
print(f"EXIT={{code}} LOADED={{','.join(loaded)}}")
"""


def test_version_flag_exits_cleanly_without_third_party_modules():
    """`main()` under --version exits 0 without loading the deferred packages.

    The fast-exit stub must satisfy every attribute the argument definitions
    read before argparse runs; missing one turned --version into an
    AttributeError instead of a version print.
    """
    script = _VERSION_SCRIPT.format(modules=THIRD_PARTY)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "EXIT=0 LOADED=" in result.stdout
    assert "Open Interpreter" in result.stdout


_JUPYTER_SCRIPT = """
import sys

import interpreter.core.terminal.languages.jupyter_language  # noqa: F401
print("jupyter_client" in sys.modules)
"""


def test_jupyter_language_import_leaves_kernel_manager_deferred():
    """Importing the language module must not import jupyter_client.

    jupyter_client costs ~169 ms to import and is only needed once a kernel is
    actually started, which happens in JupyterLanguage.__init__, not while the
    Terminal merely records which languages exist. It must therefore stay out
    of sys.modules until that point.
    """
    result = subprocess.run(
        [sys.executable, "-c", _JUPYTER_SCRIPT], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_core_import_leaves_telemetry_and_debug_info_deferred():
    """Importing the core module must not pull in telemetry or debug-info helpers.

    Both modules import `importlib.metadata` at module scope and telemetry also
    imports `requests`; system_debug_info adds psutil. They are only needed when
    a chat request is actually sent or debug output is requested, so importing
    the core module must not pay for them. The names stay patchable module
    attributes (`send_telemetry` in core, `system_info` in the debug-info
    module) so their call sites keep working.
    """
    script = _CORE_SCRIPT.format(
        modules=(
            "interpreter.core.utils.telemetry",
            "interpreter.core.utils.system_debug_info",
            "psutil",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_core_import_leaves_network_and_browser_stack_deferred():
    """Importing the core module must not pull in the HTTP and browser stacks.

    `requests` costs ~84 ms in-tree and drags in urllib3, which in turn imports
    `importlib.metadata`. Selenium, webdriver-manager and html2text are only
    needed when browser automation or error rendering actually runs. Constructing
    an interpreter pays for none of them, so importing the core module must
    leave them out of sys.modules. Their module attributes stay available (as
    deferred proxies) so existing patch points such as `providers.requests` and
    `web.requests` keep working.
    """
    script = _CORE_SCRIPT.format(
        modules=(
            "requests",
            "urllib3",
            "importlib.metadata",
            "selenium",
            "webdriver_manager",
            "html2text",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""

_DISPLAY_SCRIPT = """
import sys

import interpreter.core.toolbox.display.display  # noqa: F401
real = []
for name in {modules!r}:
    mod = sys.modules.get(name)
    if mod is not None and type(mod).__name__ != "_LazyPlaceholder":
        real.append(name)
print(",".join(real))
"""


def test_display_import_leaves_pyautogui_unimported():
    """Importing the display toolbox must not import pyautogui or its dependencies.

    pyautogui pulls in mouseinfo, pyscreeze, pymsgbox and Xlib (~70 ms in-tree),
    and the module used to force that import with a `pyautogui.size()` display
    probe at import time. That probe is gone, so the name is a deferred
    placeholder and importing the module (which happens while the Interpreter
    is constructed) must not load the real package. The headless failure mode
    stays AttributeError: Display resolves pyautogui through `_pyautogui()`,
    which returns None when the package cannot load, so `inspect.getmembers()`
    keeps skipping width/height when it lists the toolbox tools.
    """
    script = _DISPLAY_SCRIPT.format(
        modules=("pyautogui", "pyscreeze", "mouseinfo", "pymsgbox", "Xlib")
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


_PLACEHOLDER_SCRIPT = """
import sys

import interpreter.core.core  # noqa: F401
real = []
for name in {modules!r}:
    mod = sys.modules.get(name)
    if mod is not None and type(mod).__name__ != "_LazyPlaceholder":
        real.append(name)
print(",".join(real))
"""


def test_core_import_leaves_the_tokenizer_deferred():
    """Importing the core module must not load tiktoken's vocabulary data.

    `count_tokens`, `toolbox/ai` and `llm/utils/cache_aware_trim` all used
    `encoding_for_model`, but each caller falls back to a character estimate
    when tokenization fails, so `%tokens`/`%cost` output and chunking still
    work. tiktoken is a hard dependency, so deferring it cannot change
    behaviour; it only moves the BPE load from startup to the first count.
    """
    script = _PLACEHOLDER_SCRIPT.format(modules=("tiktoken", "tiktoken._tiktoken"))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_core_import_leaves_the_markdown_renderer_deferred():
    """Importing the core module must not load the markdown/pygments stack.

    `rich.markdown` pulls in markdown-it-py and pygments, and `rich.syntax`
    (used by the code block) pulls in pygments too; importing them at module
    scope made every run pay ~40 ms for a renderer it only needs once a block
    is actually drawn. `CodeBlock` and `MessageBlock` stay real classes so
    `isinstance` and `mock.patch.object(terminal_interface, "CodeBlock")` keep
    working; only the heavy imports inside their methods are deferred.
    """
    script = _CORE_SCRIPT.format(
        modules=(
            "rich.markdown",
            "rich.syntax",
            "markdown_it",
            "pygments",
            "interpreter.terminal_interface.utils.streaming_markdown",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_core_import_leaves_the_rich_console_stack_deferred():
    """Importing the core module must not load rich.console and its dependents.

    rich.console costs ~42 ms in-tree: it pulls in rich.themes, rich.text,
    rich.style and rich._log_render, and every `rich.panel`/`rich.padding`/
    `rich.table`/`rich.syntax` import drags it in transitively. None of that is
    needed to construct an Interpreter -- only to draw a live block. The block
    classes stay real module attributes (so `isinstance` checks and
    `mock.patch.object(terminal_interface, "CodeBlock")` keep working); only
    their rendering methods import the Rich stack. The `rich` package itself
    stays because `from rich import print` is cheap and used as a patch point.
    """
    script = _CORE_SCRIPT.format(
        modules=(
            "rich.console",
            "rich.live",
            "rich.table",
            "rich.markup",
            "rich.syntax",
            "rich.markdown",
            "rich.panel",
            "rich.padding",
            "rich.box",
            "rich.text",
            "rich.style",
            "rich.themes",
            "rich._log_render",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_core_import_leaves_the_macos_sms_helpers_deferred():
    """Importing the core module must not load sqlite3/plistlib for the SMS tool.

    The toolbox builds an `SMS` instance on every platform, but `SMS.get()` is
    macOS-only. Importing `sqlite3` and `plistlib` at module scope made Linux
    and Windows pay for `sqlite3.dbapi2` and `xml.parsers.expat` on startup.
    They are lazily imported now, and the module names stay patchable.
    """
    script = _PLACEHOLDER_SCRIPT.format(
        modules=("sqlite3", "sqlite3.dbapi2", "plistlib", "xml.parsers.expat")
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_core_import_leaves_difflib_deferred():
    """Importing the core module must not pull in difflib for the edit diff.

    `tools/file_edit.py` and `toolbox/files/files.py` import difflib only to
    build a close-match hint or a unified edit diff, both of which run after a
    tool call. difflib costs ~2 ms of startup on its own (heapq is loaded anyway
    via `queue`, so its subtree is not the whole 5 ms). It is imported at the
    point of use so construction does not pay for a path most runs never hit.
    """
    script = _CORE_SCRIPT.format(modules=("difflib",))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
