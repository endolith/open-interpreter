import pytest
from types import SimpleNamespace
from unittest import mock

from interpreter.core.computer.os.os import Os


def test_get_selected_text_uses_clipboard():
    """get_selected_text copies selection via clipboard and restores prior contents."""
    clipboard = SimpleNamespace(
        view=mock.Mock(side_effect=["previous", "selected text"]),
        copy=mock.Mock(),
    )
    os_module = Os(SimpleNamespace(clipboard=clipboard))
    assert os_module.get_selected_text() == "selected text"
    clipboard.copy.assert_any_call()
    clipboard.copy.assert_any_call("previous")
    assert clipboard.copy.call_count == 2


def test_notify_truncates_long_text_on_linux():
    """Notifications longer than 200 chars are shortened before plyer.notify."""
    import sys

    os_module = Os(SimpleNamespace(verbose=False))
    plyer = mock.Mock()
    with mock.patch("interpreter.core.computer.os.os.platform.system", return_value="Linux"):
        with mock.patch.dict(sys.modules, {"plyer": plyer}):
            os_module.notify("x" * 300)
    plyer.notification.notify.assert_called_once()
    message = plyer.notification.notify.call_args.kwargs["message"]
    assert len(message) <= 203  # 200 chars + "..."
    assert message.endswith("...")


def test_notify_macos_runs_osascript():
    """On macOS, notify() runs osascript with a display notification script."""
    os_module = Os(SimpleNamespace(verbose=False))
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.os.os.subprocess.run"
        ) as run:
            os_module.notify('Say "hello"')
    run.assert_called_once_with(["osascript", "-e", mock.ANY])
    script = run.call_args[0][0][2]
    assert "display notification" in script


def _darwin_notify(text):
    """Run notify() as macOS would, returning the AppleScript that was executed."""
    os_module = Os(SimpleNamespace(verbose=False))
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.os.os.subprocess.run"
        ) as run:
            os_module.notify(text)
    run.assert_called_once()
    return run.call_args[0][0][2]


def _notification_body(script):
    """Pull the quoted message out of `display notification "..." with title "..."`."""
    prefix = 'display notification "'
    start = script.index(prefix) + len(prefix)
    return script[start : script.index('" with title', start)]


def test_notify_strips_characters_that_would_break_the_applescript():
    """Quotes, newlines and shell-ish characters are removed before osascript runs.

    The AppleScript embeds the message inside double quotes, so a literal quote
    in the notification text would terminate the string early and produce a
    syntax error — or worse, splice in extra AppleScript. Newlines, angle
    brackets and ampersands are stripped for the same reason: this string is
    assembled into a command, so untrusted model output reaches a command
    interpreter.
    """
    # Curly quotes are included because the implementation strips them in the
    # same pass: a test that only uses straight quotes cannot tell whether that
    # branch of the chain is present at all.
    body = _notification_body(
        _darwin_notify(
            'He said "hi"\u201cquoted\u201d\n<b>bold</b> & more, with a \'quote\''
        )
    )

    for forbidden in ('"', "'", "\u201c", "\u201d", "\n", "<", ">", "&"):
        assert forbidden not in body, f"{forbidden!r} survived into the AppleScript"

    # Exact output, not just absence. Checking that each dangerous character is
    # missing cannot tell "removed" from "replaced with something else" — a
    # mutant turning .replace("<", "") into .replace("<", "XXXX") still leaves no
    # "<" in the result, so the assertion above passes. Only the full string
    # distinguishes stripping from substitution. The doubled spaces and the
    # mangled "bbold/b" are what removing the delimiters from <b>bold</b>
    # actually produces, and they are the proof nothing was silently rewritten.
    assert body == "He said hiquoted bbold/b  more, with a quote"


def test_notify_titles_the_notification_with_the_application_name():
    """The AppleScript sets a title, so macOS shows which app raised the alert.

    Without a title the banner is attributed to osascript or Script Editor,
    which tells a user nothing about where the notification came from. Asserting
    the exact string also stops the title drifting to a different product name.
    """
    script = _darwin_notify("hello")

    assert 'with title "Open Interpreter"' in script


def test_notify_keeps_the_ordinary_characters_in_the_message():
    """Sanitising strips dangerous characters without emptying the message.

    A blanket replace-everything implementation would satisfy the test above
    while silently dropping all content, so the words are asserted too.
    """
    body = _notification_body(_darwin_notify("Build finished successfully"))

    assert body == "Build finished successfully"


@pytest.mark.parametrize("length, truncated", [(200, False), (201, True)])
def test_notify_truncates_only_past_200_characters(length, truncated):
    """The cap is strictly greater than 200, so 200 characters survive intact.

    `>` and `>=` behave identically at every length except 200 itself, so the
    boundary needs its own case or an off-by-one passes unnoticed.
    """
    import sys

    os_module = Os(SimpleNamespace(verbose=False))
    plyer = mock.Mock()
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Linux"
    ):
        with mock.patch.dict(sys.modules, {"plyer": plyer}):
            os_module.notify("x" * length)

    message = plyer.notification.notify.call_args.kwargs["message"]
    assert message.endswith("...") is truncated
    assert len(message) == (203 if truncated else 200)


def test_notify_is_silent_when_plyer_is_missing():
    """plyer is optional, so its absence must not raise.

    `--safe` mode is documented as working without optional dependencies, so a
    user who has not installed plyer gets no notification rather than a crash
    mid-turn.
    """
    import sys

    os_module = Os(SimpleNamespace(verbose=False))
    # Make the import fail the way a missing package does.
    with mock.patch.dict(sys.modules, {"plyer": None}):
        with mock.patch(
            "interpreter.core.computer.os.os.platform.system", return_value="Linux"
        ):
            assert os_module.notify("hello") is None


def test_notify_swallows_errors_unless_verbose(capsys):
    """A failing notification is non-blocking and silent by default.

    Notifications are cosmetic, so an error there must never interrupt the turn
    that raised it — but verbose mode should still surface it, or a user who
    asks for diagnostics gets none.
    """
    quiet = Os(SimpleNamespace(verbose=False))
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.os.os.subprocess.run", side_effect=OSError("boom")
        ):
            quiet.notify("hello")
    assert capsys.readouterr().out == ""

    loud = Os(SimpleNamespace(verbose=True))
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Darwin"
    ):
        with mock.patch(
            "interpreter.core.computer.os.os.subprocess.run", side_effect=OSError("boom")
        ):
            loud.notify("hello")
    out = capsys.readouterr().out
    assert "Notification error" in out
    assert "boom" in out


def test_notify_swallows_a_failure_from_the_notifier_itself():
    """A notifier that imports but then fails is still swallowed.

    plyer raises things other than ImportError on a headless box — a missing
    notification daemon surfaces as a dbus or AttributeError from
    `plyer.notification.notify`, long after the import succeeded. Covering only
    the missing-package case would let a narrowed `except ImportError` through
    and turn a cosmetic notification into a crash mid-turn.
    """
    import sys

    os_module = Os(SimpleNamespace(verbose=False))
    plyer = mock.Mock()
    plyer.notification.notify.side_effect = AttributeError("no dbus session")
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Linux"
    ):
        with mock.patch.dict(sys.modules, {"plyer": plyer}):
            assert os_module.notify("hello") is None


def test_notify_passes_the_title_to_the_cross_platform_notifier():
    """plyer receives the application name as its title.

    The macOS path embeds the title in the AppleScript and the existing test
    checks that, but the Linux/other path passes it as a keyword argument and
    nothing asserted it — dropping `title=` silently produced untitled banners
    on every platform except macOS.
    """
    import sys

    os_module = Os(SimpleNamespace(verbose=False))
    plyer = mock.Mock()
    with mock.patch(
        "interpreter.core.computer.os.os.platform.system", return_value="Linux"
    ):
        with mock.patch.dict(sys.modules, {"plyer": plyer}):
            os_module.notify("hello")

    kwargs = plyer.notification.notify.call_args.kwargs
    assert kwargs["title"] == "Open Interpreter"
    assert kwargs["message"] == "hello"
