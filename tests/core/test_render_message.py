from types import SimpleNamespace
from unittest import mock

from interpreter.core.render_message import render_message


def test_message_without_templates_returned_unchanged():
    """Messages with no {{...}} template blocks are returned verbatim without running code."""
    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=mock.Mock()),
        verbose=False,
        debug=False,
    )
    assert render_message(interpreter, "Plain system message") == "Plain system message"


def test_template_replaced_with_code_output():
    """{{...}} template blocks are executed and replaced with the console output from computer.run."""
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "hi"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    result = render_message(interpreter, 'Prefix {{print("hi")}} suffix')
    assert result == "Prefix hi suffix"


def test_save_skills_disabled_during_template_execution():
    """Template blocks run with save_skills=False so skill side effects are skipped."""
    save_skills_during_run = []

    def fake_run(language, code, display=False):
        save_skills_during_run.append(computer.save_skills)
        yield {"format": "output", "content": "42"}

    computer = SimpleNamespace(save_skills=True, run=fake_run)
    interpreter = SimpleNamespace(computer=computer, verbose=False, debug=False)
    result = render_message(interpreter, "Answer: {{1+1}}")
    assert result == "Answer: 42"
    assert save_skills_during_run == [False]
    assert computer.save_skills is True


def test_save_skills_restored_after_render_without_templates():
    """Rendering a message without templates leaves computer.save_skills unchanged."""
    computer = SimpleNamespace(save_skills=True, run=lambda *a, **k: iter([]))
    interpreter = SimpleNamespace(computer=computer, verbose=False, debug=False)
    render_message(interpreter, "no templates")
    assert computer.save_skills is True


def test_multiline_template_is_matched():
    """A {{...}} block spanning newlines is still treated as one template.

    The split uses re.DOTALL so a template can contain newlines; without the
    flag it would not match and the raw {{...}} text would reach the model.
    """
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "multi"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    result = render_message(interpreter, "start {{\n1 +\n1\n}} end")
    assert result == "start multi end"


def test_only_both_braces_make_a_template():
    """A part must both start with {{ and end with }} to run as code.

    Re-splitting a message can leave a trailing part that ends with }} without
    starting with {{ (text after a template); with `or` that part would be
    executed as a template, so this text must pass through untouched.
    """
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "RAN"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    # The trailing part after the template ends with }} but does not start with
    # {{, so it is not a template and must be returned verbatim (only the real
    # template is replaced by its run output).
    result = render_message(interpreter, "{{1}} tail }}")
    assert result == "RAN tail }}"


def test_template_code_is_stripped_of_braces_and_whitespace():
    """The executed code is the template's inner text with braces and whitespace removed.

    part[2:-2].strip() is what the model's expression becomes; an off-by-one or a
    dropped strip would run different code than the message shows.
    """
    seen = {}

    def fake_run(language, code, display=False):
        seen["code"] = code
        yield {"format": "output", "content": "ok"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    render_message(interpreter, "{{ 1 + 1 }}")
    assert seen["code"] == "1 + 1"


def test_display_flag_follows_verbose():
    """computer.run is called with display=interpreter.verbose.

    The template's execution output is shown only when the interpreter is
    verbose, so the flag has to be forwarded rather than defaulted or dropped.
    """
    seen = {}

    def fake_run(language, code, display=False):
        seen["display"] = display
        yield {"format": "output", "content": "x"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=True,
        debug=False,
    )
    render_message(interpreter, "{{ 1 }}")
    assert seen["display"] is True


def test_only_output_formatted_lines_are_joined():
    """Only lines with format == "output" contribute; other formats are skipped.

    An `or` in the filter would also include non-output lines (e.g. active_line),
    leaking cursor updates into the rendered message.
    """
    def fake_run(language, code, display=False):
        yield {"format": "active_line", "content": "CURSOR"}
        yield {"format": "output", "content": "kept"}
        yield {"format": "console", "content": "not-output"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    assert render_message(interpreter, "{{ 1 }}") == "kept"


def test_ignore_all_above_marker_is_filtered_out():
    """Output lines containing IGNORE_ALL_ABOVE_THIS_LINE are excluded.

    That sentinel separates useful output from echoed input; a renamed marker
    would let the echoed region back into the rendered message.
    """
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "before"}
        yield {
            "format": "output",
            "content": "IGNORE_ALL_ABOVE_THIS_LINE",
        }
        yield {"format": "output", "content": "after"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    result = render_message(interpreter, "{{ 1 }}")
    assert result == "before\nafter"


def test_multiple_output_lines_joined_with_newlines():
    """Several output lines are joined with a single newline each.

    A wrapped separator would be visible in the rendered system message, so the
    join is exactly "\\n".
    """
    def fake_run(language, code, display=False):
        yield {"format": "output", "content": "one"}
        yield {"format": "output", "content": "two"}

    interpreter = SimpleNamespace(
        computer=SimpleNamespace(save_skills=True, run=fake_run),
        verbose=False,
        debug=False,
    )
    assert render_message(interpreter, "{{ 1 }}") == "one\ntwo"

