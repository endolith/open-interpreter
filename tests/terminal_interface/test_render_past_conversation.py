from unittest import mock

from interpreter.terminal_interface.render_past_conversation import (
    render_past_conversation,
)


def test_render_past_conversation_prints_user_messages(capsys):
    messages = [
        {"role": "user", "type": "message", "content": "Hello"},
        {"role": "assistant", "type": "message", "content": "Hi"},
    ]
    with mock.patch(
        "interpreter.terminal_interface.render_past_conversation.MessageBlock"
    ) as mb:
        mb.return_value = mock.Mock(type="message", message="")
        render_past_conversation(messages)
    assert "> Hello" in capsys.readouterr().out


def test_render_past_conversation_ends_active_block():
    fake_block = mock.Mock(type="message", message="")
    with mock.patch(
        "interpreter.terminal_interface.render_past_conversation.MessageBlock",
        return_value=fake_block,
    ):
        render_past_conversation(
            [{"role": "assistant", "type": "message", "content": "done"}]
        )
    fake_block.end.assert_called()


def test_render_past_conversation_user_message_ends_active_block():
    fake_block = mock.Mock(type="message", message="partial")
    with mock.patch(
        "interpreter.terminal_interface.render_past_conversation.MessageBlock",
        return_value=fake_block,
    ):
        render_past_conversation(
            [
                {"role": "assistant", "type": "message", "content": "partial"},
                {"role": "user", "type": "message", "content": "new question"},
            ]
        )
    fake_block.end.assert_called_once()


def test_render_past_conversation_code_block_accumulates():
    code_block = mock.Mock(type="code", code="", output="", language="", active_line=None)
    with mock.patch(
        "interpreter.terminal_interface.render_past_conversation.CodeBlock",
        return_value=code_block,
    ):
        render_past_conversation(
            [
                {
                    "role": "assistant",
                    "type": "code",
                    "format": "python",
                    "content": "x = 1",
                    "active_line": 1,
                }
            ]
        )
    assert code_block.language == "python"
    assert code_block.code == "x = 1"
    assert code_block.active_line == 1
    code_block.refresh.assert_called()
    code_block.end.assert_called()


def test_render_past_conversation_console_appends_to_code_output():
    code_block = mock.Mock(type="code", code="x=1", output="", language="python", active_line=None)
    with mock.patch(
        "interpreter.terminal_interface.render_past_conversation.CodeBlock",
        return_value=code_block,
    ):
        render_past_conversation(
            [
                {"role": "assistant", "type": "code", "format": "python", "content": "x=1"},
                {
                    "role": "computer",
                    "type": "console",
                    "format": "output",
                    "content": "1",
                },
            ]
        )
    assert code_block.output == "1"
    code_block.end.assert_called()


def test_render_past_conversation_switches_message_to_code_block():
    msg_block = mock.Mock(type="message", message="intro")
    code_block = mock.Mock(type="code", code="", output="", language="", active_line=None)

    with mock.patch(
        "interpreter.terminal_interface.render_past_conversation.MessageBlock",
        return_value=msg_block,
    ):
        with mock.patch(
            "interpreter.terminal_interface.render_past_conversation.CodeBlock",
            return_value=code_block,
        ):
            render_past_conversation(
                [
                    {"role": "assistant", "type": "message", "content": "intro"},
                    {"role": "assistant", "type": "code", "format": "python", "content": "1+1"},
                ]
            )
    msg_block.end.assert_called()
    assert code_block.code == "1+1"
    code_block.end.assert_called()

