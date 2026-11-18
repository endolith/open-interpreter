import pytest

from interpreter.terminal_interface.components.base_block import BaseBlock


def test_base_block_requires_subclass_implementations():
    """BaseBlock leaves update_from_message and refresh for subclasses to implement."""
    block = BaseBlock()

    with pytest.raises(NotImplementedError):
        block.update_from_message({"role": "user", "content": "hi"})

    with pytest.raises(NotImplementedError):
        block.refresh()

    block.live.stop()


def test_base_block_renders_mac_address_literally():
    """BaseBlock's console does not expand emoji shortcodes in command output.

    Rich turns a bare ":ab:" into an emoji, which mangled MAC addresses in
    rendered output; the block must print them verbatim.
    """
    block = BaseBlock()
    try:
        block.live.stop()
        with block.live.console.capture() as capture:
            block.live.console.print("dc:ef:09:ab:07:da")
    finally:
        block.live.stop()
    assert capture.get() == "dc:ef:09:ab:07:da\n"
