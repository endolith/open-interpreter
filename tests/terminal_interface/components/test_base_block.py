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
