from rich.console import Console
from rich.live import Live


class BaseBlock:
    """
    a visual "block" on the terminal.
    """

    def __init__(self):
        # emoji=False so command output (e.g. MAC addresses like dc:ef:09:ab:07:da) is not
        # interpreted as Rich emoji shortcodes (e.g. :ab: -> 🆎).
        self.live = Live(
            auto_refresh=False,
            console=Console(emoji=False),
            vertical_overflow="visible",
        )

    def update_from_message(self, message):
        raise NotImplementedError("Subclasses must implement this method")

    def end(self):
        self.refresh(cursor=False)
        self.live.stop()

    def refresh(self, cursor=True):
        raise NotImplementedError("Subclasses must implement this method")
