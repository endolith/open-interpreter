class BaseBlock:
    """
    a visual "block" on the terminal.
    """

    def __init__(self):
        # Imported here, not at module scope: rich.console / rich.live pull in
        # ~40 ms of startup that a session which never renders a block does not
        # need, and these blocks are only constructed when output is shown.
        from rich.console import Console
        from rich.live import Live

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
