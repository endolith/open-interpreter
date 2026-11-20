from ...utils.html_to_png_base64 import html_to_png_base64
from ..base_language import BaseLanguage


class HTML(BaseLanguage):
    file_extension = "html"
    name = "HTML"

    def __init__(self, computer=None):
        super().__init__()
        self.computer = computer

    def _assistant_sees_rendered_image(self):
        if not self.computer:
            return False
        return (
            getattr(self.computer.interpreter.llm, "supports_vision", False) is True
        )

    def run(self, code):
        # User sees interactive HTML
        yield {"type": "code", "format": "html", "content": code, "recipient": "user"}

        # Exactly one assistant-facing result: an image for vision models when
        # the browser can render one, otherwise the source. Providers expect one
        # result per tool call, and a second one shifts the model's reading of
        # which output belongs to which call.
        if self._assistant_sees_rendered_image():
            try:
                base64 = html_to_png_base64(code)
            except Exception:
                # A missing screenshot file (or any render failure) is a soft
                # failure: fall through to the source instead of aborting.
                base64 = None
            if base64:
                # The image is the assistant's one result; the "is displayed"
                # narration would be a second one.
                yield {
                    "type": "image",
                    "format": "base64.png",
                    "content": base64,
                    "recipient": "assistant",
                }
                return
            yield {
                "type": "console",
                "format": "output",
                "content": (
                    "Could not render a preview (the headless browser failed); "
                    "showing the source instead.\n\n```html\n" + code + "\n```"
                ),
                "recipient": "assistant",
            }
            return

        yield {
            "type": "console",
            "format": "output",
            "content": f"HTML source shown to the user:\n\n```html\n{code}\n```",
            "recipient": "assistant",
        }
