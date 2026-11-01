from ..utils.html_to_png_base64 import html_to_png_base64
from ..base_language import BaseLanguage


class HTML(BaseLanguage):
    file_extension = "html"
    name = "HTML"
    execution_mode = "display"  # Renders to the user's UI; no code is executed and no state persists.

    def __init__(self, interpreter=None):
        super().__init__()
        self.interpreter = interpreter

    def _assistant_sees_rendered_image(self):
        if not self.interpreter:
            return False
        return getattr(self.interpreter.llm, "supports_vision", False) is True

    def run(self, code):
        # User sees interactive HTML
        yield {"type": "code", "format": "html", "content": code, "recipient": "user"}

        # Exactly one assistant-facing result: an image for vision models when
        # the browser can render one, otherwise the source. Providers expect one
        # result per tool call, and a second one shifts the model's reading of
        # which output belongs to which call.
        if self._assistant_sees_rendered_image():
            base64 = html_to_png_base64(code)
            if base64:
                yield {
                    "type": "console",
                    "format": "output",
                    "content": "HTML is being displayed on the user's machine...",
                    "recipient": "assistant",
                }
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
