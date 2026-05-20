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
        # Assistant should know what's going on
        yield {
            "type": "console",
            "format": "output",
            "content": "HTML being displayed on the user's machine...",
            "recipient": "assistant",
        }

        # User sees interactive HTML
        yield {"type": "code", "format": "html", "content": code, "recipient": "user"}

        # Vision models get a screenshot; non-vision models get the source (no Moondream/png).
        if self._assistant_sees_rendered_image():
            base64 = html_to_png_base64(code)
            yield {
                "type": "image",
                "format": "base64.png",
                "content": base64,
                "recipient": "assistant",
            }
        else:
            yield {
                "type": "console",
                "format": "output",
                "content": f"HTML source shown to the user:\n\n```html\n{code}\n```",
                "recipient": "assistant",
            }
