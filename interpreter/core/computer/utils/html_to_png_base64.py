import base64
import os
import random
import string

from ....core.utils.lazy_import import lazy_import

# html2image pulls in requests/urllib3 and the browser helper modules; it is
# only needed when HTML is actually rendered to an image.
html2image = lazy_import("html2image")

from ....terminal_interface.utils.local_storage_path import get_storage_path


def _screenshot(code, save_as, *, quiet):
    """Render `code` to `save_as` in the storage path, silencing the browser."""
    # Without disable_logging the browser's subprocess inherits our stdout and
    # stderr, dumping dbus connection errors and its own "bytes written to
    # file" line into the user's console on every render. Constructing the
    # wrapper also discovers the browser executable, so a machine without
    # Chrome raises here.
    hti = html2image.Html2Image(disable_logging=quiet)
    hti.output_path = get_storage_path()
    hti.screenshot(
        html_str=code,
        save_as=save_as,
        size=(960, 540),
    )


def html_to_png_base64(code):
    """Render HTML to a base64 PNG, or None when the headless browser cannot.

    The browser is an external process that can be absent (no Chrome) or fail
    to produce a file (snap confinement, a bad page), and callers show the source
    instead in that case, so failures are reported by returning None rather
    than raising into the run loop.
    """
    # Generate a random filename for the temporary image
    temp_filename = "".join(random.choices(string.digits, k=10)) + ".png"

    # Get the full path of the temporary image file
    file_location = os.path.join(get_storage_path(), temp_filename)

    try:
        _screenshot(code, temp_filename, quiet=True)
    except Exception:
        return None

    if not os.path.exists(file_location):
        # html2image reports a failed capture only by writing no file, and the
        # quiet instance discards the browser's own explanation (missing shared
        # libraries, a sandbox refusal, a page it could not load). Render once
        # more with logging enabled so Chromium's diagnostics reach the console
        # beside the missing image instead of leaving a silent gap. The retry
        # also recovers a capture that failed transiently, and a still-missing
        # file stays non-fatal for the caller.
        try:
            _screenshot(code, temp_filename, quiet=False)
        except Exception:
            return None
        if not os.path.exists(file_location):
            return None

    # Convert the image to base64
    try:
        with open(file_location, "rb") as image_file:
            screenshot_base64 = base64.b64encode(image_file.read()).decode()
    except OSError:
        return None
    finally:
        # Delete the temporary image file
        try:
            os.remove(file_location)
        except OSError:
            pass

    return screenshot_base64
