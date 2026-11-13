import base64
import os
import random
import string

from html2image import Html2Image

from ....core.utils.lazy_import import lazy_import

html2image = lazy_import("html2image")

from ....terminal_interface.utils.local_storage_path import get_storage_path


def html_to_png_base64(code):
    # Convert the HTML into an image using html2image
    # disable_logging routes the browser subprocess to DEVNULL; without it the
    # browser inherits our stdout/stderr and prints dbus connection errors and
    # its own "bytes written to file" line into the user's console on every
    # render, burying the assistant output.
    hti = html2image.Html2Image(disable_logging=True)

    # Generate a random filename for the temporary image
    temp_filename = "".join(random.choices(string.digits, k=10)) + ".png"
    hti.output_path = get_storage_path()
    hti.screenshot(
        html_str=code,
        save_as=temp_filename,
        size=(960, 540),
    )

    # Get the full path of the temporary image file
    file_location = os.path.join(get_storage_path(), temp_filename)

    # Convert the image to base64
    with open(file_location, "rb") as image_file:
        screenshot_base64 = base64.b64encode(image_file.read()).decode()

    # Delete the temporary image file
    os.remove(file_location)

    return screenshot_base64
