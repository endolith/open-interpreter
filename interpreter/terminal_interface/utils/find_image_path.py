import os
import re


def find_image_path(text):
    """Find all image paths in text. Returns a list of existing image paths, or empty list if none found."""
    pattern = r"([A-Za-z]:\\[^:\n]*?\.(png|jpg|jpeg|PNG|JPG|JPEG))|(/[^:\n]*?\.(png|jpg|jpeg|PNG|JPG|JPEG))"
    matches = [match.group() for match in re.finditer(pattern, text) if match.group()]
    matches += [match.replace("\\", "") for match in matches if match]
    existing_paths = [match for match in matches if os.path.exists(match)]
    # Remove duplicates while preserving order
    return list(set(existing_paths))
