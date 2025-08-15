from importlib.metadata import version, PackageNotFoundError
import requests



def check_for_update():
    # Fetch the latest version from the PyPI API
    response = requests.get(f"https://pypi.org/pypi/open-interpreter/json")
    latest_version = response.json()["info"]["version"]

    # Get the current version using importlib.metadata
    current_version = version("open-interpreter")

    return latest_version > current_version
