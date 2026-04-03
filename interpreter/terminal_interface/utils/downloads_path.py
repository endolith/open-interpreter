import os
import sys

import platformdirs

# Windows Known Folder: FOLDERID_Downloads — respects Properties > Location (e.g. D:\Downloads).
# ctypes SHGetFolderPath(CSIDL_PROFILE)\Downloads and platformdirs.user_downloads_dir() do not.
_WIN_DOWNLOADS_GUID = "{374DE290-123F-4565-9164-39C4925E467B}"
_WIN_SHELL_FOLDER_KEYS = (
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
)


def _windows_downloads_dir() -> str:
    import winreg

    for subkey in _WIN_SHELL_FOLDER_KEYS:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey) as key:
                raw, _ = winreg.QueryValueEx(key, _WIN_DOWNLOADS_GUID)
        except OSError:
            continue
        return os.path.normpath(os.path.expandvars(raw))
    return os.path.join(os.environ["USERPROFILE"], "Downloads")


def get_downloads_path() -> str:
    if sys.platform == "win32":
        downloads = _windows_downloads_dir()
    else:
        downloads = platformdirs.user_downloads_dir()
    os.makedirs(downloads, exist_ok=True)
    return downloads
