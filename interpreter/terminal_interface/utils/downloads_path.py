import os
import sys

import platformdirs

# Windows: use SHGetKnownFolderPath(FOLDERID_Downloads) — same as Explorer for relocated
# folders. platformdirs.user_downloads_dir() uses SHGetFolderPath+CSIDL_PROFILE\Downloads;
# PyPI "userpaths" uses profile\Downloads if that path exists — both miss D:\Downloads-style moves.


def _windows_downloads_dir() -> str:
    import ctypes
    import ctypes.wintypes as w

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)

    CLSIDFromString = ole32.CLSIDFromString
    CLSIDFromString.argtypes = (w.LPCOLESTR, ctypes.POINTER(GUID))
    CLSIDFromString.restype = ctypes.HRESULT

    SHGetKnownFolderPath = shell32.SHGetKnownFolderPath
    SHGetKnownFolderPath.argtypes = (
        ctypes.POINTER(GUID),
        w.DWORD,
        w.HANDLE,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    SHGetKnownFolderPath.restype = ctypes.HRESULT

    CoTaskMemFree = ole32.CoTaskMemFree
    CoTaskMemFree.argtypes = (ctypes.c_void_p,)
    CoTaskMemFree.restype = None

    folder_id = GUID()
    hr = CLSIDFromString(
        "{374DE290-123F-4565-9164-39C4925E467B}", ctypes.byref(folder_id)
    )
    if hr != 0:
        raise OSError(hr, "CLSIDFromString(FOLDERID_Downloads)")

    path_out = ctypes.c_wchar_p()
    hr = SHGetKnownFolderPath(
        ctypes.byref(folder_id), 0, None, ctypes.byref(path_out)
    )
    if hr != 0:
        raise OSError(hr, "SHGetKnownFolderPath(FOLDERID_Downloads)")

    try:
        return os.path.normpath(path_out.value)
    finally:
        CoTaskMemFree(path_out)


def get_downloads_path() -> str:
    if sys.platform == "win32":
        downloads = _windows_downloads_dir()
    else:
        downloads = platformdirs.user_downloads_dir()
    os.makedirs(downloads, exist_ok=True)
    return downloads
