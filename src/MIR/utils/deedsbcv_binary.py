"""Utilities for managing prebuilt deedsBCV binaries.

This module locates precompiled deedsBCV executables (bundled with the package, from ``$MIR_DEEDSBCV_DIR``, or
downloaded to a cache from Google Drive or a release URL) without requiring local compilation. Downloaded release
archives are only available for Linux.
"""

import os
import platform
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

DEFAULT_VERSION = "v0.1.0"
DEFAULT_BASE_URL = "bundled"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mir" / "deedsbcv"

LINUX_ASSET = "deedsbcv-linux-x86_64.tar.gz"

EXECUTABLES = ["deedsBCV", "applyBCV", "applyBCVfloat", "linearBCV"]

DEFAULT_GDRIVE_IDS = {
    "deedsBCV": "16xFMWr0UxeAbjWjG8bG6jUDWO9zEzlm4",
    "applyBCV": "1vzwKqpnKOJwC3VVnQg2fIBPZZPzQUTkV",
    "applyBCVfloat": "1gDaU1Ueg7yKnK8RzFFYyCGuZHj5s5uuM",
    "linearBCV": "1bufQLr-sqoFLH6H-aRlfYogVpt2bbULK",
}


def _bundled_dir() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    return package_root / "models" / "deedsBCV" / "bin"


def _platform_asset() -> str:
    system = platform.system().lower()
    if system == "linux":
        return LINUX_ASSET
    raise RuntimeError("deedsBCV binaries are supported only on Linux.")


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zipf:
            zipf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive_path}")


def _ensure_executable(path: Path) -> None:
    if path.exists() and path.is_file():
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def get_deedsbcv_dir(
    version: str = DEFAULT_VERSION,
    base_url: str = DEFAULT_BASE_URL,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> Path:
    """Ensure deedsBCV binaries exist and return the directory path.

    The bundled ``MIR/models/deedsBCV/bin`` directory is returned if it contains any executable. Otherwise the
    binaries are looked up in, or downloaded to, ``cache_dir / version`` (a ``.complete`` marker file records a
    finished download).

    Args:
        version: Release version, used as the cache sub-directory and in release URLs.
        base_url: ``"bundled"`` (no download), ``"gdrive"`` (download each executable with ``gdown``; file IDs can
            be overridden by ``MIR_DEEDSBCV_GDRIVE_<NAME>_ID`` environment variables) or a base URL from which
            ``{base_url}/{version}/<asset>`` is downloaded and extracted.
        cache_dir: Cache root directory.
        force_download: Download again even if the cache marker exists.

    Returns:
        Directory containing the executables.

    Raises:
        RuntimeError: If ``base_url == "bundled"`` and no bundled or cached binaries exist, if a Google Drive file
            ID is missing, or if the platform is not Linux when downloading a release archive.

    """
    bundled = _bundled_dir()
    if bundled.is_dir() and any((bundled / exe).exists() for exe in EXECUTABLES):
        return bundled
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    version_dir = cache_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    marker = version_dir / ".complete"
    if marker.exists() and not force_download:
        return version_dir

    if base_url == "gdrive":
        import gdown

        for exe in EXECUTABLES:
            env_key = f"MIR_DEEDSBCV_GDRIVE_{exe.upper()}_ID"
            file_id = os.environ.get(env_key, DEFAULT_GDRIVE_IDS.get(exe, ""))
            if not file_id:
                raise RuntimeError(f"Missing Google Drive file ID for {exe}.")
            url = f"https://drive.google.com/uc?id={file_id}"
            out_path = version_dir / exe
            gdown.download(url, str(out_path), quiet=False)
            _ensure_executable(out_path)
        marker.write_text("ok")
        return version_dir

    if base_url == "bundled":
        raise RuntimeError("Bundled binaries not found and downloads disabled.")

    asset = _platform_asset()
    archive_path = version_dir / asset
    url = f"{base_url}/{version}/{asset}"
    urlretrieve(url, archive_path)
    _extract_archive(archive_path, version_dir)

    for exe in EXECUTABLES:
        candidate = version_dir / exe
        _ensure_executable(candidate)

    marker.write_text("ok")
    return version_dir


def get_deedsbcv_executable(
    name: str = "deedsBCV",
    version: str = DEFAULT_VERSION,
    base_url: str = DEFAULT_BASE_URL,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> Path:
    """Return the full path to a deedsBCV executable, downloading if needed.

    ``$MIR_DEEDSBCV_DIR`` is checked first when it is set; if the executable
    is not found there, :func:`get_deedsbcv_dir` is used. On Windows the ``.exe`` suffix is added. The returned
    path is not checked for existence in the fallback case.

    Args:
        name: Executable name (``"deedsBCV"``, ``"applyBCV"``, ``"applyBCVfloat"`` or ``"linearBCV"``).
        version: Release version passed to :func:`get_deedsbcv_dir`.
        base_url: Download source passed to :func:`get_deedsbcv_dir`.
        cache_dir: Cache root directory passed to :func:`get_deedsbcv_dir`.
        force_download: Passed to :func:`get_deedsbcv_dir`.

    Returns:
        Path to the executable.

    """
    env_dir = os.environ.get("MIR_DEEDSBCV_DIR", "")
    base = Path(env_dir) if env_dir else None  # unset: do not fall back to the current directory
    if base is not None and base.is_dir():
        exe = base / name
        if platform.system().lower() == "windows":
            exe = exe.with_suffix(".exe")
        if exe.exists():
            return exe

    bin_dir = get_deedsbcv_dir(
        version=version,
        base_url=base_url,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    exe = bin_dir / name
    if platform.system().lower() == "windows":
        exe = exe.with_suffix(".exe")
    return exe
