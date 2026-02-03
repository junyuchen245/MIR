"""L2R image format reader/writer with lightweight compression.

This format stores a small header plus a zlib-compressed payload. Data is
optionally permuted before compression to add light obfuscation.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

MAGIC = b"L2RIMG"
VERSION = 1

_DTYPE_TO_CODE = {
    np.dtype("uint8"): 1,
    np.dtype("int16"): 2,
    np.dtype("int32"): 3,
    np.dtype("float32"): 4,
    np.dtype("float64"): 5,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _default_perm(ndim: int) -> Tuple[int, ...]:
    return tuple(reversed(range(ndim)))


def write_l2r(path: str | Path, data: np.ndarray, spacing: Iterable[float] | None = None,
              perm: Iterable[int] | None = None, compresslevel: int = 3) -> None:
    data = np.asarray(data)
    dtype = data.dtype
    if dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"Unsupported dtype for L2R format: {dtype}")

    ndim = data.ndim
    if perm is None:
        perm = _default_perm(ndim)
    perm = tuple(perm)
    if len(perm) != ndim or sorted(perm) != list(range(ndim)):
        raise ValueError("Invalid permutation for L2R format")

    if spacing is None:
        spacing = (1.0,) * ndim
    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != ndim:
        raise ValueError("Spacing must have same length as data.ndim")

    data_perm = np.transpose(data, perm) if ndim > 1 else data
    raw = data_perm.tobytes(order="C")
    compressed = zlib.compress(raw, level=compresslevel)

    shape = data.shape

    header = struct.pack(
        "<6sBBBB",
        MAGIC,
        VERSION,
        ndim,
        _DTYPE_TO_CODE[dtype],
        len(perm),
    )
    header += struct.pack(f"<{ndim}I", *shape)
    header += struct.pack(f"<{ndim}f", *spacing)
    header += struct.pack(f"<{len(perm)}B", *perm)
    header += struct.pack("<Q", len(compressed))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(compressed)


def read_l2r(path: str | Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    path = Path(path)
    with open(path, "rb") as f:
        magic, version, ndim, dtype_code, perm_len = struct.unpack("<6sBBBB", f.read(10))
        if magic != MAGIC:
            raise ValueError("Invalid L2R image header")
        if version != VERSION:
            raise ValueError(f"Unsupported L2R version: {version}")

        shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
        spacing = struct.unpack(f"<{ndim}f", f.read(4 * ndim))
        perm = struct.unpack(f"<{perm_len}B", f.read(perm_len))
        comp_len = struct.unpack("<Q", f.read(8))[0]
        compressed = f.read(comp_len)

    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported L2R dtype code: {dtype_code}")

    raw = zlib.decompress(compressed)
    perm_shape = tuple(shape[i] for i in perm) if ndim > 1 else shape
    data_perm = np.frombuffer(raw, dtype=dtype).reshape(perm_shape)

    if ndim > 1:
        inv_perm = np.argsort(perm)
        data = np.transpose(data_perm, inv_perm)
    else:
        data = data_perm

    return data, tuple(spacing)
