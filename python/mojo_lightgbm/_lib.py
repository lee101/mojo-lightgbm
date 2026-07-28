from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-lightgbm.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mlgb_quantize": ([I] * 7, None),
    "mlgb_histograms": ([I] * 10, None),
    "mlgb_histogram_reduce": ([I] * 8, None),
    "mlgb_histogram_subtract_inplace": ([I] * 7, None),
    "mlgb_histogram_totals": ([I] * 4, None),
    "mlgb_best_split": ([I] * 7 + [F, F, I, I, F, F, F, F], None),
    "mlgb_partition": ([I] * 8, I),
    "mlgb_regression_gradients": ([I] * 5, None),
    "mlgb_binary_gradients": ([I] * 5, None),
    "mlgb_predict": ([I] * 11 + [F], None),
    "mlgb_predict_add": ([I] * 9, None),
    "mlgb_predict_leaf": ([I] * 11, None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    if not os.path.exists(LIB):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(array, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(array)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be represented as float64")
    if copy:
        return np.array(array, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(array, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(array)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be represented as int64")
    if np.issubdtype(source.dtype, np.floating):
        if not np.all(np.isfinite(source)) or not np.all(source == np.trunc(source)):
            raise ValueError("integer buffers require finite integral values")
    if copy:
        return np.array(array, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(array, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    return int(array.ctypes.data)
