"""Splat file I/O for Blender.

This module provides USDZ file format loaders for Gaussian splats.
Assumes worldlabs-api-python export format: scales and densities are stored in
pre-activation form and converted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _vt_array_to_numpy(vt_array, dtype=np.float32) -> np.ndarray:
    """Convert USD Vt array to numpy, handling Vec3f/Color3f/Quatf types.

    USD's Vec3fArray/Color3fArray sometimes don't convert to (N, 3) directly
    with np.array(). This function ensures proper shape.
    """
    arr = np.array(vt_array, dtype=dtype)

    # If already the right shape, return as-is
    if arr.ndim == 2:
        return arr

    # If 1D array of compound types, try to reshape
    if arr.ndim == 1 and len(arr) > 0:
        # Check if elements are iterable (Vec3f, Color3f, etc.)
        try:
            first = arr[0]
            if hasattr(first, "__len__"):
                n_components = len(first)
                return np.array(
                    [[v[i] for i in range(n_components)] for v in arr], dtype=dtype
                )
        except (TypeError, IndexError):
            pass

    return arr


@dataclass
class GaussianData:
    """Gaussian splat data in numpy format (post-activation)."""

    positions: np.ndarray  # (N, 3)
    scales: np.ndarray  # (N, 3) - activated (post-exp)
    rotations: np.ndarray  # (N, 4) quaternions (w, x, y, z)
    opacities: np.ndarray  # (N,) - activated (post-sigmoid)
    colors: np.ndarray  # (N, 3) RGB


SUPPORTED_EXTENSIONS = {".usdz"}


def load_splat(filepath: str) -> GaussianData:
    """Load Gaussian splat data from a file.

    Automatically dispatches to the appropriate loader based on file extension.

    Args:
        filepath: Path to the splat file.

    Returns:
        GaussianData with the loaded splat (all values post-activation).

    Raises:
        ValueError: If the file format is not supported.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".usdz":
        return _load_usdz(filepath)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file format: {ext}. Supported: {supported}")


def _load_usdz(filepath: str) -> GaussianData:
    """Load Gaussian splat data from a USDZ file.

    USDZ format stores pre-activation values (matching NVIDIA 3dgrut):
        - scales: log-scale -> apply exp
        - densities: logit -> apply sigmoid
    """
    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError(
            "`from pxr import Usd` failed. Upgrade your Blender version to 4.0+,\n"
            "or install usd-core in Blender's Python:\n"
            "  /path/to/blender/python -m pip install usd-core"
        ) from e

    stage = Usd.Stage.Open(filepath)
    prim = stage.GetDefaultPrim()

    # Positions (N, 3) - direct
    positions = _vt_array_to_numpy(prim.GetAttribute("positions").Get())
    print(f"[DEBUG] imported splat positions shape: {positions.shape}")

    # Scales (N, 3) - stored as log-scale, apply exp
    scales_raw = _vt_array_to_numpy(prim.GetAttribute("scales").Get())
    scales = np.exp(scales_raw)
    # Rotations (N, 4) - quaternions (w, x, y, z), direct
    rotations = _vt_array_to_numpy(prim.GetAttribute("rotations").Get())
    # Densities (N,) - stored as logit, apply sigmoid
    densities_raw = np.array(prim.GetAttribute("densities").Get(), dtype=np.float32)
    opacities = _sigmoid(densities_raw)
    # Colors (N, 3) - RGB, direct
    colors = _vt_array_to_numpy(prim.GetAttribute("features_albedo").Get())

    return GaussianData(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )
