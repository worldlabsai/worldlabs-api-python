"""Helpers for downloading and loading SPZ splats."""

from __future__ import annotations

import pathlib

import httpx
import spz
import torch

from worldlabs_api.gaussian import Gaussian3D


def download_spz(url: str, output_path: pathlib.Path) -> pathlib.Path:
    """Download an SPZ file from a public URL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=120.0) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    return output_path


def load_spz(path: pathlib.Path) -> Gaussian3D:
    """Load an SPZ file into a Gaussian3D.

    The SPZ format stores data in pre-activation form:
    - Scales: log-scale (we apply exp)
    - Alphas: before sigmoid (we apply sigmoid)
    - Rotations: (x, y, z, w) order (we convert to (w, x, y, z) for gsplat)
    - Colors: SH DC coefficients (we convert to RGB via 0.5 + sh_dc * C0)
    """

    cloud = spz.load_spz(str(path))

    # Load raw data from GaussianCloud
    positions = torch.as_tensor(cloud.positions).view(-1, 3)
    log_scales = torch.as_tensor(cloud.scales).view(-1, 3)
    rotations_xyzw = torch.as_tensor(cloud.rotations).view(-1, 4)
    alphas_pre_sigmoid = torch.as_tensor(cloud.alphas)
    sh_dc = torch.as_tensor(cloud.colors).view(-1, 3)

    # Apply activation functions
    scales = torch.exp(log_scales)
    opacities = torch.sigmoid(alphas_pre_sigmoid).unsqueeze(-1)

    # Convert SH DC coefficient to RGB: rgb = 0.5 + sh_dc * C0
    # C0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
    SH_C0 = 0.28209479177387814
    colors = (0.5 + sh_dc * SH_C0).clamp(0, 1)

    # Convert quaternion from (x, y, z, w) to (w, x, y, z) for gsplat
    rotations_wxyz = rotations_xyzw[:, [3, 0, 1, 2]]

    # Convert splat world space -Y up to +Y up (by also negating Z).
    positions = positions @ torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    # Negate y and z in the quaternion to rotate 180 degrees around x:
    rotations_wxyz = rotations_wxyz * torch.tensor(
        [1.0, 1.0, -1.0, -1.0], dtype=rotations_wxyz.dtype, device=rotations_wxyz.device
    )

    return Gaussian3D(
        mean=positions,
        scale=scales,
        quaternion=rotations_wxyz,
        opacity=opacities,
        feature=colors,
    )
