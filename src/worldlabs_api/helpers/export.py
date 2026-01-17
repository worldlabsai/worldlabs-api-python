"""Helpers for exporting Gaussian splats."""

from __future__ import annotations

import pathlib
from typing import Literal

import torch
from gsplat.exporter import export_splats

from worldlabs_api.gaussian import Gaussian3D

# SH DC coefficient constant: 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479177387814


def save_ply(
    gaussians: Gaussian3D,
    output_path: pathlib.Path,
    format: Literal["ply", "splat", "ply_compressed"] = "ply",
) -> pathlib.Path:
    """Export a Gaussian3D to a PLY file using gsplat.

    Since Gaussian3D stores activated values, this function converts them
    back to the raw format expected by gsplat's exporter:
    - scales: exp'd -> log-scale
    - opacities: sigmoid'd -> pre-sigmoid (logit)
    - colors (RGB): -> SH DC coefficients

    Args:
        gaussians: The Gaussian splat to export.
        output_path: Path to save the PLY file.
        format: Export format ("ply", "splat", or "ply_compressed").

    Returns:
        The output path.
    """

    # Convert activated values back to raw format for gsplat
    log_scales = torch.log(gaussians.scale)
    opacities_logit = torch.logit(gaussians.opacity.squeeze(-1).clamp(1e-6, 1 - 1e-6))
    sh0 = ((gaussians.feature - 0.5) / SH_C0).unsqueeze(1)  # (N, 1, 3)
    shN = torch.zeros(gaussians.mean.shape[0], 0, 3)  # No higher-order SH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_splats(
        means=gaussians.mean,
        scales=log_scales,
        quats=gaussians.quaternion,
        opacities=opacities_logit,
        sh0=sh0,
        shN=shN,
        format=format,
        save_to=str(output_path),
    )

    return output_path
