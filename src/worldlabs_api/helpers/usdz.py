"""Helpers for saving Gaussian splats as USDZ files.

Format matches NVIDIA 3dgrut: scales and densities are stored in pre-activation
form (log-scale and logit respectively) for renderer compatibility.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import torch
from pxr import Sdf, Usd, UsdGeom, UsdUtils, Vt

from worldlabs_api.gaussian import Gaussian3D


def _numpy_to_vec3f_array(arr: np.ndarray) -> Vt.Vec3fArray:
    """Convert (N, 3) numpy array to Vt.Vec3fArray using buffer protocol."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return Vt.Vec3fArray.FromNumpy(arr)


def _numpy_to_quatf_array(arr: np.ndarray) -> Vt.QuatfArray:
    """Convert (N, 4) numpy array (w, x, y, z) to Vt.QuatfArray using buffer protocol."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return Vt.QuatfArray.FromNumpy(arr)


def _numpy_to_float_array(arr: np.ndarray) -> Vt.FloatArray:
    """Convert (N,) numpy array to Vt.FloatArray using buffer protocol."""
    arr = np.ascontiguousarray(arr.ravel(), dtype=np.float32)
    return Vt.FloatArray.FromNumpy(arr)


def save_usdz(gaussians: Gaussian3D, output_path: pathlib.Path) -> pathlib.Path:
    """Save a Gaussian3D to a compressed USDZ file.

    Stores data in pre-activation form (matching NVIDIA 3dgrut format):
        - scales: log-scale (pre-exp)
        - densities: logit (pre-sigmoid)
        - rotations: normalized quaternions (w, x, y, z)
        - features_albedo: RGB [0, 1]

    Attribute mapping:
        mean -> positions
        log(scale) -> scales
        quaternion -> rotations
        logit(opacity) -> densities
        feature -> features_albedo
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to numpy, applying inverse activations for 3dgrut compatibility
    positions_np = gaussians.mean.detach().cpu().numpy()
    rotations_np = gaussians.quaternion.detach().cpu().numpy()
    features_np = gaussians.feature.detach().cpu().numpy()

    # Convert activated values to pre-activation (matching 3dgrut)
    # scales: exp'd -> log-scale
    scales_np = torch.log(gaussians.scale).detach().cpu().numpy()
    # opacities: sigmoid'd -> logit (pre-sigmoid)
    opacity_clamped = gaussians.opacity.squeeze(-1).clamp(1e-6, 1 - 1e-6)
    densities_np = torch.logit(opacity_clamped).detach().cpu().numpy()

    # USDZ is a package format, so we must create a .usdc first then package it
    with tempfile.TemporaryDirectory() as tmp_dir:
        usdc_path = pathlib.Path(tmp_dir) / "gaussians.usdc"

        stage = Usd.Stage.CreateNew(str(usdc_path))
        stage.SetMetadata("upAxis", UsdGeom.Tokens.y)

        prim = stage.DefinePrim("/Gaussians", "Scope")
        stage.SetDefaultPrim(prim)

        # Positions (N, 3)
        positions_attr = prim.CreateAttribute(
            "positions", Sdf.ValueTypeNames.Point3fArray
        )
        positions_attr.Set(_numpy_to_vec3f_array(positions_np))

        # Scales (N, 3) - stored as log-scale (pre-activation)
        scales_attr = prim.CreateAttribute("scales", Sdf.ValueTypeNames.Vector3fArray)
        scales_attr.Set(_numpy_to_vec3f_array(scales_np))

        # Rotations (N, 4) - stored as (w, x, y, z) quaternions
        rotations_attr = prim.CreateAttribute(
            "rotations", Sdf.ValueTypeNames.QuatfArray
        )
        rotations_attr.Set(_numpy_to_quatf_array(rotations_np))

        # Densities (N,) - stored as logit (pre-activation)
        densities_attr = prim.CreateAttribute(
            "densities", Sdf.ValueTypeNames.FloatArray
        )
        densities_attr.Set(_numpy_to_float_array(densities_np))

        # Features albedo (N, 3) - RGB colors [0, 1]
        features_attr = prim.CreateAttribute(
            "features_albedo", Sdf.ValueTypeNames.Color3fArray
        )
        features_attr.Set(_numpy_to_vec3f_array(features_np))

        stage.GetRootLayer().Save()

        # Package into USDZ
        UsdUtils.CreateNewUsdzPackage(str(usdc_path), str(output_path))

    return output_path
