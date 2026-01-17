"""Helpers for rendering Gaussian splats with gsplat."""

from __future__ import annotations

import math
import pathlib
from typing import Iterable

import gsplat
import mediapy as media
import numpy as np
import torch

from worldlabs_api.gaussian import Camera, Gaussian3D, stack_cameras


def look_at(camera_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a camera-to-world matrix looking from camera_pos to target."""

    forward = target - camera_pos
    forward = forward / torch.linalg.norm(forward)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=camera_pos.dtype)
    right = torch.cross(forward, up, dim=-1)
    right = right / torch.linalg.norm(right)
    true_up = torch.cross(right, forward, dim=-1)

    # OpenCV convention: +Y is up, +Z is forward
    rotation = torch.stack([right, -true_up, forward], dim=1)
    camera_to_world = torch.eye(4, dtype=camera_pos.dtype)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = camera_pos
    return camera_to_world


def make_turntable_cameras(
    num_frames: int,
    radius: float,
    height: int,
    width: int,
    fov_degrees: float = 60.0,
    elevation: float = 0.0,
) -> list[Camera]:
    """Generate a simple turntable camera path around the origin."""

    focal = 0.5 * width / math.tan(math.radians(fov_degrees) / 2.0)
    cameras: list[Camera] = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        position = torch.tensor(
            [
                radius * math.cos(angle),
                elevation,
                radius * math.sin(angle),
            ],
            dtype=torch.float32,
        )
        camera_to_world = look_at(position, torch.zeros(3))
        cameras.append(
            Camera(
                height=height,
                width=width,
                fx=focal,
                fy=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                camera_to_world=camera_to_world,
            )
        )
    return cameras


def render_gaussians(
    gaussians: Gaussian3D,
    cameras: Iterable[Camera],
    background_rgb: torch.Tensor | None = None,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Render gaussians from cameras with gsplat.

    Note: World Labs SPZ files contain RGB colors only (no spherical harmonics).
    """

    cameras_list = list(cameras)
    viewmats, ks = stack_cameras([camera.to(device) for camera in cameras_list])

    gaussians = gaussians.to(device)
    colors = gaussians.feature  # RGB only, shape (N, 3)

    if colors.shape[-1] != 3:
        raise ValueError(f"Expected RGB colors with shape (N, 3), got {colors.shape}")

    with torch.inference_mode():
        features, alphas, _ = gsplat.rasterization(
            means=gaussians.mean,
            quats=gaussians.quaternion,
            scales=gaussians.scale,
            opacities=gaussians.opacity.squeeze(-1),
            colors=colors,
            viewmats=viewmats,
            Ks=ks,
            width=cameras_list[0].width,
            height=cameras_list[0].height,
            sh_degree=None,
            render_mode="RGB",
        )
        rgb = features
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = rgb.clamp(0, 1).detach().cpu().numpy()
        alpha = alphas.clamp(0, 1).detach().cpu().numpy()
        # Composite over white background
        if background_rgb is not None:
            rgb = rgb * alpha + (1 - alpha) * background_rgb
        return rgb


def render_video(
    gaussians: Gaussian3D,
    cameras: Iterable[Camera],
    output_path: pathlib.Path,
    fps: int = 30,
    device: str | torch.device = "cuda",
) -> pathlib.Path:
    """Render a turntable video and save to mp4."""

    frames = render_gaussians(gaussians, cameras, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(output_path, frames, fps=fps)
    return output_path
