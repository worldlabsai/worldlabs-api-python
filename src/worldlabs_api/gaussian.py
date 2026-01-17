"""Torch dataclasses for Gaussian splats and cameras."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class Gaussian3D:
    """A minimal Gaussian splat container.

    All values are stored in their "activated" form (not raw SPZ format).

    Attributes:
        mean: XYZ position, shape (N, 3)
        scale: XYZ scale (after exp activation), shape (N, 3)
        quaternion: Rotation as (w, x, y, z) quaternion for gsplat, shape (N, 4)
        opacity: Opacity values (after sigmoid activation), shape (N, 1)
        feature: RGB colors, shape (N, 3)
    """

    mean: torch.Tensor
    scale: torch.Tensor
    quaternion: torch.Tensor
    opacity: torch.Tensor
    feature: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mean.shape[-1] != 3:
            raise ValueError(f"mean must have shape (N, 3), got {self.mean.shape}")
        if self.scale.shape[-1] != 3:
            raise ValueError(f"scale must have shape (N, 3), got {self.scale.shape}")
        if self.quaternion.shape[-1] != 4:
            raise ValueError(
                f"quaternion must have shape (N, 4), got {self.quaternion.shape}"
            )
        if self.opacity.shape[-1] != 1:
            raise ValueError(
                f"opacity must have shape (N, 1), got {self.opacity.shape}"
            )
        if self.feature.shape[-1] != 3:
            raise ValueError(
                f"feature (RGB) must have shape (N, 3), got {self.feature.shape}"
            )

    def to(self, device: torch.device | str) -> "Gaussian3D":
        return Gaussian3D(
            mean=self.mean.to(device),
            scale=self.scale.to(device),
            quaternion=self.quaternion.to(device),
            opacity=self.opacity.to(device),
            feature=self.feature.to(device),
        )


@dataclass(kw_only=True)
class Camera:
    """Basic pinhole camera with a camera-to-world matrix."""

    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: torch.Tensor

    def __post_init__(self) -> None:
        if self.camera_to_world.shape != (4, 4):
            raise ValueError("camera_to_world must be 4x4")

    def world_to_camera(self) -> torch.Tensor:
        return torch.linalg.inv(self.camera_to_world)

    def intrinsics_matrix(self) -> torch.Tensor:
        return torch.tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=self.camera_to_world.dtype,
            device=self.camera_to_world.device,
        )

    def to(self, device: torch.device | str) -> "Camera":
        return Camera(
            height=self.height,
            width=self.width,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            camera_to_world=self.camera_to_world.to(device),
        )


def stack_cameras(cameras: list[Camera]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked (viewmats, Ks) tensors for gsplat."""

    viewmats = torch.stack([camera.world_to_camera() for camera in cameras])
    ks = torch.stack([camera.intrinsics_matrix() for camera in cameras])
    return viewmats, ks
