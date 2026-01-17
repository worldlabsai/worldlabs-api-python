"""Pydantic models for the World Labs public API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class Permission(BaseModel):
    """Access control for a world."""

    public: bool = Field(default=False, description="Whether the world is public")


class MediaAssetKind(str, Enum):
    """High-level media asset type."""

    IMAGE = "image"
    VIDEO = "video"


class MediaAsset(BaseModel):
    """A user-uploaded media asset stored in managed storage."""

    media_asset_id: str = Field(..., description="Server-generated media asset ID")
    file_name: str = Field(..., description="File name")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional application-specific metadata"
    )
    extension: str | None = Field(
        default=None,
        description="File extension without dot",
        examples=["mp4", "png", "jpg"],
    )
    kind: MediaAssetKind = Field(
        ..., description="High-level media type", examples=["image", "video"]
    )
    created_at: datetime | None = Field(None, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")


class UploadUrlInfo(BaseModel):
    """Information required to upload raw bytes directly to storage."""

    upload_url: str = Field(..., description="Signed URL for uploading bytes via PUT")
    required_headers: dict[str, str] | None = Field(
        None,
        description="Headers that must be included when uploading",
    )
    upload_method: str = Field(..., description="Upload method")
    curl_example: str | None = Field(None, description="Optional curl example")


class MediaAssetPrepareUploadRequest(BaseModel):
    """Request to prepare a media asset upload."""

    file_name: str = Field(..., description="File name")
    extension: str | None = Field(
        default=None,
        description="File extension without dot",
        examples=["mp4", "png", "jpg"],
    )
    kind: MediaAssetKind = Field(..., description="High-level media type")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional application-specific metadata"
    )


class MediaAssetPrepareUploadResponse(BaseModel):
    """Response from preparing a media asset upload."""

    media_asset: MediaAsset = Field(..., description="The created media asset")
    upload_info: UploadUrlInfo = Field(..., description="Upload URL information")


class MediaAssetReference(BaseModel):
    """Reference to a previously uploaded MediaAsset."""

    source: Literal["media_asset"] = "media_asset"
    media_asset_id: str = Field(
        ...,
        description="ID of a MediaAsset resource previously created and marked READY",
    )


class UriReference(BaseModel):
    """Reference to content via a publicly accessible URL."""

    source: Literal["uri"] = "uri"
    uri: str = Field(..., description="Publicly accessible URL pointing to the media")


class DataBase64Reference(BaseModel):
    """Reference to content via base64-encoded data."""

    source: Literal["data_base64"] = "data_base64"
    data_base64: str = Field(..., description="Base64-encoded content data")
    extension: str | None = Field(
        None, description="File extension without dot (e.g., 'jpg', 'png')"
    )


Content = Annotated[
    MediaAssetReference | UriReference | DataBase64Reference,
    Field(discriminator="source"),
]


class BasePrompt(BaseModel):
    """Base class for prompts with optional text guidance and recaption control."""

    text_prompt: str | None = Field(
        None, description="Optional text guidance (auto-generated if not provided)"
    )
    disable_recaption: bool | None = Field(
        None, description="If True, use text_prompt as-is without recaptioning"
    )

    @model_validator(mode="after")
    def validate_disable_recaption_requires_text(self) -> "BasePrompt":
        if self.disable_recaption and not self.text_prompt:
            raise ValueError("text_prompt is required when disable_recaption is True")
        return self


class WorldTextPrompt(BasePrompt):
    """Text-to-world generation."""

    type: Literal["text"] = "text"

    @model_validator(mode="after")
    def validate_text_prompt_required(self) -> "WorldTextPrompt":
        if not self.text_prompt:
            raise ValueError("text_prompt is required for text-to-world generation")
        return self


class ImagePrompt(BasePrompt):
    """Image-to-world generation."""

    type: Literal["image"] = "image"
    image_prompt: Content = Field(..., description="Image content")
    is_pano: bool | None = Field(
        None, description="Whether the provided image is already a panorama"
    )


class SphericallyLocatedContent(BaseModel):
    """Content with a preferred location on the sphere."""

    azimuth: float | None = Field(None, description="Azimuth angle in degrees")
    content: Content = Field(..., description="The content at this location")


class MultiImagePrompt(BasePrompt):
    """Multi-image-to-world generation."""

    type: Literal["multi-image"] = "multi-image"
    multi_image_prompt: list[SphericallyLocatedContent] = Field(
        ..., description="List of images with optional spherical locations"
    )
    reconstruct_images: bool = Field(
        default=False,
        description=(
            "Whether to use reconstruction mode (allows up to 8 images, otherwise 4)"
        ),
    )

    @field_validator("multi_image_prompt")
    @classmethod
    def validate_not_empty(
        cls, value: list[SphericallyLocatedContent]
    ) -> list[SphericallyLocatedContent]:
        if not value:
            raise ValueError("multi_image_prompt must contain at least one image")
        return value


class VideoPrompt(BasePrompt):
    """Video-to-world generation."""

    type: Literal["video"] = "video"
    video_prompt: Content = Field(..., description="Video content")


WorldPromptTypeUnion = Annotated[
    WorldTextPrompt | ImagePrompt | MultiImagePrompt | VideoPrompt,
    Field(discriminator="type"),
]


MarbleModelType = Literal["Marble 0.1-mini", "Marble 0.1-plus"]


class WorldsGenerateRequest(BaseModel):
    """Request to generate a world.

    Supports text, image, multi-image, video, depth-pano, and inpaint-pano inputs.
    """

    display_name: str | None = Field(None, description="Optional display name")
    tags: list[str] | None = Field(None, description="Optional tags")
    seed: int | None = Field(None, ge=0, description="Random seed for generation")
    world_prompt: WorldPromptTypeUnion = Field(
        ..., description="The prompt specifying how to generate the world"
    )
    model: MarbleModelType | None = Field(
        default=None, description="Model to use for generation"
    )
    permission: Permission | None = Field(
        default=None, description="Access control permissions for the world"
    )


class MeshAssets(BaseModel):
    """Mesh asset URLs."""

    collider_mesh_url: str | None = Field(None, description="Collider mesh URL")


class ImageryAssets(BaseModel):
    """Imagery asset URLs."""

    pano_url: str | None = Field(None, description="Panorama image URL")


class SplatAssets(BaseModel):
    """Gaussian splat asset URLs."""

    spz_urls: dict[str, str] | None = Field(
        None, description="URLs for SPZ Gaussian splat files"
    )


class WorldAssets(BaseModel):
    """Downloadable outputs of world generation."""

    mesh: MeshAssets | None = Field(None, description="Mesh assets")
    imagery: ImageryAssets | None = Field(None, description="Imagery assets")
    splats: SplatAssets | None = Field(None, description="Gaussian splat assets")
    thumbnail_url: str | None = Field(None, description="Thumbnail URL for the world")
    caption: str | None = Field(None, description="AI-generated world description")


class World(BaseModel):
    """A generated world, including asset URLs."""

    id: str = Field(
        ...,
        description="World identifier",
        validation_alias=AliasChoices("id", "world_id"),
        serialization_alias="id",
    )
    display_name: str | None = Field(None, description="Display name")
    tags: list[str] | None = Field(None, description="Tags associated with the world")
    assets: WorldAssets | None = Field(None, description="Generated world assets")
    created_at: datetime | None = Field(None, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    permission: Permission | None = Field(
        None, description="Access control permissions for the world"
    )
    world_prompt: WorldPromptTypeUnion | None = Field(None, description="World prompt")
    world_marble_url: str | None = Field(
        None, description="World Marble URL"
    )
    model: str | None = Field(None, description="Model used for generation")


T = TypeVar("T")


class OperationError(BaseModel):
    """Error information for a failed operation."""

    code: int | None = Field(None, description="Error code")
    message: str | None = Field(None, description="Error message")


class Operation(BaseModel, Generic[T]):
    """A long-running operation representing asynchronous processing."""

    operation_id: str = Field(..., description="Operation identifier")
    created_at: datetime | None = Field(None, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")
    done: bool = Field(..., description="True if the operation is completed")
    error: OperationError | None = Field(
        default=None, description="Error information if the operation failed"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Service-specific metadata, such as progress percentage",
    )
    response: T | None = Field(
        default=None,
        description="Result payload when done=true and no error",
    )


class GetWorldResponse(BaseModel):
    """Response containing a world resource."""

    world: World


class ListWorldsResponse(BaseModel):
    """Response containing a list of API-generated worlds."""

    worlds: list[World] = Field(..., description="List of worlds")
    next_page_token: str | None = Field(
        None, description="Token for fetching the next page of results"
    )


class ListWorldsRequest(BaseModel):
    """Query params for listing worlds with optional filters."""

    status: Literal["SUCCEEDED", "PENDING", "FAILED", "RUNNING"] | None = Field(
        None, description="Filter by world status"
    )
    model: MarbleModelType | None = Field(
        None, description="Filter by model used for generation"
    )
    tags: list[str] | None = Field(
        None, description="Filter by tags (returns worlds with ANY of these tags)"
    )
    is_public: bool | None = Field(
        None, description="Filter by public visibility"
    )
    created_after: datetime | None = Field(
        None, description="Filter worlds created after this timestamp (inclusive)"
    )
    created_before: datetime | None = Field(
        None, description="Filter worlds created before this timestamp (exclusive)"
    )
    sort_by: Literal["created_at", "updated_at"] | None = Field(
        default=None, description="Sort results by created_at or updated_at"
    )
    page_size: int | None = Field(
        default=None, ge=1, le=100, description="Number of results per page"
    )
    page_token: str | None = Field(
        None, description="Cursor token for pagination"
    )
