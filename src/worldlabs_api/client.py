"""World Labs public API client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from worldlabs_api import models

logger = structlog.get_logger(__name__)


class WorldLabsClient:
    """Synchronous client for the World Labs public API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.worldlabs.ai",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WorldLabsClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "WLT-Api-Key": self._api_key,
        }

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        response = self._client.request(
            method,
            self._url(path),
            headers=self._headers(),
            json=json_data,
            params=params,
        )
        response.raise_for_status()
        return response

    def prepare_media_upload(
        self, request: models.MediaAssetPrepareUploadRequest
    ) -> models.MediaAssetPrepareUploadResponse:
        response = self._request(
            "POST",
            "/marble/v1/media-assets:prepare_upload",
            json_data=request.model_dump(exclude_none=True),
        )
        return models.MediaAssetPrepareUploadResponse.model_validate(response.json())

    def get_media_asset(self, media_asset_id: str) -> models.MediaAsset:
        response = self._request(
            "GET",
            f"/marble/v1/media-assets/{media_asset_id}",
        )
        return models.MediaAsset.model_validate(response.json())

    def generate_world(
        self, request: models.WorldsGenerateRequest
    ) -> models.Operation[models.World]:
        response = self._request(
            "POST",
            "/marble/v1/worlds:generate",
            json_data=request.model_dump(exclude_none=True),
        )
        return models.Operation[models.World].model_validate(response.json())

    def get_world(self, world_id: str) -> models.World:
        response = self._request("GET", f"/marble/v1/worlds/{world_id}")
        payload = response.json()
        if "world" in payload:
            return models.World.model_validate(payload["world"])
        return models.World.model_validate(payload)

    def list_worlds(
        self, request: models.ListWorldsRequest | None = None
    ) -> models.ListWorldsResponse:
        json_data = request.model_dump(exclude_none=True) if request else {}
        response = self._request("POST", "/marble/v1/worlds:list", json_data=json_data)
        return models.ListWorldsResponse.model_validate(response.json())

    def get_operation(self, operation_id: str) -> models.Operation[models.World]:
        response = self._request(
            "GET",
            f"/marble/v1/operations/{operation_id}",
        )
        return models.Operation[models.World].model_validate(response.json())

    def poll_operation(
        self,
        operation_id: str,
        interval_seconds: float = 5.0,
        timeout_seconds: float | None = 600.0,
    ) -> models.Operation[models.World]:
        """Poll an operation until done or timeout."""
        start = time.time()
        while True:
            operation = self.get_operation(operation_id)
            if operation.done:
                return operation
            if timeout_seconds is not None:
                elapsed = time.time() - start
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        "Operation did not complete within the timeout"
                    )
            time.sleep(interval_seconds)


class AsyncWorldLabsClient:
    """Asynchronous client for the World Labs public API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.worldlabs.ai",
        timeout_seconds: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncWorldLabsClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "WLT-Api-Key": self._api_key,
        }

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        response = await self._client.request(
            method,
            self._url(path),
            headers=self._headers(),
            json=json_data,
            params=params,
        )
        response.raise_for_status()
        return response

    async def prepare_media_upload(
        self, request: models.MediaAssetPrepareUploadRequest
    ) -> models.MediaAssetPrepareUploadResponse:
        response = await self._request(
            "POST",
            "/marble/v1/media-assets:prepare_upload",
            json_data=request.model_dump(exclude_none=True),
        )
        return models.MediaAssetPrepareUploadResponse.model_validate(response.json())

    async def get_media_asset(self, media_asset_id: str) -> models.MediaAsset:
        response = await self._request(
            "GET",
            f"/marble/v1/media-assets/{media_asset_id}",
        )
        return models.MediaAsset.model_validate(response.json())

    async def generate_world(
        self, request: models.WorldsGenerateRequest
    ) -> models.Operation[models.World]:
        response = await self._request(
            "POST",
            "/marble/v1/worlds:generate",
            json_data=request.model_dump(exclude_none=True),
        )
        return models.Operation[models.World].model_validate(response.json())

    async def get_world(self, world_id: str) -> models.World:
        response = await self._request("GET", f"/marble/v1/worlds/{world_id}")
        payload = response.json()
        if "world" in payload:
            return models.World.model_validate(payload["world"])
        return models.World.model_validate(payload)

    async def list_worlds(
        self, request: models.ListWorldsRequest | None = None
    ) -> models.ListWorldsResponse:
        json_data = request.model_dump(exclude_none=True) if request else {}
        response = await self._request("POST", "/marble/v1/worlds:list", json_data=json_data)
        return models.ListWorldsResponse.model_validate(response.json())

    async def get_operation(self, operation_id: str) -> models.Operation[models.World]:
        response = await self._request(
            "GET",
            f"/marble/v1/operations/{operation_id}",
        )
        return models.Operation[models.World].model_validate(response.json())

    async def poll_operation(
        self,
        operation_id: str,
        interval_seconds: float = 5.0,
        timeout_seconds: float | None = 600.0,
    ) -> models.Operation[models.World]:
        """Poll an operation until done or timeout."""
        start = time.time()
        while True:
            operation = await self.get_operation(operation_id)
            if operation.done:
                return operation
            if timeout_seconds is not None:
                elapsed = time.time() - start
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        "Operation did not complete within the timeout"
                    )
            await asyncio.sleep(interval_seconds)
