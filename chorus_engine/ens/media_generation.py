"""Shared ENS media generation wrapper for tool execution paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


@dataclass
class ENSMediaResult:
    """Normalized media artifact returned by ENS media execution."""

    media_type: str
    success: bool
    image_id: Optional[int] = None
    video_id: Optional[int] = None
    file_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    format: Optional[str] = None
    duration_seconds: Optional[float] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None
    scene_message_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "success": self.success,
            "image_id": self.image_id,
            "video_id": self.video_id,
            "file_path": self.file_path,
            "thumbnail_path": self.thumbnail_path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "format": self.format,
            "duration_seconds": self.duration_seconds,
            "generation_time": self.generation_time,
            "error": self.error,
            "scene_message_id": self.scene_message_id,
        }


class ENSMediaGenerator:
    """Small wrapper that normalizes legacy generation response contracts."""

    def __init__(
        self,
        *,
        image_executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        video_executor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> None:
        self._image_executor = image_executor
        self._video_executor = video_executor

    async def execute(self, media_type: str, payload: Dict[str, Any]) -> ENSMediaResult:
        if media_type == "image":
            raw = await self._image_executor(payload)
            return ENSMediaResult(
                media_type="image",
                success=bool(raw.get("success")),
                image_id=raw.get("image_id"),
                file_path=raw.get("file_path"),
                thumbnail_path=raw.get("thumbnail_path"),
                prompt=raw.get("prompt"),
                negative_prompt=raw.get("negative_prompt"),
                generation_time=raw.get("generation_time"),
                error=raw.get("error"),
                scene_message_id=raw.get("scene_message_id"),
            )
        if media_type == "video":
            raw = await self._video_executor(payload)
            return ENSMediaResult(
                media_type="video",
                success=bool(raw.get("success")),
                video_id=raw.get("video_id"),
                file_path=raw.get("file_path"),
                thumbnail_path=raw.get("thumbnail_path"),
                prompt=raw.get("prompt"),
                negative_prompt=raw.get("negative_prompt"),
                format=raw.get("format"),
                duration_seconds=raw.get("duration_seconds"),
                generation_time=raw.get("generation_time"),
                error=raw.get("error"),
                scene_message_id=raw.get("scene_message_id"),
            )
        raise RuntimeError(f"Unsupported media_type: {media_type}")
