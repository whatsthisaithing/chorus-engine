"""Repository for video database operations."""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from chorus_engine.models.conversation import GeneratedVideo

logger = logging.getLogger(__name__)


class VideoRepository:
    """
    Repository for video CRUD operations.
    
    Handles database interactions for generated videos,
    including creation, retrieval, and deletion.
    """
    
    def __init__(self, db: Session):
        """
        Initialize repository.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    def create_video(
        self,
        conversation_id: str,
        file_path: str,
        prompt: str,
        thumbnail_path: Optional[str] = None,
        format: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        workflow_file: Optional[str] = None,
        comfy_prompt_id: Optional[str] = None,
        generation_time_seconds: Optional[float] = None
    ) -> GeneratedVideo:
        """
        Create a new video record.
        
        Args:
            conversation_id: Conversation ID
            file_path: Path to video file (relative)
            prompt: Prompt used for generation
            thumbnail_path: Path to thumbnail image
            format: Video format (webm, mp4, webp, etc.)
            duration_seconds: Video duration
            width: Video width in pixels
            height: Video height in pixels
            negative_prompt: Negative prompt
            workflow_file: ComfyUI workflow file used
            comfy_prompt_id: ComfyUI prompt ID
            generation_time_seconds: Time taken to generate
        
        Returns:
            Created Video object
        """
        video = GeneratedVideo(
            conversation_id=conversation_id,
            file_path=file_path,
            thumbnail_path=thumbnail_path,
            format=format,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            workflow_file=workflow_file,
            comfy_prompt_id=comfy_prompt_id,
            generation_time_seconds=generation_time_seconds
        )
        
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        
        logger.info(f"Created video record: {video.id} for conversation {conversation_id}")
        
        return video
    
    def get_video_by_id(self, video_id: int) -> Optional[GeneratedVideo]:
        """
        Get video by ID.
        
        Args:
            video_id: Video ID
        
        Returns:
            GeneratedVideo object or None
        """
        return self.db.query(GeneratedVideo).filter(GeneratedVideo.id == video_id).first()
    
    def get_by_id(self, video_id: int) -> Optional[GeneratedVideo]:
        """
        Get video by ID (alias for get_video_by_id).
        
        Args:
            video_id: Video ID
        
        Returns:
            GeneratedVideo object or None
        """
        return self.get_video_by_id(video_id)
    
    def list_videos_for_conversation(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[GeneratedVideo]:
        """
        List videos for a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of videos to return
            offset: Number of videos to skip
        
        Returns:
            List of GeneratedVideo objects (newest first)
        """
        return (
            self.db.query(GeneratedVideo)
            .filter(GeneratedVideo.conversation_id == conversation_id)
            .order_by(desc(GeneratedVideo.created_at))
            .limit(limit)
            .offset(offset)
            .all()
        )
    
    def get_by_conversation(self, conversation_id: str) -> List[GeneratedVideo]:
        """
        Get all videos for a conversation (alias for list_videos_for_conversation with no limit).
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            List of GeneratedVideo objects (newest first)
        """
        logger.info(f"[VIDEO REPO] Querying videos for conversation_id: {conversation_id}")
        results = (
            self.db.query(GeneratedVideo)
            .filter(GeneratedVideo.conversation_id == conversation_id)
            .order_by(desc(GeneratedVideo.created_at))
            .all()
        )
        logger.info(f"[VIDEO REPO] Found {len(results)} videos")
        return results
    
    def get_video_count_for_conversation(self, conversation_id: str) -> int:
        """
        Get total video count for a conversation.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Number of videos
        """
        return (
            self.db.query(GeneratedVideo)
            .filter(GeneratedVideo.conversation_id == conversation_id)
            .count()
        )
    
    def delete_video(self, video_id: int) -> bool:
        """
        Delete a video record.
        
        Note: This only deletes the database record.
        File cleanup should be handled by the service layer.
        
        Args:
            video_id: Video ID
        
        Returns:
            True if deleted, False if not found
        """
        video = self.get_video_by_id(video_id)
        
        if not video:
            logger.warning(f"Video {video_id} not found for deletion")
            return False
        
        self.db.delete(video)
        self.db.commit()
        
        logger.info(f"Deleted video record: {video_id}")
        
        return True
    
    def delete(self, video_id: int) -> bool:
        """
        Delete a video record (alias for delete_video).
        
        Args:
            video_id: Video ID
        
        Returns:
            True if deleted, False if not found
        """
        return self.delete_video(video_id)
    
    def delete_all_for_conversation(self, conversation_id: str) -> int:
        """
        Delete all videos for a conversation.
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Number of videos deleted
        """
        count = (
            self.db.query(GeneratedVideo)
            .filter(GeneratedVideo.conversation_id == conversation_id)
            .delete()
        )
        
        self.db.commit()
        
        logger.info(f"Deleted {count} videos for conversation {conversation_id}")
        
        return count
