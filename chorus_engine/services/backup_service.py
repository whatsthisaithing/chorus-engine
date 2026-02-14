"""Character backup service for creating complete character snapshots."""

import logging
import json
import shutil
import zipfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

from chorus_engine.models.conversation import (
    Conversation, Thread, Message, Memory,
    GeneratedImage, GeneratedVideo, AudioMessage, VoiceSample,
    ConversationSummary, MomentPin, ImageAttachment
)
from chorus_engine.models.continuity import (
    ContinuityRelationshipState,
    ContinuityArc,
    ContinuityBootstrapCache,
    ContinuityPreference,
)
from chorus_engine.models.document import Document, DocumentChunk, DocumentAccessLog
from chorus_engine.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class BackupError(Exception):
    """Raised when backup operation fails."""
    pass


class CharacterBackupService:
    """Service for creating character backups."""
    
    # Current backup format version - increment when format changes
    BACKUP_FORMAT_VERSION = 2
    
    def __init__(
        self,
        db: Session,
        backup_dir: Path = Path("data/backups"),
        characters_dir: Path = Path("characters"),
        images_dir: Path = Path("data/character_images"),
        media_images_dir: Path = Path("data/images"),
        media_videos_dir: Path = Path("data/videos"),
        media_audio_dir: Path = Path("data/audio"),
        voice_samples_dir: Path = Path("data/voice_samples"),
        workflows_dir: Path = Path("workflows"),
        documents_dir: Path = Path("data/documents"),
    ):
        """
        Initialize backup service.
        
        Args:
            db: SQLAlchemy database session
            backup_dir: Directory to store backup files
            characters_dir: Character YAML files directory
            images_dir: Character profile images directory
            media_images_dir: Generated images directory
            media_videos_dir: Generated videos directory
            media_audio_dir: Message audio directory
            voice_samples_dir: Voice samples directory
            workflows_dir: ComfyUI workflows directory
        """
        self.db = db
        self.backup_dir = backup_dir
        self.characters_dir = characters_dir
        self.images_dir = images_dir
        self.media_images_dir = media_images_dir
        self.media_videos_dir = media_videos_dir
        self.media_audio_dir = media_audio_dir
        self.voice_samples_dir = voice_samples_dir
        self.workflows_dir = workflows_dir
        self.documents_dir = documents_dir
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CharacterBackupService initialized (backup_dir: {backup_dir})")
    
    def backup_character(
        self,
        character_id: str,
        include_workflows: bool = True,
        notes: Optional[str] = None
    ) -> Path:
        """
        Create complete backup of a character.
        
        Args:
            character_id: ID of character to backup
            include_workflows: Whether to include workflow files
            notes: Optional notes to include in manifest
        
        Returns:
            Path to created .zip file
        
        Raises:
            BackupError: If backup creation fails
        """
        logger.info(f"Starting backup for character: {character_id}")
        
        try:
            # Generate timestamp for backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create character-specific backup directory
            char_backup_dir = self.backup_dir / character_id
            char_backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary directory for building backup
            temp_dir = char_backup_dir / f"temp_{timestamp}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Step 1: Export SQL data
                logger.info("Exporting SQL database records...")
                conversations = self._export_conversations(character_id)
                memories = self._export_memories(character_id)
                workflow_records = self._export_workflow_records(character_id)
                media_records = self._export_media_records(character_id)
                moment_pins = self._export_moment_pins(character_id)
                continuity_data = self._export_continuity_data(character_id)
                image_attachments = self._export_image_attachments(character_id)
                document_data = self._export_document_data(character_id, conversations)
                
                # Step 2: Get character configuration
                logger.info("Loading character configuration...")
                character_config = self._get_character_config(character_id)

                # Step 3: Collect media files
                logger.info("Collecting media files...")
                media_files = self._collect_media_files(
                    character_id,
                    conversations,
                    media_records,
                    image_attachments,
                    document_data,
                    character_config
                )
                
                # Step 4: Collect workflow files
                workflows = {}
                if include_workflows:
                    logger.info("Collecting workflow files...")
                    workflows = self._collect_workflows(character_id)
                
                # Step 5: Generate manifest
                logger.info("Generating backup manifest...")
                manifest = self._generate_manifest(
                    character_id=character_id,
                    conversations=conversations,
                    memories=memories,
                    workflow_records=workflow_records,
                    media_records=media_records,
                    moment_pins=moment_pins,
                    continuity_data=continuity_data,
                    image_attachments=image_attachments,
                    document_data=document_data,
                    media_files=media_files,
                    workflows=workflows,
                    character_config=character_config,
                    notes=notes
                )
                
                # Step 6: Create archive
                logger.info("Creating backup archive...")
                backup_path = self._create_archive(
                    character_id=character_id,
                    timestamp=timestamp,
                    temp_dir=temp_dir,
                    manifest=manifest,
                    conversations=conversations,
                    memories=memories,
                    workflow_records=workflow_records,
                    media_records=media_records,
                    moment_pins=moment_pins,
                    continuity_data=continuity_data,
                    image_attachments=image_attachments,
                    document_data=document_data,
                    media_files=media_files,
                    character_config=character_config,
                    workflows=workflows
                )
                
                logger.info(f"Backup created successfully: {backup_path}")
                logger.info(f"Backup size: {backup_path.stat().st_size / (1024*1024):.2f} MB")
                
                return backup_path
                
            finally:
                # Cleanup temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        
        except Exception as e:
            logger.error(f"Backup failed for character {character_id}: {e}", exc_info=True)
            raise BackupError(f"Failed to backup character {character_id}: {e}") from e
    
    def _export_conversations(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Export all conversations with nested threads and messages.
        
        Args:
            character_id: Character ID
        
        Returns:
            List of conversation dicts with full nesting
        """
        conversations = self.db.query(Conversation).filter(
            Conversation.character_id == character_id
        ).all()
        
        result = []
        for conv in conversations:
            conv_dict = {
                'id': conv.id,
                'character_id': conv.character_id,
                'title': conv.title,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat(),
                'is_private': conv.is_private,
                'tts_enabled': conv.tts_enabled,
                'last_extracted_message_count': conv.last_extracted_message_count,
                'image_confirmation_disabled': conv.image_confirmation_disabled,
                'video_confirmation_disabled': getattr(conv, 'video_confirmation_disabled', 'false'),
                'allow_image_offers': getattr(conv, 'allow_image_offers', 'true'),
                'allow_video_offers': getattr(conv, 'allow_video_offers', 'true'),
                'last_image_offer_at': conv.last_image_offer_at.isoformat() if getattr(conv, 'last_image_offer_at', None) else None,
                'last_video_offer_at': conv.last_video_offer_at.isoformat() if getattr(conv, 'last_video_offer_at', None) else None,
                'last_image_offer_message_count': getattr(conv, 'last_image_offer_message_count', None),
                'last_video_offer_message_count': getattr(conv, 'last_video_offer_message_count', None),
                'image_offer_count': getattr(conv, 'image_offer_count', 0),
                'video_offer_count': getattr(conv, 'video_offer_count', 0),
                'continuity_mode': getattr(conv, 'continuity_mode', 'ask'),
                'continuity_choice_remembered': getattr(conv, 'continuity_choice_remembered', 'false'),
                'primary_user': getattr(conv, 'primary_user', None),
                'title_auto_generated': conv.title_auto_generated,
                'last_analyzed_at': conv.last_analyzed_at.isoformat() if conv.last_analyzed_at else None,
                'last_summary_analyzed_at': conv.last_summary_analyzed_at.isoformat() if getattr(conv, 'last_summary_analyzed_at', None) else None,
                'last_memories_analyzed_at': conv.last_memories_analyzed_at.isoformat() if getattr(conv, 'last_memories_analyzed_at', None) else None,
                'source': getattr(conv, 'source', 'web'),  # Handle old schema
                'threads': []
            }
            
            # Load threads
            threads = self.db.query(Thread).filter(
                Thread.conversation_id == conv.id
            ).all()
            
            for thread in threads:
                thread_dict = {
                    'id': thread.id,
                    'conversation_id': thread.conversation_id,
                    'title': thread.title,
                    'created_at': thread.created_at.isoformat(),
                    'updated_at': thread.updated_at.isoformat(),
                    'messages': []
                }
                
                # Load messages
                messages = self.db.query(Message).filter(
                    Message.thread_id == thread.id
                ).order_by(Message.created_at).all()
                
                for msg in messages:
                    msg_dict = {
                        'id': msg.id,
                        'thread_id': msg.thread_id,
                        'role': msg.role.value,
                        'content': msg.content,
                        'meta_data': msg.meta_data,
                        'created_at': msg.created_at.isoformat(),
                        'is_private': msg.is_private,
                        'emotional_weight': getattr(msg, 'emotional_weight', None),
                        'summary': getattr(msg, 'summary', None),
                        'preserve_full_text': getattr(msg, 'preserve_full_text', 'true'),
                        'deleted_at': msg.deleted_at.isoformat() if getattr(msg, 'deleted_at', None) else None,
                    }
                    thread_dict['messages'].append(msg_dict)
                
                conv_dict['threads'].append(thread_dict)
            
            # Load conversation summary if exists
            summary = self.db.query(ConversationSummary).filter(
                ConversationSummary.conversation_id == conv.id
            ).first()
            
            if summary:
                conv_dict['summary'] = {
                    'id': summary.id,
                    'thread_id': summary.thread_id,
                    'summary': summary.summary,  # Actual summary text
                    'summary_type': summary.summary_type,
                    'message_range_start': summary.message_range_start,
                    'message_range_end': summary.message_range_end,
                    'message_count': summary.message_count,
                    'key_topics': summary.key_topics,
                    'participants': summary.participants,
                    'emotional_arc': summary.emotional_arc,
                    'tone': summary.tone,
                    'open_questions': getattr(summary, 'open_questions', None),
                    'manual': getattr(summary, 'manual', 'false'),
                    'created_at': summary.created_at.isoformat()
                }
            
            result.append(conv_dict)
        
        logger.info(f"Exported {len(result)} conversations")
        return result
    
    def _export_memories(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Export all memories for a character.
        
        Args:
            character_id: Character ID
        
        Returns:
            List of memory dicts
        """
        memories = self.db.query(Memory).filter(
            Memory.character_id == character_id
        ).all()
        
        result = []
        for mem in memories:
            mem_dict = {
                'id': mem.id,
                'character_id': mem.character_id,
                'conversation_id': mem.conversation_id,
                'thread_id': mem.thread_id,
                'memory_type': mem.memory_type.value,
                'content': mem.content,
                'vector_id': mem.vector_id,
                'embedding_model': mem.embedding_model,
                'priority': mem.priority,
                'tags': mem.tags,
                'confidence': mem.confidence,
                'category': mem.category,
                'status': mem.status,
                'source_messages': mem.source_messages,  # JSON array of message IDs
                'durability': getattr(mem, 'durability', 'situational'),
                'pattern_eligible': int(getattr(mem, 'pattern_eligible', 0)),
                'emotional_weight': mem.emotional_weight,
                'participants': mem.participants,
                'key_moments': mem.key_moments,
                'summary': getattr(mem, 'summary', None),  # Phase 8 field
                'source': getattr(mem, 'source', 'web'),
                'meta_data': mem.meta_data,
                'created_at': mem.created_at.isoformat()
            }
            result.append(mem_dict)
        
        logger.info(f"Exported {len(result)} memories")
        return result
    
    def _export_workflow_records(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Export all Workflow database records for a character.
        
        Args:
            character_id: Character ID
        
        Returns:
            List of workflow record dicts
        """
        from chorus_engine.models.workflow import Workflow
        
        # Get character name from character_id (assumes character_id == character_name for now)
        # TODO: Might need to look up actual character name from character.yaml
        workflows = self.db.query(Workflow).filter(
            Workflow.character_name == character_id
        ).all()
        
        result = []
        for wf in workflows:
            wf_dict = {
                'id': wf.id,
                'character_name': wf.character_name,
                'workflow_name': wf.workflow_name,
                'workflow_file_path': wf.workflow_file_path,
                'workflow_type': wf.workflow_type,
                'is_default': wf.is_default,
                'trigger_word': wf.trigger_word,
                'default_style': wf.default_style,
                'negative_prompt': wf.negative_prompt,
                'self_description': wf.self_description,
                'created_at': wf.created_at.isoformat(),
                'updated_at': wf.updated_at.isoformat()
            }
            result.append(wf_dict)
        
        logger.info(f"Exported {len(result)} workflow records")
        return result
    
    def _export_media_records(self, character_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export generated image, video, and audio records.
        
        Args:
            character_id: Character ID
        
        Returns:
            Dict with keys: images, videos, audio_messages, voice_samples
        """
        result = {
            'images': [],
            'videos': [],
            'audio_messages': [],
            'voice_samples': []
        }
        
        # Export generated images
        images = self.db.query(GeneratedImage).filter(
            GeneratedImage.character_id == character_id
        ).all()
        
        for img in images:
            result['images'].append({
                'id': img.id,
                'conversation_id': img.conversation_id,
                'thread_id': img.thread_id,
                'message_id': img.message_id,
                'character_id': img.character_id,
                'prompt': img.prompt,
                'negative_prompt': img.negative_prompt,
                'workflow_file': img.workflow_file,
                'file_path': img.file_path,
                'thumbnail_path': img.thumbnail_path,
                'width': img.width,
                'height': img.height,
                'seed': img.seed,
                'notes': img.notes,
                'created_at': img.created_at.isoformat(),
                'generation_time': img.generation_time
            })
        
        # Export generated videos (need to join through Conversation to filter by character)
        videos = self.db.query(GeneratedVideo).join(
            Conversation, GeneratedVideo.conversation_id == Conversation.id
        ).filter(
            Conversation.character_id == character_id
        ).all()
        
        for vid in videos:
            result['videos'].append({
                'id': vid.id,
                'conversation_id': vid.conversation_id,
                'file_path': vid.file_path,
                'thumbnail_path': vid.thumbnail_path,
                'format': vid.format,
                'duration_seconds': vid.duration_seconds,
                'width': vid.width,
                'height': vid.height,
                'prompt': vid.prompt,
                'negative_prompt': vid.negative_prompt,
                'workflow_file': vid.workflow_file,
                'comfy_prompt_id': vid.comfy_prompt_id,
                'generation_time_seconds': vid.generation_time_seconds,
                'created_at': vid.created_at.isoformat()
            })
        
        # Export audio messages (need to join with messages to get character)
        # AudioMessage doesn't have character_id directly, so we filter via message_id
        from chorus_engine.models.conversation import Message
        
        audio_messages = self.db.query(AudioMessage).join(
            Message, AudioMessage.message_id == Message.id
        ).join(
            Thread, Message.thread_id == Thread.id
        ).join(
            Conversation, Thread.conversation_id == Conversation.id
        ).filter(
            Conversation.character_id == character_id
        ).all()
        
        for audio in audio_messages:
            result['audio_messages'].append({
                'id': audio.id,
                'message_id': audio.message_id,
                'audio_filename': audio.audio_filename,
                'workflow_name': audio.workflow_name,
                'generation_duration': audio.generation_duration,
                'text_preprocessed': audio.text_preprocessed,
                'voice_sample_id': audio.voice_sample_id,
                'created_at': audio.created_at.isoformat()
            })
        
        # Export voice samples
        voice_samples = self.db.query(VoiceSample).filter(
            VoiceSample.character_id == character_id
        ).all()
        
        for sample in voice_samples:
            result['voice_samples'].append({
                'id': sample.id,
                'character_id': sample.character_id,
                'filename': sample.filename,
                'transcript': sample.transcript,
                'is_default': sample.is_default,
                'uploaded_at': sample.uploaded_at.isoformat()
            })
        
        logger.info(f"Exported media records: {len(result['images'])} images, "
                   f"{len(result['videos'])} videos, {len(result['audio_messages'])} audio, "
                   f"{len(result['voice_samples'])} voice samples")
        
        return result
    
    def _export_moment_pins(self, character_id: str) -> List[Dict[str, Any]]:
        pins = self.db.query(MomentPin).filter(MomentPin.character_id == character_id).all()
        result: List[Dict[str, Any]] = []
        for pin in pins:
            result.append({
                "id": pin.id,
                "user_id": pin.user_id,
                "character_id": pin.character_id,
                "conversation_id": pin.conversation_id,
                "created_at": pin.created_at.isoformat() if pin.created_at else None,
                "selected_message_ids": pin.selected_message_ids,
                "transcript_snapshot": pin.transcript_snapshot,
                "what_happened": pin.what_happened,
                "why_model": pin.why_model,
                "why_user": pin.why_user,
                "quote_snippet": pin.quote_snippet,
                "tags": pin.tags,
                "reinforcement_score": pin.reinforcement_score,
                "turns_since_reinforcement": pin.turns_since_reinforcement,
                "archived": pin.archived,
                "telemetry_flags": pin.telemetry_flags,
                "vector_id": pin.vector_id,
            })
        logger.info(f"Exported {len(result)} moment pins")
        return result

    def _export_continuity_data(self, character_id: str) -> Dict[str, Any]:
        state = (
            self.db.query(ContinuityRelationshipState)
            .filter(ContinuityRelationshipState.character_id == character_id)
            .first()
        )
        arcs = (
            self.db.query(ContinuityArc)
            .filter(ContinuityArc.character_id == character_id)
            .all()
        )
        cache = (
            self.db.query(ContinuityBootstrapCache)
            .filter(ContinuityBootstrapCache.character_id == character_id)
            .first()
        )
        pref = (
            self.db.query(ContinuityPreference)
            .filter(ContinuityPreference.character_id == character_id)
            .first()
        )
        return {
            "relationship_state": {
                "id": state.id,
                "character_id": state.character_id,
                "familiarity_level": state.familiarity_level,
                "tone_baseline": state.tone_baseline,
                "interaction_contract": state.interaction_contract,
                "boundaries": state.boundaries,
                "assistant_role_frame": state.assistant_role_frame,
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            } if state else None,
            "arcs": [{
                "id": arc.id,
                "character_id": arc.character_id,
                "title": arc.title,
                "kind": arc.kind,
                "summary": arc.summary,
                "status": arc.status,
                "confidence": arc.confidence,
                "stickiness": arc.stickiness,
                "last_touched_conversation_id": arc.last_touched_conversation_id,
                "last_touched_conversation_at": arc.last_touched_conversation_at.isoformat() if arc.last_touched_conversation_at else None,
                "frequency_count": arc.frequency_count,
                "created_at": arc.created_at.isoformat() if arc.created_at else None,
                "updated_at": arc.updated_at.isoformat() if arc.updated_at else None,
            } for arc in arcs],
            "bootstrap_cache": {
                "id": cache.id,
                "character_id": cache.character_id,
                "bootstrap_packet_internal": cache.bootstrap_packet_internal,
                "bootstrap_packet_user_preview": cache.bootstrap_packet_user_preview,
                "bootstrap_generated_at": cache.bootstrap_generated_at.isoformat() if cache.bootstrap_generated_at else None,
                "bootstrap_inputs_fingerprint": cache.bootstrap_inputs_fingerprint,
                "updated_at": cache.updated_at.isoformat() if cache.updated_at else None,
            } if cache else None,
            "preference": {
                "character_id": pref.character_id,
                "default_mode": pref.default_mode,
                "skip_preview": pref.skip_preview,
                "updated_at": pref.updated_at.isoformat() if pref.updated_at else None,
            } if pref else None,
        }

    def _export_image_attachments(self, character_id: str) -> List[Dict[str, Any]]:
        attachments = self.db.query(ImageAttachment).filter(ImageAttachment.character_id == character_id).all()
        result: List[Dict[str, Any]] = []
        for attachment in attachments:
            result.append({
                "id": attachment.id,
                "message_id": attachment.message_id,
                "conversation_id": attachment.conversation_id,
                "character_id": attachment.character_id,
                "original_path": attachment.original_path,
                "processed_path": attachment.processed_path,
                "original_filename": attachment.original_filename,
                "file_size": attachment.file_size,
                "mime_type": attachment.mime_type,
                "width": attachment.width,
                "height": attachment.height,
                "uploaded_at": attachment.uploaded_at.isoformat() if attachment.uploaded_at else None,
                "vision_processed": attachment.vision_processed,
                "vision_skipped": attachment.vision_skipped,
                "vision_skip_reason": attachment.vision_skip_reason,
                "vision_model": attachment.vision_model,
                "vision_backend": attachment.vision_backend,
                "vision_processed_at": attachment.vision_processed_at.isoformat() if attachment.vision_processed_at else None,
                "vision_processing_time_ms": attachment.vision_processing_time_ms,
                "vision_observation": attachment.vision_observation,
                "vision_confidence": attachment.vision_confidence,
                "vision_tags": attachment.vision_tags,
                "description": attachment.description,
                "source": attachment.source,
            })
        logger.info(f"Exported {len(result)} image attachments")
        return result

    def _export_document_data(self, character_id: str, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        conversation_ids = {conv["id"] for conv in conversations}
        docs = self.db.query(Document).filter(
            (Document.character_id == character_id) |
            (Document.conversation_id.in_(list(conversation_ids)))
        ).all()
        doc_ids = [doc.id for doc in docs]
        chunks = self.db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(doc_ids)).all() if doc_ids else []
        access_logs = self.db.query(DocumentAccessLog).filter(
            (DocumentAccessLog.document_id.in_(doc_ids)) |
            (DocumentAccessLog.conversation_id.in_(list(conversation_ids)))
        ).all() if (doc_ids or conversation_ids) else []
        return {
            "documents": [{
                "id": doc.id,
                "filename": doc.filename,
                "storage_key": doc.storage_key,
                "file_type": doc.file_type,
                "file_size_bytes": doc.file_size_bytes,
                "page_count": doc.page_count,
                "title": doc.title,
                "description": doc.description,
                "author": doc.author,
                "character_id": doc.character_id,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "last_accessed": doc.last_accessed.isoformat() if doc.last_accessed else None,
                "document_scope": doc.document_scope,
                "conversation_id": doc.conversation_id,
                "processing_status": doc.processing_status,
                "processing_error": doc.processing_error,
                "chunk_count": doc.chunk_count,
                "vector_collection_id": doc.vector_collection_id,
                "metadata_json": doc.metadata_json,
            } for doc in docs],
            "chunks": [{
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "content_length": chunk.content_length,
                "page_numbers": chunk.page_numbers,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_method": chunk.chunk_method,
                "overlap_tokens": chunk.overlap_tokens,
                "embedding_model": chunk.embedding_model,
                "embedding_created_at": chunk.embedding_created_at.isoformat() if chunk.embedding_created_at else None,
                "metadata_json": chunk.metadata_json,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            } for chunk in chunks],
            "access_logs": [{
                "id": log.id,
                "document_id": log.document_id,
                "conversation_id": log.conversation_id,
                "message_id": log.message_id,
                "access_type": log.access_type,
                "chunks_retrieved": log.chunks_retrieved,
                "chunk_ids": log.chunk_ids,
                "query": log.query,
                "relevance_score": log.relevance_score,
                "accessed_at": log.accessed_at.isoformat() if log.accessed_at else None,
            } for log in access_logs],
        }
    
    def _collect_media_files(
        self,
        character_id: str,
        conversations: List[Dict],
        media_records: Dict[str, List[Dict]],
        image_attachments: List[Dict[str, Any]],
        document_data: Dict[str, Any],
        character_config: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Identify and collect all media files for backup.
        
        Args:
            character_id: Character ID
            conversations: Exported conversations
            media_records: Exported media records
            image_attachments: Exported image attachment rows
            document_data: Exported document rows/chunks/logs
        
        Returns:
            Dict mapping archive paths to source file paths
        """
        media_files = {}
        
        # Get conversation IDs for filtering
        conversation_ids = [conv['id'] for conv in conversations]
        
        # Collect generated images
        for img in media_records['images']:
            if img['conversation_id'] in conversation_ids:
                file_path = Path(img['file_path'])
                if file_path.exists():
                    archive_path = f"media/images/{img['conversation_id']}/{file_path.name}"
                    media_files[archive_path] = file_path
                else:
                    logger.warning(f"Image file not found: {file_path}")
                
                # Include thumbnail if exists
                if img['thumbnail_path']:
                    thumb_path = Path(img['thumbnail_path'])
                    if thumb_path.exists():
                        archive_path = f"media/images/{img['conversation_id']}/{thumb_path.name}"
                        media_files[archive_path] = thumb_path
        
        # Collect generated videos
        for vid in media_records['videos']:
            if vid['conversation_id'] in conversation_ids:
                file_path = Path(vid['file_path'])
                if file_path.exists():
                    archive_path = f"media/videos/{vid['conversation_id']}/{file_path.name}"
                    media_files[archive_path] = file_path
                else:
                    logger.warning(f"Video file not found: {file_path}")
                
                # Include thumbnail if exists
                if vid['thumbnail_path']:
                    thumb_path = Path(vid['thumbnail_path'])
                    if thumb_path.exists():
                        archive_path = f"media/videos/{vid['conversation_id']}/{thumb_path.name}"
                        media_files[archive_path] = thumb_path
        
        # Collect audio files
        for audio in media_records['audio_messages']:
            # AudioMessage stores just the filename in audio_filename field
            # Files are stored in data/audio/
            audio_filename = audio['audio_filename']
            file_path = self.media_audio_dir / audio_filename
            if file_path.exists():
                archive_path = f"media/audio/{audio_filename}"
                media_files[archive_path] = file_path
            else:
                logger.warning(f"Audio file not found: {file_path}")
        
        # Collect voice samples
        for sample in media_records['voice_samples']:
            # VoiceSample stores just the filename in filename field
            # Files are stored in data/voice_samples/{character_id}/
            filename = sample['filename']
            file_path = self.voice_samples_dir / character_id / filename
            if file_path.exists():
                archive_path = f"media/voice_samples/{character_id}/{filename}"
                media_files[archive_path] = file_path
            else:
                logger.warning(f"Voice sample not found: {file_path}")

        # Collect image attachment files
        for attachment in image_attachments:
            original_path = attachment.get("original_path")
            if original_path:
                src = Path(original_path)
                if src.exists():
                    media_files[f"media/attachments/original/{src.name}"] = src
            processed_path = attachment.get("processed_path")
            if processed_path:
                src = Path(processed_path)
                if src.exists():
                    media_files[f"media/attachments/processed/{src.name}"] = src

        # Collect uploaded document source files
        for document in document_data.get("documents", []):
            storage_key = document.get("storage_key")
            if not storage_key:
                continue
            src = self.documents_dir / storage_key
            if src.exists():
                media_files[f"media/documents/{storage_key}"] = src
            else:
                logger.warning(f"Document source file not found: {src}")
        
        # Collect character profile image
        profile_image = self._find_profile_image(character_id, character_config)
        if profile_image and profile_image.exists():
            media_files[f"profile_image{profile_image.suffix}"] = profile_image
        
        logger.info(f"Collected {len(media_files)} media files for backup")
        return media_files
    
    def _find_profile_image(self, character_id: str, character_config: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Find character profile image."""
        configured_image = (character_config or {}).get("profile_image")
        if configured_image:
            configured_path = self.images_dir / Path(str(configured_image)).name
            if configured_path.exists():
                return configured_path

        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            # Try exact character_id match
            image_path = self.images_dir / f"{character_id}{ext}"
            if image_path.exists():
                return image_path
            
            # Try with character_id prefix (e.g., nova_profile.png)
            for img_file in self.images_dir.glob(f"{character_id}*{ext}"):
                return img_file
        
        return None
    
    def _get_character_config(self, character_id: str) -> Dict[str, Any]:
        """
        Load character YAML configuration.
        
        Args:
            character_id: Character ID
        
        Returns:
            Character config dict
        
        Raises:
            BackupError: If character YAML not found
        """
        yaml_path = self.characters_dir / f"{character_id}.yaml"
        
        if not yaml_path.exists():
            raise BackupError(f"Character YAML not found: {yaml_path}")
        
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded character configuration from {yaml_path}")
        return config
    
    def _collect_workflows(self, character_id: str) -> Dict[str, Dict[str, Path]]:
        """
        Collect workflow files for character.
        
        Args:
            character_id: Character ID
        
        Returns:
            Dict mapping workflow types to {name: path} dicts
        """
        workflows = {'image': {}, 'audio': {}, 'video': {}}
        
        canonical_dir = self.workflows_dir / character_id
        if canonical_dir.exists():
            for wf_type in ["image", "audio", "video"]:
                type_dir = canonical_dir / wf_type
                if type_dir.exists():
                    for workflow_file in type_dir.glob("*.json"):
                        workflows[wf_type][workflow_file.name] = workflow_file

        # Also support legacy layout workflows/{type}/{character_id}
        for wf_type in ["image", "audio", "video"]:
            legacy_dir = self.workflows_dir / wf_type / character_id
            if legacy_dir.exists():
                for workflow_file in legacy_dir.glob("*.json"):
                    workflows[wf_type].setdefault(workflow_file.name, workflow_file)
        
        total = sum(len(w) for w in workflows.values())
        logger.info(f"Collected {total} workflow files")
        
        return workflows
    
    def _generate_manifest(
        self,
        character_id: str,
        conversations: List[Dict],
        memories: List[Dict],
        workflow_records: List[Dict],
        media_records: Dict,
        moment_pins: List[Dict[str, Any]],
        continuity_data: Dict[str, Any],
        image_attachments: List[Dict[str, Any]],
        document_data: Dict[str, Any],
        media_files: Dict,
        workflows: Dict,
        character_config: Dict,
        notes: Optional[str]
    ) -> Dict[str, Any]:
        """Generate backup manifest with metadata."""
        
        # Get engine version from config
        try:
            config_loader = ConfigLoader()
            engine_version = "1.0.0"  # TODO: Add version to system config
        except:
            engine_version = "unknown"
        
        # Get current Alembic schema version
        try:
            from alembic import command
            from alembic.config import Config
            from alembic.script import ScriptDirectory
            
            alembic_cfg = Config("alembic.ini")
            script = ScriptDirectory.from_config(alembic_cfg)
            schema_version = script.get_current_head()
        except:
            schema_version = "unknown"
        
        # Count message totals
        message_count = sum(
            len(thread['messages'])
            for conv in conversations
            for thread in conv['threads']
        )
        
        thread_count = sum(len(conv['threads']) for conv in conversations)
        
        manifest = {
            'backup_format_version': self.BACKUP_FORMAT_VERSION,
            'engine_version': engine_version,
            'backup_date': datetime.now().isoformat(),
            'character_id': character_id,
            'character_name': character_config.get('name', character_id),
            'schema_version': schema_version,
            
            'counts': {
                'conversations': len(conversations),
                'threads': thread_count,
                'messages': message_count,
                'memories': len(memories),
                'workflow_records': len(workflow_records),
                'images': len(media_records['images']),
                'videos': len(media_records['videos']),
                'audio_files': len(media_records['audio_messages']),
                'voice_samples': len(media_records['voice_samples']),
                'media_files_total': len(media_files),
                'moment_pins': len(moment_pins),
                'image_attachments': len(image_attachments),
                'continuity_arcs': len(continuity_data.get("arcs", [])),
                'documents': len(document_data.get("documents", [])),
                'document_chunks': len(document_data.get("chunks", [])),
                'document_access_logs': len(document_data.get("access_logs", [])),
            },
            
            'files': {
                'character_yaml': 'character.yaml',
                'profile_image': next((k for k in media_files.keys() if k.startswith('profile_image')), None),
                'conversations': 'database/conversations.json',
                'memories': 'database/memories.json',
                'workflow_records': 'database/workflow_records.json',
                'media_records': 'database/media_records.json',
                'moment_pins': 'database/moment_pins.json',
                'continuity': 'database/continuity.json',
                'image_attachments': 'database/image_attachments.json',
                'documents': 'database/documents.json',
            },

            'workflow_counts': {
                'image': len(workflows['image']),
                'audio': len(workflows['audio']),
                'video': len(workflows['video'])
            }
        }
        
        if notes:
            manifest['notes'] = notes
        
        return manifest
    
    def _create_archive(
        self,
        character_id: str,
        timestamp: str,
        temp_dir: Path,
        manifest: Dict,
        conversations: List[Dict],
        memories: List[Dict],
        workflow_records: List[Dict],
        media_records: Dict,
        moment_pins: List[Dict[str, Any]],
        continuity_data: Dict[str, Any],
        image_attachments: List[Dict[str, Any]],
        document_data: Dict[str, Any],
        media_files: Dict,
        character_config: Dict,
        workflows: Dict
    ) -> Path:
        """
        Create ZIP archive with all backup data.
        
        Args:
            character_id: Character ID
            timestamp: Timestamp string for filename
            temp_dir: Temporary directory for building archive
            manifest: Backup manifest
            conversations: Conversations data
            memories: Memories data
            workflow_records: Workflow database records
            media_records: Media records data
            moment_pins: Moment pin records
            continuity_data: Continuity state records
            image_attachments: Image attachment records
            document_data: Document records
            media_files: Dict of media files to include
            character_config: Character YAML config
            workflows: Workflow files
        
        Returns:
            Path to created backup .zip file
        """
        # Create subdirectories in temp
        (temp_dir / 'database').mkdir(exist_ok=True)
        (temp_dir / 'workflows').mkdir(parents=True, exist_ok=True)
        
        # Write JSON files
        with open(temp_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        with open(temp_dir / 'database' / 'conversations.json', 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        with open(temp_dir / 'database' / 'memories.json', 'w', encoding='utf-8') as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)
        
        with open(temp_dir / 'database' / 'workflow_records.json', 'w', encoding='utf-8') as f:
            json.dump(workflow_records, f, indent=2, ensure_ascii=False)
        
        with open(temp_dir / 'database' / 'media_records.json', 'w', encoding='utf-8') as f:
            json.dump(media_records, f, indent=2, ensure_ascii=False)

        with open(temp_dir / 'database' / 'moment_pins.json', 'w', encoding='utf-8') as f:
            json.dump(moment_pins, f, indent=2, ensure_ascii=False)

        with open(temp_dir / 'database' / 'continuity.json', 'w', encoding='utf-8') as f:
            json.dump(continuity_data, f, indent=2, ensure_ascii=False)

        with open(temp_dir / 'database' / 'image_attachments.json', 'w', encoding='utf-8') as f:
            json.dump(image_attachments, f, indent=2, ensure_ascii=False)

        with open(temp_dir / 'database' / 'documents.json', 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
        
        # Write character YAML
        import yaml
        with open(temp_dir / 'character.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(character_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Copy media files
        for archive_path, source_path in media_files.items():
            dest_path = temp_dir / archive_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
        
        # Copy workflow files
        for workflow_type, workflow_dict in workflows.items():
            for workflow_name, workflow_path in workflow_dict.items():
                dest_path = temp_dir / 'workflows' / character_id / workflow_type / workflow_name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(workflow_path, dest_path)
        
        # Create ZIP archive
        backup_filename = f"{character_id}_backup_{timestamp}.zip"
        backup_path = self.backup_dir / character_id / backup_filename
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through temp_dir and add all files (cross-platform)
            for root, dirs, files in os.walk(temp_dir):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)
        
        return backup_path
