"""Character restore service for restoring character snapshots from backups."""

import logging
import json
import shutil
import zipfile
import os
import yaml
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_

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
from chorus_engine.models.workflow import Workflow
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore
from chorus_engine.db.moment_pin_vector_store import MomentPinVectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.utils.startup_sync import (
    sync_memory_vectors,
    sync_conversation_summary_vectors,
    sync_moment_pin_vectors,
)

logger = logging.getLogger(__name__)


class RestoreError(Exception):
    """Exception raised when character restoration fails."""
    pass


class CharacterRestoreService:
    """Service for restoring character backups."""
    
    SUPPORTED_BACKUP_VERSIONS = [1, 2]
    
    def __init__(
        self,
        db: Session,
        characters_dir: Path = None,
        data_dir: Path = None,
        workflows_dir: Path = None
    ):
        """
        Initialize restore service.
        
        Args:
            db: Database session
            characters_dir: Directory containing character YAML files
            data_dir: Root data directory
            workflows_dir: Directory containing workflow files
        """
        self.db = db
        self.characters_dir = characters_dir or Path("characters")
        self.data_dir = data_dir or Path("data")
        self.workflows_dir = workflows_dir or Path("workflows")
        
        # Initialize vector store
        vector_store_dir = self.data_dir / "vector_store"
        self.vector_store = VectorStore(persist_directory=vector_store_dir)
        
        # Create directories if needed
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
    
    def restore_character(
        self,
        backup_file: Path,
        new_character_id: Optional[str] = None,
        rename_if_exists: bool = False,
        overwrite: bool = False,
        cleanup_orphans: bool = False
    ) -> Dict[str, Any]:
        """
        Restore a character from backup file.
        
        Args:
            backup_file: Path to .zip backup file
            new_character_id: Optional custom ID for restored character (takes precedence over rename_if_exists)
            rename_if_exists: If True, auto-rename character if ID exists (appends timestamp)
            overwrite: If True, overwrite existing character (requires confirmation)
            cleanup_orphans: If True, automatically clean up orphaned data before restore
        
        Returns:
            Dict with restoration summary
        
        Raises:
            RestoreError: If restoration fails
        """
        logger.info(f"Starting restore from: {backup_file}")
        
        # Create temporary directory for extraction
        temp_dir = self.data_dir / "temp" / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Extract archive
            logger.info("Extracting backup archive...")
            self._extract_archive(backup_file, temp_dir)
            
            # Step 2: Load and validate manifest
            logger.info("Loading backup manifest...")
            manifest = self._load_manifest(temp_dir)
            
            # Step 3: Validate backup version
            self._validate_version(manifest)
            if manifest.get("backup_format_version") == 1:
                logger.warning("Restoring legacy v1 archive; vector payload will be ignored and rebuilt from SQL.")
            
            # Step 4: Check character collision
            character_id = manifest['character_id']
            original_id = character_id
            
            # If user specified a custom ID, use that
            if new_character_id:
                character_id = new_character_id
                logger.info(f"Using custom character ID: {character_id}")
            
            # Check for existing character and orphaned data
            yaml_exists, orphan_counts = self._character_exists(character_id)
            has_orphans = any(count > 0 for count in orphan_counts.values())
            
            if yaml_exists:
                # Character YAML exists - handle normally
                if overwrite:
                    logger.warning(f"Overwriting existing character: {character_id}")
                    self._delete_existing_character(character_id)
                elif rename_if_exists and not new_character_id:
                    # Only auto-rename if user didn't specify custom ID
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    character_id = f"{character_id}_restored_{timestamp}"
                    logger.info(f"Auto-renamed character from {original_id} to {character_id}")
                else:
                    raise RestoreError(
                        f"Character {character_id} already exists. "
                        f"Use new_character_id parameter, rename_if_exists=True, or overwrite=True"
                    )
            elif has_orphans:
                # No YAML but has orphaned data - this will cause duplicates!
                orphan_summary = ", ".join([f"{k}={v}" for k, v in orphan_counts.items() if v > 0])
                
                if cleanup_orphans:
                    logger.warning(f"Cleaning up orphaned data for {character_id}: {orphan_summary}")
                    deleted_counts = self._cleanup_orphaned_data(character_id)
                    logger.info(f"Orphaned data cleaned: {deleted_counts}")
                else:
                    logger.warning(f"Found orphaned data for {character_id}: {orphan_summary}")
                    raise RestoreError(
                        f"Character {character_id} has orphaned data from incomplete deletion: {orphan_summary}. "
                        f"This will cause duplicate conversations/memories. "
                        f"Use cleanup_orphans=True to remove orphaned data first, or use a different character_id."
                    )
            
            # Step 5: Restore character configuration
            logger.info("Restoring character configuration...")
            self._restore_character_config(temp_dir, character_id)
            
            # Step 6: Restore SQL data
            logger.info("Restoring database records...")
            restored_counts, id_maps = self._restore_sql_data(temp_dir, character_id, original_id)
            
            # Step 7: Restore media files
            logger.info("Restoring media files...")
            media_count = self._restore_media_files(temp_dir, character_id, id_maps['conversations'])
            restored_counts['media_files'] = media_count
            
            # Step 8: Restore workflows
            logger.info("Restoring workflow files...")
            workflow_count = self._restore_workflows(temp_dir, character_id)
            restored_counts['workflow_files'] = workflow_count

            # Step 9: Rebuild vectors from SQL (vectorless backup model)
            logger.info("Rebuilding vectors from restored SQL data...")
            rebuild_stats = self._rebuild_vectors(character_id)
            
            logger.info(f"Restore completed successfully for character: {character_id}")
            
            return {
                'character_id': character_id,
                'original_id': original_id,
                'renamed': character_id != original_id,
                'backup_date': manifest.get('backup_date'),
                'restored_counts': restored_counts,
                'rebuild_stats': rebuild_stats
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {e}", exc_info=True)
            raise RestoreError(f"Failed to restore character: {e}") from e
        
        finally:
            # Cleanup temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
    
    def _extract_archive(self, backup_file: Path, temp_dir: Path):
        """Extract backup archive to temporary directory."""
        if not backup_file.exists():
            raise RestoreError(f"Backup file not found: {backup_file}")
        
        try:
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(temp_dir)
            logger.debug(f"Extracted archive to: {temp_dir}")
        except zipfile.BadZipFile:
            raise RestoreError(f"Invalid backup file (not a valid ZIP): {backup_file}")
    
    def _load_manifest(self, temp_dir: Path) -> Dict[str, Any]:
        """Load and parse manifest.json from extracted backup."""
        manifest_path = temp_dir / "manifest.json"
        
        if not manifest_path.exists():
            raise RestoreError("Backup archive is missing manifest.json")
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            logger.debug(f"Loaded manifest: version={manifest.get('backup_format_version')}")
            return manifest
        except json.JSONDecodeError as e:
            raise RestoreError(f"Invalid manifest.json: {e}")
    
    def _validate_version(self, manifest: Dict[str, Any]):
        """Validate backup format version is supported."""
        version = manifest.get('backup_format_version')
        
        if version is None:
            raise RestoreError("Manifest missing backup_format_version")
        
        if version not in self.SUPPORTED_BACKUP_VERSIONS:
            raise RestoreError(
                f"Unsupported backup version {version}. "
                f"Supported versions: {self.SUPPORTED_BACKUP_VERSIONS}"
            )
    
    def _character_exists(self, character_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if character already exists and detect orphaned data.
        
        Returns:
            Tuple of (yaml_exists, orphan_counts)
            - yaml_exists: True if character YAML file exists
            - orphan_counts: Dict with counts of orphaned database records
        """
        yaml_exists = (self.characters_dir / f"{character_id}.yaml").exists()
        
        # Check for orphaned database records (from incomplete deletions)
        orphan_counts = {
            'conversations': 0,
            'memories': 0,
            'vectors': 0
        }
        
        # Count orphaned conversations
        orphan_counts['conversations'] = self.db.query(Conversation).filter(
            Conversation.character_id == character_id
        ).count()
        
        # Count orphaned memories
        orphan_counts['memories'] = self.db.query(Memory).filter(
            Memory.character_id == character_id
        ).count()
        
        # Check for orphaned vector collection
        collection = self.vector_store.get_collection(character_id)
        if collection:
            orphan_counts['vectors'] = collection.count()
        
        return yaml_exists, orphan_counts
    
    def _cleanup_orphaned_data(self, character_id: str) -> Dict[str, int]:
        """
        Clean up orphaned database records for a character.
        
        This removes data left behind from incomplete character deletions.
        
        Args:
            character_id: Character ID to clean up
            
        Returns:
            Dict with counts of deleted items
        """
        deleted_counts = {
            'conversations': 0,
            'threads': 0,
            'messages': 0,
            'memories': 0,
            'images': 0,
            'videos': 0,
            'audio': 0,
            'voice_samples': 0,
            'image_attachments': 0,
            'moment_pins': 0,
            'documents': 0,
            'document_chunks': 0,
            'document_access_logs': 0,
            'continuity_arcs': 0,
            'vectors': 0,
            'summary_vectors': 0,
            'moment_pin_vectors': 0
        }
        
        try:
            # Delete conversations (CASCADE will handle threads, messages, summaries, media)
            conversations = self.db.query(Conversation).filter(
                Conversation.character_id == character_id
            ).all()
            conversation_ids = [conv.id for conv in conversations]
            
            for conv in conversations:
                deleted_counts['conversations'] += 1
                # Count related records before deletion
                deleted_counts['threads'] += len(conv.threads)
                for thread in conv.threads:
                    deleted_counts['messages'] += len(thread.messages)
                
                self.db.delete(conv)
            
            # Delete memories
            memories = self.db.query(Memory).filter(
                Memory.character_id == character_id
            ).all()
            deleted_counts['memories'] = len(memories)
            for memory in memories:
                self.db.delete(memory)
            
            # Delete generated images
            images = self.db.query(GeneratedImage).filter(
                GeneratedImage.character_id == character_id
            ).all()
            deleted_counts['images'] = len(images)
            for image in images:
                self.db.delete(image)
            
            # Delete voice samples
            samples = self.db.query(VoiceSample).filter(
                VoiceSample.character_id == character_id
            ).all()
            deleted_counts['voice_samples'] = len(samples)
            for sample in samples:
                self.db.delete(sample)
            
            # Delete workflow records
            workflows = self.db.query(Workflow).filter(
                Workflow.character_name == character_id
            ).all()
            for workflow in workflows:
                self.db.delete(workflow)

            # Delete image attachments
            attachments = self.db.query(ImageAttachment).filter(
                ImageAttachment.character_id == character_id
            ).all()
            deleted_counts["image_attachments"] = len(attachments)
            for attachment in attachments:
                self.db.delete(attachment)

            # Delete moment pins
            pins = self.db.query(MomentPin).filter(
                MomentPin.character_id == character_id
            ).all()
            deleted_counts["moment_pins"] = len(pins)
            for pin in pins:
                self.db.delete(pin)

            # Delete continuity records
            cont_state = self.db.query(ContinuityRelationshipState).filter(
                ContinuityRelationshipState.character_id == character_id
            ).first()
            if cont_state:
                self.db.delete(cont_state)
            cont_arcs = self.db.query(ContinuityArc).filter(
                ContinuityArc.character_id == character_id
            ).all()
            deleted_counts["continuity_arcs"] = len(cont_arcs)
            for arc in cont_arcs:
                self.db.delete(arc)
            cont_cache = self.db.query(ContinuityBootstrapCache).filter(
                ContinuityBootstrapCache.character_id == character_id
            ).first()
            if cont_cache:
                self.db.delete(cont_cache)
            cont_pref = self.db.query(ContinuityPreference).filter(
                ContinuityPreference.character_id == character_id
            ).first()
            if cont_pref:
                self.db.delete(cont_pref)

            # Delete documents and related rows
            docs_query = self.db.query(Document).filter(Document.character_id == character_id)
            if conversation_ids:
                docs_query = self.db.query(Document).filter(
                    or_(
                        Document.character_id == character_id,
                        Document.conversation_id.in_(conversation_ids)
                    )
                )
            docs = docs_query.all()
            deleted_counts["documents"] = len(docs)
            doc_ids = [doc.id for doc in docs]
            if doc_ids:
                deleted_counts["document_chunks"] = self.db.query(DocumentChunk).filter(
                    DocumentChunk.document_id.in_(doc_ids)
                ).count()
                deleted_counts["document_access_logs"] = self.db.query(DocumentAccessLog).filter(
                    DocumentAccessLog.document_id.in_(doc_ids)
                ).count()
            for doc in docs:
                self.db.delete(doc)
            
            # Commit database changes
            self.db.commit()
            
            # Delete vector collection
            if self.vector_store.delete_collection(character_id):
                deleted_counts['vectors'] = 1

            # Delete summary and moment pin vector collections as well
            summary_store = ConversationSummaryVectorStore(self.data_dir / "vector_store")
            if summary_store.delete_collection(character_id):
                deleted_counts["summary_vectors"] = 1
            pin_store = MomentPinVectorStore(self.data_dir / "vector_store")
            collection = pin_store.get_collection(character_id)
            if collection is not None:
                try:
                    pin_store.client.delete_collection(name=f"moment_pins_{character_id}")
                    deleted_counts["moment_pin_vectors"] = 1
                except Exception:
                    pass
            
            logger.info(f"Cleaned up orphaned data for {character_id}: {deleted_counts}")
            return deleted_counts
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to cleanup orphaned data: {e}")
            raise RestoreError(f"Cleanup failed: {e}")
    
    def _delete_existing_character(self, character_id: str):
        """
        Delete existing character completely (for overwrite mode).
        
        Removes:
        - Character YAML configuration
        - Profile images
        - All database records (conversations, memories, etc.)
        - Vector store collection
        - Media files (images, videos, audio, voice samples)
        - Workflow files
        
        Args:
            character_id: Character ID to delete
        """
        try:
            deleted_items = []
            
            # Delete character YAML
            yaml_path = self.characters_dir / f"{character_id}.yaml"
            if yaml_path.exists():
                yaml_path.unlink()
                deleted_items.append("YAML")
                logger.debug(f"Deleted character YAML: {yaml_path}")
            
            # Delete profile images (may have various extensions)
            profile_dir = self.data_dir / "character_images"
            if profile_dir.exists():
                for ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                    for profile_file in profile_dir.glob(f"{character_id}.*{ext}"):
                        profile_file.unlink()
                        deleted_items.append(f"profile image ({profile_file.name})")
                        logger.debug(f"Deleted profile image: {profile_file}")
            
            # Delete media files
            # Images
            images_dir = self.data_dir / "images"
            if images_dir.exists():
                for conv_dir in images_dir.iterdir():
                    if conv_dir.is_dir():
                        # Check if this conversation belongs to the character
                        conv = self.db.query(Conversation).filter(
                            Conversation.id == conv_dir.name,
                            Conversation.character_id == character_id
                        ).first()
                        if conv:
                            shutil.rmtree(conv_dir)
                            deleted_items.append(f"image directory ({conv_dir.name})")
            
            # Videos
            videos_dir = self.data_dir / "videos"
            if videos_dir.exists():
                for conv_dir in videos_dir.iterdir():
                    if conv_dir.is_dir():
                        conv = self.db.query(Conversation).filter(
                            Conversation.id == conv_dir.name,
                            Conversation.character_id == character_id
                        ).first()
                        if conv:
                            shutil.rmtree(conv_dir)
                            deleted_items.append(f"video directory ({conv_dir.name})")
            
            # Voice samples
            voice_samples_dir = self.data_dir / "voice_samples" / character_id
            if voice_samples_dir.exists():
                shutil.rmtree(voice_samples_dir)
                deleted_items.append("voice samples")
            
            # Workflow files (canonical and legacy layouts)
            canonical_workflows_dir = self.workflows_dir / character_id
            if canonical_workflows_dir.exists():
                shutil.rmtree(canonical_workflows_dir)
                deleted_items.append("workflows")

            workflow_types = ['image', 'audio', 'video']
            for wf_type in workflow_types:
                workflow_dir = self.workflows_dir / wf_type / character_id
                if workflow_dir.exists():
                    shutil.rmtree(workflow_dir)
                    deleted_items.append(f"{wf_type} workflows")

            # Remove attachment files for this character.
            attachments = self.db.query(ImageAttachment).filter(
                ImageAttachment.character_id == character_id
            ).all()
            for attachment in attachments:
                for candidate in [attachment.original_path, attachment.processed_path]:
                    if not candidate:
                        continue
                    path_obj = Path(candidate)
                    if path_obj.exists():
                        try:
                            path_obj.unlink()
                        except Exception:
                            logger.debug(f"Could not delete attachment file {path_obj}")

            # Remove stored document files for this character.
            docs = self.db.query(Document).filter(Document.character_id == character_id).all()
            if not docs:
                conv_ids = [row[0] for row in self.db.query(Conversation.id).filter(Conversation.character_id == character_id).all()]
                if conv_ids:
                    docs = self.db.query(Document).filter(Document.conversation_id.in_(conv_ids)).all()
            for doc in docs:
                storage_path = self.data_dir / "documents" / doc.storage_key
                if storage_path.exists():
                    try:
                        storage_path.unlink()
                    except Exception:
                        logger.debug(f"Could not delete document file {storage_path}")
            
            # Clean up all database records and vectors
            deleted_counts = self._cleanup_orphaned_data(character_id)
            
            logger.info(f"Deleted character {character_id}: {', '.join(deleted_items)}")
            logger.info(f"Database cleanup: {deleted_counts}")
            
        except Exception as e:
            logger.error(f"Failed to delete character {character_id}: {e}")
            raise RestoreError(f"Character deletion failed: {e}")
    
    def _restore_character_config(self, temp_dir: Path, character_id: str):
        """Restore character.yaml file."""
        source_yaml = temp_dir / "character.yaml"
        dest_yaml = self.characters_dir / f"{character_id}.yaml"
        
        if not source_yaml.exists():
            raise RestoreError("Backup missing character.yaml")
        
        # Load and potentially update character_id if renamed
        with open(source_yaml, 'r', encoding='utf-8') as f:
            char_config = yaml.safe_load(f)
        
        # Update ID if character was renamed
        if 'id' in char_config:
            char_config['id'] = character_id
        
        # Write to characters directory
        with open(dest_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(char_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Restored character configuration to: {dest_yaml}")
        
        # Restore profile image if present
        profile_images = list(temp_dir.glob("profile_image.*"))
        if profile_images:
            profile_src = profile_images[0]
            profile_dest = self.data_dir / "character_images" / f"{character_id}{profile_src.suffix}"
            profile_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(profile_src, profile_dest)
            logger.debug(f"Restored profile image to: {profile_dest}")
    
    def _restore_sql_data(
        self,
        temp_dir: Path,
        character_id: str,
        original_id: str
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, str]]]:
        """
        Restore all SQL database records with ID remapping.
        
        Generates new UUIDs for all primary keys and updates all foreign key references
        to prevent collisions with existing data. This allows restoring the same backup
        multiple times or restoring into a database with orphaned records.
        
        Returns:
            Tuple of (counts dict, id_maps dict)
        """
        # ID mapping tables to track old_id -> new_id conversions
        id_maps = {
            'conversations': {},
            'threads': {},
            'messages': {},
            'memories': {},
            'moment_pins': {},
            'image_attachments': {},
            'documents': {},
            'document_chunks': {},
            'voice_samples': {},
        }
        
        counts = {
            'conversations': 0,
            'threads': 0,
            'messages': 0,
            'memories': 0,
            'workflow_records': 0,
            'images': 0,
            'videos': 0,
            'audio': 0,
            'voice_samples': 0,
            'moment_pins': 0,
            'image_attachments': 0,
            'continuity_arcs': 0,
            'documents': 0,
            'document_chunks': 0,
            'document_access_logs': 0,
        }
        
        db_dir = temp_dir / "database"
        
        # Restore conversations (with nested threads and messages)
        conversations_file = db_dir / "conversations.json"
        if conversations_file.exists():
            with open(conversations_file, 'r', encoding='utf-8') as f:
                conversations_data = json.load(f)
            
            for conv_data in conversations_data:
                # Generate new conversation ID and track mapping
                old_conv_id = conv_data['id']
                new_conv_id = str(uuid.uuid4())
                id_maps['conversations'][old_conv_id] = new_conv_id
                
                # Update character_id if renamed
                conv_data['character_id'] = character_id
                
                # Create conversation with new ID
                conv = Conversation(
                    id=new_conv_id,
                    character_id=conv_data['character_id'],
                    title=conv_data['title'],
                    created_at=datetime.fromisoformat(conv_data['created_at']),
                    updated_at=datetime.fromisoformat(conv_data['updated_at']),
                    is_private=conv_data.get('is_private', 'false'),
                    tts_enabled=conv_data.get('tts_enabled'),
                    last_extracted_message_count=conv_data.get('last_extracted_message_count', 0),
                    image_confirmation_disabled=conv_data.get('image_confirmation_disabled', 'false'),
                    video_confirmation_disabled=conv_data.get('video_confirmation_disabled', 'false'),
                    allow_image_offers=conv_data.get('allow_image_offers', 'true'),
                    allow_video_offers=conv_data.get('allow_video_offers', 'true'),
                    last_image_offer_at=datetime.fromisoformat(conv_data['last_image_offer_at']) if conv_data.get('last_image_offer_at') else None,
                    last_video_offer_at=datetime.fromisoformat(conv_data['last_video_offer_at']) if conv_data.get('last_video_offer_at') else None,
                    last_image_offer_message_count=conv_data.get('last_image_offer_message_count'),
                    last_video_offer_message_count=conv_data.get('last_video_offer_message_count'),
                    image_offer_count=conv_data.get('image_offer_count', 0),
                    video_offer_count=conv_data.get('video_offer_count', 0),
                    continuity_mode=conv_data.get('continuity_mode', 'ask'),
                    continuity_choice_remembered=conv_data.get('continuity_choice_remembered', 'false'),
                    primary_user=conv_data.get('primary_user'),
                    title_auto_generated=conv_data.get('title_auto_generated'),
                    last_analyzed_at=datetime.fromisoformat(conv_data['last_analyzed_at']) if conv_data.get('last_analyzed_at') else None,
                    last_summary_analyzed_at=datetime.fromisoformat(conv_data['last_summary_analyzed_at']) if conv_data.get('last_summary_analyzed_at') else None,
                    last_memories_analyzed_at=datetime.fromisoformat(conv_data['last_memories_analyzed_at']) if conv_data.get('last_memories_analyzed_at') else None,
                    source=conv_data.get('source', 'web')
                )
                self.db.add(conv)
                counts['conversations'] += 1
                
                # Restore threads
                for thread_data in conv_data.get('threads', []):
                    # Generate new thread ID and track mapping
                    old_thread_id = thread_data['id']
                    new_thread_id = str(uuid.uuid4())
                    id_maps['threads'][old_thread_id] = new_thread_id
                    
                    thread = Thread(
                        id=new_thread_id,
                        conversation_id=new_conv_id,  # Use new conversation ID
                        title=thread_data['title'],
                        created_at=datetime.fromisoformat(thread_data['created_at']),
                        updated_at=datetime.fromisoformat(thread_data['updated_at'])
                    )
                    self.db.add(thread)
                    counts['threads'] += 1
                    
                    # Restore messages
                    for msg_data in thread_data.get('messages', []):
                        # Generate new message ID and track mapping
                        old_msg_id = msg_data['id']
                        new_msg_id = str(uuid.uuid4())
                        id_maps['messages'][old_msg_id] = new_msg_id
                        
                        message = Message(
                            id=new_msg_id,
                            thread_id=new_thread_id,  # Use new thread ID
                            role=msg_data['role'],
                            content=msg_data['content'],
                            meta_data=msg_data.get('meta_data'),
                            created_at=datetime.fromisoformat(msg_data['created_at']),
                            is_private=msg_data.get('is_private', 'false'),
                            emotional_weight=msg_data.get('emotional_weight'),
                            summary=msg_data.get('summary'),
                            preserve_full_text=msg_data.get('preserve_full_text', 'true'),
                            deleted_at=datetime.fromisoformat(msg_data['deleted_at']) if msg_data.get('deleted_at') else None,
                        )
                        self.db.add(message)
                        counts['messages'] += 1
                
                # Restore conversation summary if present
                if 'summary' in conv_data and conv_data['summary']:
                    summary_data = conv_data['summary']
                    old_thread_id = summary_data.get('thread_id')
                    new_thread_id = id_maps['threads'].get(old_thread_id) if old_thread_id else None
                    
                    summary = ConversationSummary(
                        id=str(uuid.uuid4()),  # Generate new ID
                        conversation_id=new_conv_id,  # Use new conversation ID
                        thread_id=new_thread_id,  # Use new thread ID if present
                        summary=summary_data['summary'],
                        summary_type=summary_data.get('summary_type', 'progressive'),
                        message_range_start=summary_data['message_range_start'],
                        message_range_end=summary_data['message_range_end'],
                        message_count=summary_data['message_count'],
                        key_topics=summary_data.get('key_topics'),
                        participants=summary_data.get('participants'),
                        emotional_arc=summary_data.get('emotional_arc'),
                        tone=summary_data.get('tone'),
                        open_questions=summary_data.get('open_questions'),
                        manual=summary_data.get('manual', 'false'),
                        created_at=datetime.fromisoformat(summary_data['created_at'])
                    )
                    self.db.add(summary)
        
        # Restore memories
        memories_file = db_dir / "memories.json"
        if memories_file.exists():
            with open(memories_file, 'r', encoding='utf-8') as f:
                memories_data = json.load(f)
            
            for mem_data in memories_data:
                # Generate new memory ID and track mapping
                old_mem_id = mem_data['id']
                new_mem_id = str(uuid.uuid4())
                id_maps['memories'][old_mem_id] = new_mem_id
                
                # Update character_id if renamed
                mem_data['character_id'] = character_id
                
                # Remap conversation_id and thread_id if they exist
                old_conv_id = mem_data.get('conversation_id')
                new_conv_id = id_maps['conversations'].get(old_conv_id) if old_conv_id else None
                
                old_thread_id = mem_data.get('thread_id')
                new_thread_id = id_maps['threads'].get(old_thread_id) if old_thread_id else None
                
                # Remap source_messages (list of message IDs)
                source_messages = mem_data.get('source_messages')
                if source_messages and isinstance(source_messages, list):
                    new_source_messages = []
                    for old_msg_id in source_messages:
                        new_msg_id = id_maps['messages'].get(old_msg_id)
                        if new_msg_id:
                            new_source_messages.append(new_msg_id)
                    source_messages = new_source_messages if new_source_messages else None
                
                memory = Memory(
                    id=new_mem_id,
                    conversation_id=new_conv_id,
                    thread_id=new_thread_id,
                    character_id=mem_data['character_id'],
                    memory_type=mem_data['memory_type'],
                    content=mem_data['content'],
                    vector_id=None,
                    embedding_model=mem_data.get('embedding_model', 'all-MiniLM-L6-v2'),
                    priority=mem_data.get('priority', 50),
                    tags=mem_data.get('tags'),
                    confidence=mem_data.get('confidence'),
                    category=mem_data.get('category'),
                    status=mem_data.get('status', 'approved'),
                    source_messages=source_messages,
                    durability=mem_data.get('durability', 'situational'),
                    pattern_eligible=int(mem_data.get('pattern_eligible', 0)),
                    emotional_weight=mem_data.get('emotional_weight'),
                    participants=mem_data.get('participants'),
                    key_moments=mem_data.get('key_moments'),
                    summary=mem_data.get('summary'),
                    source=mem_data.get('source', 'web'),
                    meta_data=mem_data.get('meta_data'),
                    created_at=datetime.fromisoformat(mem_data['created_at'])
                )
                self.db.add(memory)
                counts['memories'] += 1
        
        # Restore workflow records
        workflow_records_file = db_dir / "workflow_records.json"
        if workflow_records_file.exists():
            with open(workflow_records_file, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            for wf_data in workflow_data:
                # Update character_name if renamed
                wf_data['character_name'] = character_id
                
                workflow = Workflow(
                    # Generate new ID (workflows use integer autoincrement, so let DB assign)
                    character_name=wf_data['character_name'],
                    workflow_name=wf_data['workflow_name'],
                    workflow_file_path=wf_data['workflow_file_path'],
                    workflow_type=wf_data.get('workflow_type', 'image'),
                    is_default=wf_data.get('is_default', False),
                    trigger_word=wf_data.get('trigger_word'),
                    default_style=wf_data.get('default_style'),
                    negative_prompt=wf_data.get('negative_prompt'),
                    self_description=wf_data.get('self_description'),
                    created_at=datetime.fromisoformat(wf_data['created_at']),
                    updated_at=datetime.fromisoformat(wf_data['updated_at'])
                )
                self.db.add(workflow)
                counts['workflow_records'] += 1
        
        # Restore media records
        media_records_file = db_dir / "media_records.json"
        if media_records_file.exists():
            with open(media_records_file, 'r', encoding='utf-8') as f:
                media_data = json.load(f)
            
            # Restore images
            for img_data in media_data.get('images', []):
                # Update character_id if renamed
                img_data['character_id'] = character_id
                
                # Remap conversation_id, thread_id, message_id
                old_conv_id = img_data['conversation_id']
                new_conv_id = id_maps['conversations'].get(old_conv_id)
                
                old_thread_id = img_data['thread_id']
                new_thread_id = id_maps['threads'].get(old_thread_id)
                
                old_msg_id = img_data.get('message_id')
                new_msg_id = id_maps['messages'].get(old_msg_id) if old_msg_id else None
                
                if new_conv_id and new_thread_id:  # Only add if IDs mapped successfully
                    remapped_image_path = self._remap_conversation_media_path(
                        img_data['file_path'],
                        old_conv_id,
                        new_conv_id
                    )
                    remapped_thumb_path = self._remap_conversation_media_path(
                        img_data.get('thumbnail_path'),
                        old_conv_id,
                        new_conv_id
                    )
                    image = GeneratedImage(
                        # Let DB assign new ID (integer autoincrement)
                        conversation_id=new_conv_id,
                        thread_id=new_thread_id,
                        message_id=new_msg_id,
                        character_id=img_data['character_id'],
                        prompt=img_data['prompt'],
                        negative_prompt=img_data.get('negative_prompt'),
                        workflow_file=img_data['workflow_file'],
                        file_path=remapped_image_path,
                        thumbnail_path=remapped_thumb_path,
                        width=img_data.get('width'),
                        height=img_data.get('height'),
                        seed=img_data.get('seed'),
                        notes=img_data.get('notes'),
                        created_at=datetime.fromisoformat(img_data['created_at']),
                        generation_time=img_data.get('generation_time')
                    )
                    self.db.add(image)
                    counts['images'] += 1
            
            # Restore videos
            for vid_data in media_data.get('videos', []):
                # Remap conversation_id
                old_conv_id = vid_data['conversation_id']
                new_conv_id = id_maps['conversations'].get(old_conv_id)
                
                if new_conv_id:  # Only add if ID mapped successfully
                    remapped_video_path = self._remap_conversation_media_path(
                        vid_data['file_path'],
                        old_conv_id,
                        new_conv_id
                    )
                    remapped_thumb_path = self._remap_conversation_media_path(
                        vid_data.get('thumbnail_path'),
                        old_conv_id,
                        new_conv_id
                    )
                    video = GeneratedVideo(
                        # Let DB assign new ID (integer autoincrement)
                        conversation_id=new_conv_id,
                        file_path=remapped_video_path,
                        thumbnail_path=remapped_thumb_path,
                        format=vid_data.get('format'),
                        duration_seconds=vid_data.get('duration_seconds'),
                        width=vid_data.get('width'),
                        height=vid_data.get('height'),
                        prompt=vid_data['prompt'],
                        negative_prompt=vid_data.get('negative_prompt'),
                        workflow_file=vid_data.get('workflow_file'),
                        comfy_prompt_id=vid_data.get('comfy_prompt_id'),
                        generation_time_seconds=vid_data.get('generation_time_seconds'),
                        created_at=datetime.fromisoformat(vid_data['created_at'])
                    )
                    self.db.add(video)
                    counts['videos'] += 1
            
            # Restore voice samples
            for sample_data in media_data.get('voice_samples', []):
                # Update character_id if renamed
                sample_data['character_id'] = character_id

                old_voice_id = sample_data.get("id")
                sample = VoiceSample(
                    # Let DB assign new ID (integer autoincrement)
                    character_id=sample_data['character_id'],
                    filename=sample_data['filename'],
                    transcript=sample_data['transcript'],
                    is_default=sample_data.get('is_default', 0),
                    uploaded_at=datetime.fromisoformat(sample_data['uploaded_at'])
                )
                self.db.add(sample)
                self.db.flush()
                if old_voice_id is not None:
                    id_maps["voice_samples"][str(old_voice_id)] = sample.id
                counts['voice_samples'] += 1

            # Restore audio messages after voice sample IDs are remapped
            for audio_data in media_data.get('audio_messages', []):
                old_msg_id = audio_data['message_id']
                new_msg_id = id_maps['messages'].get(old_msg_id)

                old_voice_id = audio_data.get("voice_sample_id")
                new_voice_id = None
                if old_voice_id is not None:
                    new_voice_id = id_maps["voice_samples"].get(str(old_voice_id))

                if new_msg_id:
                    audio = AudioMessage(
                        message_id=new_msg_id,
                        audio_filename=audio_data['audio_filename'],
                        workflow_name=audio_data.get('workflow_name'),
                        generation_duration=audio_data.get('generation_duration'),
                        text_preprocessed=audio_data.get('text_preprocessed'),
                        voice_sample_id=new_voice_id,
                        created_at=datetime.fromisoformat(audio_data['created_at'])
                    )
                    self.db.add(audio)
                    counts['audio'] += 1

        # Restore moment pins (v2)
        moment_pins_file = db_dir / "moment_pins.json"
        if moment_pins_file.exists():
            with open(moment_pins_file, 'r', encoding='utf-8') as f:
                pins_data = json.load(f)

            for pin_data in pins_data:
                old_pin_id = pin_data["id"]
                new_pin_id = str(uuid.uuid4())
                id_maps["moment_pins"][old_pin_id] = new_pin_id

                old_conv_id = pin_data.get("conversation_id")
                new_conv_id = id_maps["conversations"].get(old_conv_id) if old_conv_id else None
                selected_messages = []
                for old_msg in pin_data.get("selected_message_ids", []) or []:
                    new_msg = id_maps["messages"].get(old_msg)
                    if new_msg:
                        selected_messages.append(new_msg)

                pin = MomentPin(
                    id=new_pin_id,
                    user_id=pin_data.get("user_id", "User"),
                    character_id=character_id,
                    conversation_id=new_conv_id,
                    created_at=datetime.fromisoformat(pin_data["created_at"]) if pin_data.get("created_at") else datetime.utcnow(),
                    selected_message_ids=selected_messages,
                    transcript_snapshot=pin_data.get("transcript_snapshot", ""),
                    what_happened=pin_data.get("what_happened", ""),
                    why_model=pin_data.get("why_model", ""),
                    why_user=pin_data.get("why_user"),
                    quote_snippet=pin_data.get("quote_snippet"),
                    tags=pin_data.get("tags") or [],
                    reinforcement_score=pin_data.get("reinforcement_score", 1.0),
                    turns_since_reinforcement=pin_data.get("turns_since_reinforcement", 0),
                    archived=pin_data.get("archived", 0),
                    telemetry_flags=pin_data.get("telemetry_flags") or {},
                    vector_id=None,
                )
                self.db.add(pin)
                counts["moment_pins"] += 1

        # Restore continuity tables (v2)
        continuity_file = db_dir / "continuity.json"
        if continuity_file.exists():
            with open(continuity_file, 'r', encoding='utf-8') as f:
                continuity_data = json.load(f) or {}

            state_data = continuity_data.get("relationship_state")
            if state_data:
                self.db.add(ContinuityRelationshipState(
                    id=str(uuid.uuid4()),
                    character_id=character_id,
                    familiarity_level=state_data.get("familiarity_level", "new"),
                    tone_baseline=state_data.get("tone_baseline") or [],
                    interaction_contract=state_data.get("interaction_contract") or [],
                    boundaries=state_data.get("boundaries") or [],
                    assistant_role_frame=state_data.get("assistant_role_frame", ""),
                    updated_at=datetime.fromisoformat(state_data["updated_at"]) if state_data.get("updated_at") else datetime.utcnow(),
                ))

            for arc_data in continuity_data.get("arcs", []) or []:
                self.db.add(ContinuityArc(
                    id=str(uuid.uuid4()),
                    character_id=character_id,
                    title=arc_data.get("title", "Untitled"),
                    kind=arc_data.get("kind", "theme"),
                    summary=arc_data.get("summary", ""),
                    status=arc_data.get("status", "active"),
                    confidence=arc_data.get("confidence", "medium"),
                    stickiness=arc_data.get("stickiness", "normal"),
                    last_touched_conversation_id=id_maps["conversations"].get(arc_data.get("last_touched_conversation_id")),
                    last_touched_conversation_at=datetime.fromisoformat(arc_data["last_touched_conversation_at"]) if arc_data.get("last_touched_conversation_at") else None,
                    frequency_count=arc_data.get("frequency_count", 0),
                    created_at=datetime.fromisoformat(arc_data["created_at"]) if arc_data.get("created_at") else datetime.utcnow(),
                    updated_at=datetime.fromisoformat(arc_data["updated_at"]) if arc_data.get("updated_at") else datetime.utcnow(),
                ))
                counts["continuity_arcs"] += 1

            cache_data = continuity_data.get("bootstrap_cache")
            if cache_data:
                self.db.add(ContinuityBootstrapCache(
                    id=str(uuid.uuid4()),
                    character_id=character_id,
                    bootstrap_packet_internal=cache_data.get("bootstrap_packet_internal", ""),
                    bootstrap_packet_user_preview=cache_data.get("bootstrap_packet_user_preview", ""),
                    bootstrap_generated_at=datetime.fromisoformat(cache_data["bootstrap_generated_at"]) if cache_data.get("bootstrap_generated_at") else None,
                    bootstrap_inputs_fingerprint=cache_data.get("bootstrap_inputs_fingerprint"),
                    updated_at=datetime.fromisoformat(cache_data["updated_at"]) if cache_data.get("updated_at") else datetime.utcnow(),
                ))

            pref_data = continuity_data.get("preference")
            if pref_data:
                self.db.add(ContinuityPreference(
                    character_id=character_id,
                    default_mode=pref_data.get("default_mode", "ask"),
                    skip_preview=pref_data.get("skip_preview", 0),
                    updated_at=datetime.fromisoformat(pref_data["updated_at"]) if pref_data.get("updated_at") else datetime.utcnow(),
                ))

        # Restore image attachments (v2)
        attachments_file = db_dir / "image_attachments.json"
        if attachments_file.exists():
            with open(attachments_file, 'r', encoding='utf-8') as f:
                attachments_data = json.load(f) or []
            for row in attachments_data:
                old_id = row.get("id")
                new_id = str(uuid.uuid4())
                if old_id:
                    id_maps["image_attachments"][old_id] = new_id
                old_conv = row.get("conversation_id")
                new_conv = id_maps["conversations"].get(old_conv)
                old_msg = row.get("message_id")
                new_msg = id_maps["messages"].get(old_msg)
                if not new_conv or not new_msg:
                    continue
                attachment = ImageAttachment(
                    id=new_id,
                    message_id=new_msg,
                    conversation_id=new_conv,
                    character_id=character_id,
                    original_path=self._normalize_restored_attachment_path(row.get("original_path"), processed=False),
                    processed_path=self._normalize_restored_attachment_path(row.get("processed_path"), processed=True),
                    original_filename=row.get("original_filename"),
                    file_size=row.get("file_size"),
                    mime_type=row.get("mime_type"),
                    width=row.get("width"),
                    height=row.get("height"),
                    uploaded_at=datetime.fromisoformat(row["uploaded_at"]) if row.get("uploaded_at") else datetime.utcnow(),
                    vision_processed=row.get("vision_processed", "false"),
                    vision_skipped=row.get("vision_skipped", "false"),
                    vision_skip_reason=row.get("vision_skip_reason"),
                    vision_model=row.get("vision_model"),
                    vision_backend=row.get("vision_backend"),
                    vision_processed_at=datetime.fromisoformat(row["vision_processed_at"]) if row.get("vision_processed_at") else None,
                    vision_processing_time_ms=row.get("vision_processing_time_ms"),
                    vision_observation=row.get("vision_observation"),
                    vision_confidence=row.get("vision_confidence"),
                    vision_tags=row.get("vision_tags"),
                    description=row.get("description"),
                    source=row.get("source", "web"),
                )
                self.db.add(attachment)
                counts["image_attachments"] += 1

        # Restore documents/chunks/access logs (v2)
        documents_file = db_dir / "documents.json"
        if documents_file.exists():
            with open(documents_file, 'r', encoding='utf-8') as f:
                document_data = json.load(f) or {}

            for doc_data in document_data.get("documents", []):
                old_doc_id = doc_data.get("id")
                old_conv = doc_data.get("conversation_id")
                new_conv = id_maps["conversations"].get(old_conv) if old_conv else None
                document = Document(
                    filename=doc_data.get("filename"),
                    storage_key=doc_data.get("storage_key"),
                    file_type=doc_data.get("file_type"),
                    file_size_bytes=doc_data.get("file_size_bytes", 0),
                    page_count=doc_data.get("page_count"),
                    title=doc_data.get("title"),
                    description=doc_data.get("description"),
                    author=doc_data.get("author"),
                    character_id=character_id if doc_data.get("character_id") else None,
                    created_at=datetime.fromisoformat(doc_data["created_at"]) if doc_data.get("created_at") else datetime.utcnow(),
                    uploaded_at=datetime.fromisoformat(doc_data["uploaded_at"]) if doc_data.get("uploaded_at") else datetime.utcnow(),
                    last_accessed=datetime.fromisoformat(doc_data["last_accessed"]) if doc_data.get("last_accessed") else None,
                    document_scope=doc_data.get("document_scope", "conversation"),
                    conversation_id=new_conv,
                    processing_status=doc_data.get("processing_status", "completed"),
                    processing_error=doc_data.get("processing_error"),
                    chunk_count=doc_data.get("chunk_count", 0),
                    vector_collection_id=None,
                    metadata_json=doc_data.get("metadata_json"),
                )
                self.db.add(document)
                self.db.flush()
                if old_doc_id is not None:
                    id_maps["documents"][str(old_doc_id)] = document.id
                counts["documents"] += 1

            for chunk_data in document_data.get("chunks", []):
                old_doc_id = chunk_data.get("document_id")
                new_doc_id = id_maps["documents"].get(str(old_doc_id))
                if not new_doc_id:
                    continue
                new_chunk_id = f"doc_{new_doc_id}_chunk_{chunk_data.get('chunk_index', 0)}_{uuid.uuid4().hex[:8]}"
                old_chunk_id = chunk_data.get("chunk_id")
                if old_chunk_id:
                    id_maps["document_chunks"][old_chunk_id] = new_chunk_id
                chunk = DocumentChunk(
                    document_id=new_doc_id,
                    chunk_index=chunk_data.get("chunk_index", 0),
                    chunk_id=new_chunk_id,
                    content=chunk_data.get("content", ""),
                    content_length=chunk_data.get("content_length", len(chunk_data.get("content", ""))),
                    page_numbers=chunk_data.get("page_numbers"),
                    start_line=chunk_data.get("start_line"),
                    end_line=chunk_data.get("end_line"),
                    chunk_method=chunk_data.get("chunk_method", "semantic"),
                    overlap_tokens=chunk_data.get("overlap_tokens", 0),
                    embedding_model=chunk_data.get("embedding_model"),
                    embedding_created_at=datetime.fromisoformat(chunk_data["embedding_created_at"]) if chunk_data.get("embedding_created_at") else None,
                    metadata_json=chunk_data.get("metadata_json"),
                    created_at=datetime.fromisoformat(chunk_data["created_at"]) if chunk_data.get("created_at") else datetime.utcnow(),
                )
                self.db.add(chunk)
                counts["document_chunks"] += 1

            for log_data in document_data.get("access_logs", []):
                old_doc_id = log_data.get("document_id")
                new_doc_id = id_maps["documents"].get(str(old_doc_id))
                if not new_doc_id:
                    continue
                old_conv = log_data.get("conversation_id")
                new_conv = id_maps["conversations"].get(old_conv) if old_conv else None
                self.db.add(DocumentAccessLog(
                    document_id=new_doc_id,
                    conversation_id=new_conv,
                    message_id=None,
                    access_type=log_data.get("access_type", "retrieval"),
                    chunks_retrieved=log_data.get("chunks_retrieved", 0),
                    chunk_ids=self._remap_chunk_id_csv(log_data.get("chunk_ids"), id_maps["document_chunks"]),
                    query=log_data.get("query"),
                    relevance_score=log_data.get("relevance_score"),
                    accessed_at=datetime.fromisoformat(log_data["accessed_at"]) if log_data.get("accessed_at") else datetime.utcnow(),
                ))
                counts["document_access_logs"] += 1
        
        # Commit all changes
        self.db.commit()
        
        logger.info(f"Restored SQL data: {counts}")
        return counts, id_maps
    
    def _restore_media_files(
        self,
        temp_dir: Path,
        character_id: str,
        conversation_id_map: Dict[str, str]
    ) -> int:
        """
        Restore media files to data directory.
        
        Returns:
            Number of files restored
        """
        media_dir = temp_dir / "media"
        
        if not media_dir.exists():
            logger.info("No media files in backup")
            return 0
        
        file_count = 0
        
        # Restore images (conversation IDs remapped)
        images_src = media_dir / "images"
        if images_src.exists():
            for old_conv_dir in images_src.iterdir():
                if not old_conv_dir.is_dir():
                    continue
                new_conv_id = conversation_id_map.get(old_conv_dir.name, old_conv_dir.name)
                images_dest = self.data_dir / "images" / new_conv_id
                images_dest.mkdir(parents=True, exist_ok=True)
                file_count += self._copy_directory_tree(old_conv_dir, images_dest)
        
        # Restore videos (conversation IDs remapped)
        videos_src = media_dir / "videos"
        if videos_src.exists():
            for old_conv_dir in videos_src.iterdir():
                if not old_conv_dir.is_dir():
                    continue
                new_conv_id = conversation_id_map.get(old_conv_dir.name, old_conv_dir.name)
                videos_dest = self.data_dir / "videos" / new_conv_id
                videos_dest.mkdir(parents=True, exist_ok=True)
                file_count += self._copy_directory_tree(old_conv_dir, videos_dest)
        
        # Restore audio
        audio_src = media_dir / "audio"
        if audio_src.exists():
            audio_dest = self.data_dir / "audio"
            audio_dest.mkdir(parents=True, exist_ok=True)
            file_count += self._copy_directory_tree(audio_src, audio_dest)
        
        # Restore voice samples
        voice_src = media_dir / "voice_samples"
        if voice_src.exists():
            voice_dest = self.data_dir / "voice_samples" / character_id
            voice_dest.mkdir(parents=True, exist_ok=True)
            # Backward compat: old backups may store directly under voice_samples/
            char_specific_src = voice_src / character_id
            if char_specific_src.exists():
                file_count += self._copy_directory_tree(char_specific_src, voice_dest)
            else:
                file_count += self._copy_directory_tree(voice_src, voice_dest)

        # Restore image attachments
        attachment_src = media_dir / "attachments"
        if attachment_src.exists():
            attachment_dest = self.data_dir / "images" / "attachments"
            attachment_dest.mkdir(parents=True, exist_ok=True)
            for root, _, files in os.walk(attachment_src):
                root_path = Path(root)
                for file_name in files:
                    src_file = root_path / file_name
                    dest_file = attachment_dest / file_name
                    shutil.copy2(src_file, dest_file)
                    file_count += 1

        # Restore uploaded documents
        document_src = media_dir / "documents"
        if document_src.exists():
            documents_dest = self.data_dir / "documents"
            documents_dest.mkdir(parents=True, exist_ok=True)
            file_count += self._copy_directory_tree(document_src, documents_dest)
        
        logger.info(f"Restored {file_count} media files")
        return file_count
    
    def _restore_workflows(self, temp_dir: Path, character_id: str) -> int:
        """
        Restore workflow files to workflows directory.
        
        Returns:
            Number of workflow files restored
        """
        workflows_src = temp_dir / "workflows"
        
        if not workflows_src.exists():
            logger.info("No workflow files in backup")
            return 0
        
        # Create canonical character workflow directory
        workflows_dest = self.workflows_dir / character_id
        workflows_dest.mkdir(parents=True, exist_ok=True)

        # v2 stores workflows/{character_id}/{type}; v1 stored workflows/{type}
        canonical_src = workflows_src / character_id
        if canonical_src.exists():
            file_count = self._copy_directory_tree(canonical_src, workflows_dest)
        else:
            file_count = self._copy_directory_tree(workflows_src, workflows_dest)
        
        logger.info(f"Restored {file_count} workflow files")
        return file_count

    def _remap_conversation_media_path(self, path_text: Optional[str], old_conv_id: Optional[str], new_conv_id: Optional[str]) -> Optional[str]:
        if not path_text or not old_conv_id or not new_conv_id:
            return path_text
        marker = f"/{old_conv_id}/"
        return path_text.replace(marker, f"/{new_conv_id}/").replace(f"\\{old_conv_id}\\", f"\\{new_conv_id}\\")

    def _remap_chunk_id_csv(self, chunk_ids_text: Optional[str], chunk_id_map: Dict[str, str]) -> Optional[str]:
        if not chunk_ids_text:
            return chunk_ids_text
        parts = [part.strip() for part in chunk_ids_text.split(",") if part.strip()]
        mapped = [chunk_id_map.get(part, part) for part in parts]
        return ",".join(mapped)

    def _normalize_restored_attachment_path(self, source_path: Optional[str], processed: bool) -> Optional[str]:
        if not source_path:
            return None
        suffix = Path(source_path).suffix
        stem = Path(source_path).stem
        folder = self.data_dir / "images" / "attachments"
        folder.mkdir(parents=True, exist_ok=True)
        if processed and not stem.startswith("processed_"):
            stem = f"processed_{stem}"
        return str(folder / f"{stem}{suffix}")

    def _rebuild_vectors(self, character_id: str) -> Dict[str, Any]:
        vector_store = VectorStore(persist_directory=self.data_dir / "vector_store")
        summary_store = ConversationSummaryVectorStore(self.data_dir / "vector_store")
        pin_store = MomentPinVectorStore(self.data_dir / "vector_store")
        embedding_service = EmbeddingService()

        memory_stats = {
            "synced": 0,
            "deleted_orphans": 0,
            "errors": 0,
        }
        summary_stats = {
            "synced": 0,
            "deleted_orphans": 0,
            "errors": 0,
        }
        pin_stats = {
            "synced": 0,
            "deleted_orphans": 0,
            "errors": 0,
        }

        try:
            memory_stats_raw = self._run_async_sync(sync_memory_vectors(
                db_session=self.db,
                vector_store=vector_store,
                embedding_service=embedding_service,
                character_id=character_id
            ))
            memory_stats.update({
                "synced": memory_stats_raw.get("synced", 0),
                "deleted_orphans": memory_stats_raw.get("deleted_orphans", 0),
                "errors": memory_stats_raw.get("errors", 0),
            })
        except Exception as e:
            logger.warning(f"Memory vector rebuild failed for {character_id}: {e}")
            memory_stats["errors"] += 1

        try:
            summary_stats_raw = self._run_async_sync(sync_conversation_summary_vectors(
                db_session=self.db,
                summary_vector_store=summary_store,
                embedding_service=embedding_service,
                character_id=character_id
            ))
            summary_stats.update({
                "synced": summary_stats_raw.get("synced", 0),
                "deleted_orphans": summary_stats_raw.get("deleted_orphans", 0),
                "errors": summary_stats_raw.get("errors", 0),
            })
        except Exception as e:
            logger.warning(f"Summary vector rebuild failed for {character_id}: {e}")
            summary_stats["errors"] += 1

        try:
            pin_stats_raw = self._run_async_sync(sync_moment_pin_vectors(
                db_session=self.db,
                moment_pin_vector_store=pin_store,
                embedding_service=embedding_service,
                character_id=character_id
            ))
            pin_stats.update({
                "synced": pin_stats_raw.get("synced", 0),
                "deleted_orphans": pin_stats_raw.get("deleted_orphans", 0),
                "errors": pin_stats_raw.get("errors", 0),
            })
        except Exception as e:
            logger.warning(f"Moment pin vector rebuild failed for {character_id}: {e}")
            pin_stats["errors"] += 1

        return {
            "memory_vectors": memory_stats,
            "summary_vectors": summary_stats,
            "moment_pin_vectors": pin_stats,
        }

    def _run_async_sync(self, coro):
        import asyncio
        import threading
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # Run coroutine on a dedicated thread/loop when caller is already inside an event loop.
            result_box: Dict[str, Any] = {}
            error_box: Dict[str, Exception] = {}

            def runner():
                thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)
                try:
                    result_box["value"] = thread_loop.run_until_complete(coro)
                except Exception as e:  # pragma: no cover - passthrough
                    error_box["error"] = e
                finally:
                    thread_loop.close()

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            thread.join()
            if "error" in error_box:
                raise error_box["error"]
            return result_box.get("value")
        return loop.run_until_complete(coro)
    
    def _copy_directory_tree(self, src_dir: Path, dest_dir: Path) -> int:
        """
        Copy entire directory tree from src to dest.
        
        Returns:
            Number of files copied
        """
        file_count = 0
        
        for root, dirs, files in os.walk(src_dir):
            root_path = Path(root)
            
            # Calculate relative path from source
            rel_path = root_path.relative_to(src_dir)
            dest_path = dest_dir / rel_path
            
            # Create destination directory
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in files:
                src_file = root_path / file
                dest_file = dest_path / file
                shutil.copy2(src_file, dest_file)
                file_count += 1
        
        return file_count
