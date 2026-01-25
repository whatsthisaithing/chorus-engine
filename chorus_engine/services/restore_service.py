"""Character restore service for restoring character snapshots from backups.

This service restores complete character backups including:
- Character configuration (YAML)
- SQL database records (conversations, threads, messages, memories)
- Vector store data (ChromaDB embeddings)
- Media files (images, videos, audio, voice samples)
- Workflow files (ComfyUI workflows)

Supports restoration from .zip backup files created by backup_service.py
"""

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

from chorus_engine.models.conversation import (
    Conversation, Thread, Message, Memory, 
    GeneratedImage, GeneratedVideo, AudioMessage, VoiceSample,
    ConversationSummary
)
from chorus_engine.models.workflow import Workflow
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


class RestoreError(Exception):
    """Exception raised when character restoration fails."""
    pass


class CharacterRestoreService:
    """Service for restoring character backups."""
    
    SUPPORTED_BACKUP_VERSIONS = [1]  # Currently support version 1
    
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
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Restore a character from backup file.
        
        Args:
            backup_file: Path to .zip backup file
            new_character_id: Optional custom ID for restored character (takes precedence over rename_if_exists)
            rename_if_exists: If True, auto-rename character if ID exists (appends timestamp)
            overwrite: If True, overwrite existing character (requires confirmation)
        
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
            
            # Step 4: Check character collision
            character_id = manifest['character_id']
            original_id = character_id
            
            # If user specified a custom ID, use that
            if new_character_id:
                character_id = new_character_id
                logger.info(f"Using custom character ID: {character_id}")
            
            if self._character_exists(character_id):
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
            
            # Step 5: Restore character configuration
            logger.info("Restoring character configuration...")
            self._restore_character_config(temp_dir, character_id)
            
            # Step 6: Restore SQL data
            logger.info("Restoring database records...")
            restored_counts, id_maps = self._restore_sql_data(temp_dir, character_id, original_id)
            
            # Step 7: Restore vector store with remapped memory IDs
            logger.info("Restoring vector store...")
            vector_count = self._restore_vector_store(temp_dir, character_id, id_maps['memories'])
            restored_counts['vectors'] = vector_count
            
            # Step 8: Restore media files
            logger.info("Restoring media files...")
            media_count = self._restore_media_files(temp_dir, character_id)
            restored_counts['media_files'] = media_count
            
            # Step 9: Restore workflows
            logger.info("Restoring workflow files...")
            workflow_count = self._restore_workflows(temp_dir, character_id)
            restored_counts['workflow_files'] = workflow_count
            
            logger.info(f"Restore completed successfully for character: {character_id}")
            
            return {
                'character_id': character_id,
                'original_id': original_id,
                'renamed': character_id != original_id,
                'backup_date': manifest.get('backup_date'),
                'restored_counts': restored_counts
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
    
    def _character_exists(self, character_id: str) -> bool:
        """
        Check if character already exists.
        
        Only checks for YAML file since that's the authoritative source.
        Database orphans from incomplete deletions are ignored.
        """
        yaml_exists = (self.characters_dir / f"{character_id}.yaml").exists()
        return yaml_exists
    
    def _delete_existing_character(self, character_id: str):
        """Delete existing character data (for overwrite mode)."""
        # TODO: Implement full character deletion
        # This should delete:
        # - All conversations, threads, messages
        # - All memories
        # - All media records
        # - Vector store collection
        # - Media files
        # - Workflow files
        # - Character YAML
        logger.warning(f"Character deletion not yet fully implemented for {character_id}")
        raise NotImplementedError("Character overwrite not yet implemented")
    
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
            'memories': {}
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
            'voice_samples': 0
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
                    title_auto_generated=conv_data.get('title_auto_generated'),
                    last_analyzed_at=datetime.fromisoformat(conv_data['last_analyzed_at']) if conv_data.get('last_analyzed_at') else None,
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
                            preserve_full_text=msg_data.get('preserve_full_text', 'true')
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
                    vector_id=new_mem_id,  # Use new memory ID as vector ID
                    embedding_model=mem_data.get('embedding_model', 'all-MiniLM-L6-v2'),
                    priority=mem_data.get('priority', 50),
                    tags=mem_data.get('tags'),
                    confidence=mem_data.get('confidence'),
                    category=mem_data.get('category'),
                    status=mem_data.get('status', 'approved'),
                    source_messages=source_messages,
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
                    image = GeneratedImage(
                        # Let DB assign new ID (integer autoincrement)
                        conversation_id=new_conv_id,
                        thread_id=new_thread_id,
                        message_id=new_msg_id,
                        character_id=img_data['character_id'],
                        prompt=img_data['prompt'],
                        negative_prompt=img_data.get('negative_prompt'),
                        workflow_file=img_data['workflow_file'],
                        file_path=img_data['file_path'],
                        thumbnail_path=img_data.get('thumbnail_path'),
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
                    video = GeneratedVideo(
                        # Let DB assign new ID (integer autoincrement)
                        conversation_id=new_conv_id,
                        file_path=vid_data['file_path'],
                        thumbnail_path=vid_data.get('thumbnail_path'),
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
            
            # Restore audio messages
            for audio_data in media_data.get('audio_messages', []):
                # Remap message_id
                old_msg_id = audio_data['message_id']
                new_msg_id = id_maps['messages'].get(old_msg_id)
                
                if new_msg_id:  # Only add if ID mapped successfully
                    audio = AudioMessage(
                        # Let DB assign new ID (integer autoincrement)
                        message_id=new_msg_id,
                        audio_filename=audio_data['audio_filename'],
                        workflow_name=audio_data.get('workflow_name'),
                        generation_duration=audio_data.get('generation_duration'),
                        text_preprocessed=audio_data.get('text_preprocessed'),
                        voice_sample_id=audio_data.get('voice_sample_id'),
                        created_at=datetime.fromisoformat(audio_data['created_at'])
                    )
                    self.db.add(audio)
                    counts['audio'] += 1
            
            # Restore voice samples
            for sample_data in media_data.get('voice_samples', []):
                # Update character_id if renamed
                sample_data['character_id'] = character_id
                
                sample = VoiceSample(
                    # Let DB assign new ID (integer autoincrement)
                    character_id=sample_data['character_id'],
                    filename=sample_data['filename'],
                    transcript=sample_data['transcript'],
                    is_default=sample_data.get('is_default', 0),
                    uploaded_at=datetime.fromisoformat(sample_data['uploaded_at'])
                )
                self.db.add(sample)
                counts['voice_samples'] += 1
        
        # Commit all changes
        self.db.commit()
        
        logger.info(f"Restored SQL data: {counts}")
        return counts, id_maps
    
    def _restore_vector_store(self, temp_dir: Path, character_id: str, memory_id_map: Dict[str, str]) -> int:
        """
        Restore vector store from backup with remapped memory IDs.
        
        Args:
            temp_dir: Temporary directory with extracted backup
            character_id: New character ID
            memory_id_map: Mapping of old_memory_id -> new_memory_id
        
        Returns:
            Number of vectors restored
        """
        vectors_file = temp_dir / "vectors" / "chromadb_export.json"
        
        if not vectors_file.exists():
            logger.warning("No vector store data in backup")
            return 0
        
        try:
            with open(vectors_file, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)
            
            vectors = vector_data.get('vectors', [])
            
            if not vectors:
                logger.info("No vectors in backup")
                return 0
            
            # Get or create collection for character
            collection = self.vector_store.get_or_create_collection(character_id)
            
            # Prepare data for batch insert with remapped IDs
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for vec in vectors:
                old_id = vec['id']
                # Use remapped memory ID as vector ID
                new_id = memory_id_map.get(old_id, old_id)  # Fallback to old ID if not in map
                
                # Convert embedding list back to format ChromaDB expects
                # ChromaDB will handle conversion to numpy array internally
                embedding = vec.get('embedding')
                if embedding:
                    ids.append(new_id)
                    embeddings.append(embedding)
                    documents.append(vec.get('document', ''))
                    metadatas.append(vec.get('metadata', {}))
            
            # Add vectors to collection
            if ids:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Restored {len(ids)} vectors to collection {character_id}")
            return len(ids)
            
        except Exception as e:
            logger.error(f"Failed to restore vector store: {e}")
            # Continue without vectors - they can be regenerated
            return 0
    
    def _restore_media_files(self, temp_dir: Path, character_id: str) -> int:
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
        
        # Restore images
        images_src = media_dir / "images"
        if images_src.exists():
            images_dest = self.data_dir / "images"
            images_dest.mkdir(parents=True, exist_ok=True)
            file_count += self._copy_directory_tree(images_src, images_dest)
        
        # Restore videos
        videos_src = media_dir / "videos"
        if videos_src.exists():
            videos_dest = self.data_dir / "videos"
            videos_dest.mkdir(parents=True, exist_ok=True)
            file_count += self._copy_directory_tree(videos_src, videos_dest)
        
        # Restore audio
        audio_src = media_dir / "audio"
        if audio_src.exists():
            audio_dest = self.data_dir / "audio"
            audio_dest.mkdir(parents=True, exist_ok=True)
            file_count += self._copy_directory_tree(audio_src, audio_dest)
        
        # Restore voice samples
        voice_src = media_dir / "voice_samples"
        if voice_src.exists():
            voice_dest = self.data_dir / "voice_samples"
            voice_dest.mkdir(parents=True, exist_ok=True)
            file_count += self._copy_directory_tree(voice_src, voice_dest)
        
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
        
        # Create character workflow directory
        workflows_dest = self.workflows_dir / character_id
        workflows_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy all workflow files maintaining structure
        file_count = self._copy_directory_tree(workflows_src, workflows_dest)
        
        logger.info(f"Restored {file_count} workflow files")
        return file_count
    
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
