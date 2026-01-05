"""
Nova Setup Script
=================

This script configures Nova with all supported features:
- Profile picture
- Voice sample for TTS (Chatterbox)
- Default image generation workflow

Run this after initial installation to see a fully-featured character in action.
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path to import chorus_engine modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chorus_engine.models.conversation import VoiceSample
from chorus_engine.models.workflow import Workflow
from chorus_engine.repositories.voice_sample_repository import VoiceSampleRepository
from chorus_engine.repositories.workflow_repository import WorkflowRepository


class NovaSetup:
    """Handles Nova character setup."""
    
    def __init__(self):
        """Initialize setup paths."""
        self.addon_dir = Path(__file__).parent
        self.engine_root = self.addon_dir.parent.parent
        self.files_dir = self.addon_dir / "files"
        
        # Source files in addon
        self.profile_image_src = self.files_dir / "nova_profile.png"
        self.voice_sample_src = self.files_dir / "nova_voice_sample.mp3"
        self.workflow_src = self.files_dir / "nova_default_workflow.json"
        
        # Destination paths
        self.character_yaml = self.engine_root / "characters" / "nova.yaml"
        self.profile_image_dest = self.engine_root / "data" / "character_images" / "nova_profile.png"
        self.voice_sample_dir = self.engine_root / "data" / "voice_samples" / "nova"
        self.voice_sample_dest = self.voice_sample_dir / "nova_voice_sample.mp3"
        self.workflow_dir = self.engine_root / "workflows" / "nova" / "image"
        self.workflow_dest = self.workflow_dir / "Nova Default.json"
        
        # Database
        self.db_path = self.engine_root / "data" / "chorus.db"
        
        # Voice sample transcript
        self.voice_transcript = (
            "I've been thinking about how creativity works, you know? It's like... "
            "sometimes the best ideas come when you're not even trying. When you're "
            "just letting your mind wander through possibilities."
        )
        
    def verify_files(self):
        """Verify all required files exist."""
        print("Checking required files...")
        
        missing = []
        if not self.profile_image_src.exists():
            missing.append(str(self.profile_image_src))
        if not self.voice_sample_src.exists():
            missing.append(str(self.voice_sample_src))
        if not self.workflow_src.exists():
            missing.append(str(self.workflow_src))
        if not self.character_yaml.exists():
            missing.append(str(self.character_yaml))
        
        if missing:
            print("❌ Missing required files:")
            for f in missing:
                print(f"  - {f}")
            return False
        
        print("✓ All required files found")
        return True
    
    def setup_profile_image(self):
        """Copy profile image and update YAML."""
        print("\n📸 Setting up profile image...")
        
        # Ensure destination directory exists
        self.profile_image_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        shutil.copy2(self.profile_image_src, self.profile_image_dest)
        print(f"  ✓ Copied profile image to {self.profile_image_dest}")
        
        # Update YAML
        with open(self.character_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['profile_image'] = 'nova_profile.png'
        
        with open(self.character_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"  ✓ Updated nova.yaml with profile_image")
    
    def setup_voice_sample(self):
        """Copy voice sample and create database entry."""
        print("\n🎤 Setting up voice sample...")
        
        # Ensure destination directory exists
        self.voice_sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy voice sample
        shutil.copy2(self.voice_sample_src, self.voice_sample_dest)
        print(f"  ✓ Copied voice sample to {self.voice_sample_dest}")
        
        # Create database entry
        if not self.db_path.exists():
            print("  ⚠ Database not found. Voice sample will be registered on first run.")
            return
        
        try:
            engine = create_engine(f"sqlite:///{self.db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()
            
            repo = VoiceSampleRepository(session)
            
            # Check if sample already exists
            existing = session.query(VoiceSample).filter(
                VoiceSample.character_id == 'nova',
                VoiceSample.filename == 'nova_voice_sample.mp3'
            ).first()
            
            if existing:
                print("  ✓ Voice sample already registered in database")
            else:
                repo.create(
                    character_id='nova',
                    filename='nova_voice_sample.mp3',
                    transcript=self.voice_transcript,
                    is_default=True
                )
                print("  ✓ Registered voice sample in database")
            
            session.close()
        except Exception as e:
            print(f"  ⚠ Could not register voice sample in database: {e}")
            print("  ℹ Voice sample will be registered on first run")
    
    def setup_tts_config(self):
        """Ensure TTS is configured in YAML."""
        print("\n🔊 Configuring TTS settings...")
        
        with open(self.character_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Ensure voice config exists
        if 'voice' not in config:
            config['voice'] = {}
        
        config['voice']['enabled'] = True
        config['voice']['always_on'] = False
        if 'tts_provider' not in config['voice']:
            config['voice']['tts_provider'] = {}
        config['voice']['tts_provider']['provider'] = 'chatterbox'
        
        with open(self.character_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print("  ✓ TTS configured to use Chatterbox")
    
    def setup_image_generation_config(self):
        """Ensure image generation is enabled in YAML."""
        print("\n🖼️ Configuring image generation...")
        
        with open(self.character_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Ensure image generation is enabled
        if 'image_generation' not in config:
            config['image_generation'] = {}
        config['image_generation']['enabled'] = True
        
        with open(self.character_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print("  ✓ Image generation enabled")
    
    def run(self):
        """Run complete setup."""
        print("=" * 60)
        print("Nova Character Setup")
        print("=" * 60)
        print("\nThis will configure Nova with:")
        print("  • Profile picture")
        print("  • Voice sample for TTS (Chatterbox)")
        print("  • Image generation enabled (workflow setup required separately)")
        print()
        
        if not self.verify_files():
            print("\n❌ Setup failed: Missing required files")
            return False
        
        try:
            self.setup_profile_image()
            self.setup_voice_sample()
            self.setup_tts_config()
            self.setup_image_generation_config()
            
            print("\n" + "=" * 60)
            print("✅ Nova setup complete!")
            print("=" * 60)
            print("\nNova is now configured with:")
            print("  ✓ Profile picture")
            print("  ✓ Voice sample for TTS cloning")
            print("  ✓ Image generation enabled")
            print("\n📋 Next Steps:")
            print("  1. Configure ComfyUI workflow (see README for details):")
            print("     - Copy Nova's LoRA from files/ to ComfyUI (optional)")
            print("     - Edit files/nova_default_workflow.json for your setup")
            print("     - Test workflow in ComfyUI")
            print("     - Copy to workflows/nova/image/ and register via UI")
            print("  2. Start Chorus Engine: python -m chorus_engine.main")
            print("  3. Open the web interface")
            print("  4. Select Nova and start chatting")
            print("  5. Enable TTS to hear Nova's voice")
            print("  6. Use '@img' commands once workflow is configured")
            print()
            return True
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    setup = NovaSetup()
    success = setup.run()
    sys.exit(0 if success else 1)
