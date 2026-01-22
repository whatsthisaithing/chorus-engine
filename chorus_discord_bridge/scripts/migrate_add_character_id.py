"""
Apply database migration to add character_id column to conversation_mappings.

This migration allows multiple bots to operate in the same Discord channel
with separate conversations per bot.
"""
import sys
from pathlib import Path

# Add parent directory to path to import bridge modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chorus_discord_bridge.bridge.database import Database
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Apply migration to all bot databases."""
    # List of bot directories to migrate
    bot_dirs = [
        Path(__file__).parent.parent.parent / "bots" / "marcusbot",
        Path(__file__).parent.parent.parent / "bots" / "novabot",
    ]
    
    for bot_dir in bot_dirs:
        db_path = bot_dir / "storage" / "state.db"
        
        if not db_path.exists():
            logger.warning(f"Skipping {bot_dir.name}: database not found at {db_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Migrating {bot_dir.name}: {db_path}")
        logger.info(f"{'='*60}")
        
        db = Database(str(db_path))
        
        current_version = db.get_schema_version()
        logger.info(f"Current schema version: {current_version}")
        
        if current_version < 2:
            logger.info("Applying migration to add character_id column...")
            if db.migrate(target_version=2):
                logger.info(f"✓ {bot_dir.name} migrated successfully to v2")
            else:
                logger.error(f"✗ {bot_dir.name} migration failed")
        else:
            logger.info(f"✓ {bot_dir.name} already at v{current_version}")
        
        db.close()
    
    logger.info(f"\n{'='*60}")
    logger.info("Migration complete!")
    logger.info("Please restart both bots for changes to take effect.")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
