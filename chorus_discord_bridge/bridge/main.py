"""Main entry point for Chorus Discord Bridge."""

import sys
import logging
import asyncio
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge.config import BridgeConfig, ConfigError
from bridge.bot import ChorusBot


def setup_logging(config: BridgeConfig):
    """
    Setup logging configuration.
    
    Args:
        config: Bridge configuration
    """
    # Create logs directory if it doesn't exist
    log_file = Path(config.bridge_log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.bridge_log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.bridge_log_rotation:
        file_handler = RotatingFileHandler(
            config.bridge_log_file,
            maxBytes=config.bridge_log_max_bytes,
            backupCount=config.bridge_log_backup_count,
            encoding='utf-8'
        )
    else:
        file_handler = logging.FileHandler(
            config.bridge_log_file,
            encoding='utf-8'
        )
    
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from discord.py
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('discord.http').setLevel(logging.WARNING)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Chorus Discord Bridge v0.1.0")
    print("=" * 60)
    
    try:
        # Load configuration (resolve path relative to this file)
        print("Loading configuration...")
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = BridgeConfig(str(config_path))
        print(f"✓ Configuration loaded: {config}")
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Chorus Discord Bridge Starting")
        logger.info("=" * 60)
        logger.info(f"Character: {config.chorus_character_id}")
        logger.info(f"Chorus API: {config.chorus_api_url}")
        logger.info(f"Log Level: {config.bridge_log_level}")
        
        # Create and run bot
        print("\nStarting Discord bot...")
        logger.info("Initializing Discord bot...")
        
        bot = ChorusBot(config)
        
        # Run the bot
        logger.info("Connecting to Discord...")
        print("Connecting to Discord...")
        print("(Press Ctrl+C to stop)")
        print()
        
        bot.run(config.discord_bot_token, log_handler=None)
        
    except ConfigError as e:
        print(f"\n❌ Configuration Error:\n{e}")
        print("\nPlease check your configuration files:")
        print("  - config.yaml (copy from config.yaml.template)")
        print("  - .env (copy from .env.template)")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        logger.info("Received shutdown signal")
        sys.exit(0)
    
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n❌ Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
