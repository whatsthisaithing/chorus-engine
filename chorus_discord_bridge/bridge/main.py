"""Main entry point for Chorus Discord Bridge."""

import sys
import logging
import asyncio
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge.config import BridgeConfig, ConfigError, BotConfig
from bridge.bot import ChorusBot
from bridge.chorus_client import ChorusClient


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
    print("Chorus Discord Bridge v0.2.0 - Multi-Bot Support")
    print("=" * 60)
    
    try:
        # Load configuration (resolve path relative to this file)
        print("Loading configuration...")
        config_path = Path(__file__).parent.parent / "config.yaml"
        config = BridgeConfig(str(config_path))
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Chorus Discord Bridge Starting - Multi-Bot Mode")
        logger.info("=" * 60)
        logger.info(f"Chorus API: {config.chorus_api_url}")
        logger.info(f"Log Level: {config.bridge_log_level}")
        
        # Get enabled bots
        enabled_bots = [bot for bot in config.bots if bot.enabled]
        
        if not enabled_bots:
            logger.error("No enabled bots found in configuration!")
            print("\n No enabled bots configured. Check config.yaml")
            sys.exit(1)
        
        logger.info(f"Found {len(enabled_bots)} enabled bot(s):")
        for bot_config in enabled_bots:
            logger.info(f"  - {bot_config.character_id}")
        
        # Create shared Chorus client (more efficient than per-bot)
        logger.info("Creating shared Chorus API client...")
        shared_chorus_client = ChorusClient(
            api_url=config.chorus_api_url,
            api_key=config.chorus_api_key,
            timeout=config.chorus_timeout,
            retry_attempts=config.chorus_retry_attempts,
            retry_delay=config.chorus_retry_delay
        )
        
        # Verify Chorus Engine connection
        if not shared_chorus_client.health_check():
            logger.warning("Cannot connect to Chorus Engine API!")
            logger.warning(f"Attempted URL: {config.chorus_api_url}")
            print(f"\n Cannot connect to Chorus Engine at {config.chorus_api_url}")
            print("  Make sure Chorus Engine is running and CHORUS_API_URL is correct in .env")
            sys.exit(1)
        else:
            logger.info(" Connected to Chorus Engine")
            print(f" Connected to Chorus Engine: {config.chorus_api_url}")
        
        # Create bot instances
        print(f"\nInitializing {len(enabled_bots)} bot(s)...")
        bots = []
        for bot_config in enabled_bots:
            logger.info(f"Initializing bot for character '''{bot_config.character_id}''...")
            print(f"  - {bot_config.character_id}")
            
            bot = ChorusBot(
                config=config,
                character_id=bot_config.character_id,
                chorus_client=shared_chorus_client
            )
            bots.append((bot, bot_config.bot_token))
        
        print(f"\n All bots initialized")
        print("\nConnecting to Discord...")
        logger.info("Starting all bots...")
        print("(Press Ctrl+C to stop)")
        print()
        
        # Run all bots concurrently
        async def run_all_bots():
            tasks = []
            for bot, token in bots:
                # Each bot.start() is a coroutine that runs the bot
                tasks.append(bot.start(token))
            
            # Run all bots concurrently
            await asyncio.gather(*tasks)
        
        # Run the event loop
        asyncio.run(run_all_bots())
        
    except ConfigError as e:
        print(f"\n Configuration Error:\n{e}")
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
        print(f"\n Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

