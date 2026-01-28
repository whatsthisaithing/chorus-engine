"""Main entry point for Chorus Discord Bridge."""

import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from bridge.config import BridgeConfig, ConfigError, BotConfig
from bridge.bot import ChorusBot
from bridge.chorus_client import ChorusClient


def setup_logging(config: BridgeConfig):
    """
    Setup logging configuration with timestamped log files.
    
    Args:
        config: Bridge configuration
    
    Returns:
        Path: The log file path that was created
    """
    # Create logs directory (use storage/logs instead of just storage)
    logs_dir = Path(__file__).parent.parent / "storage" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file (new file for each session)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"bridge_{timestamp}.log"
    
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
    
    # File handler (new file for each session)
    file_handler = logging.FileHandler(
        log_file,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from discord.py
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('discord.http').setLevel(logging.WARNING)
    
    return log_file


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
        
        # Setup logging with timestamped file
        log_file = setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Chorus Discord Bridge Starting - Multi-Bot Mode")
        logger.info("=" * 60)
        logger.info(f"Log file: {log_file}")
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
            # Verify Chorus Engine connection (do this in async context)
            logger.info("Verifying Chorus Engine connection...")
            if not await shared_chorus_client.health_check():
                logger.warning("Cannot connect to Chorus Engine API!")
                logger.warning(f"Attempted URL: {config.chorus_api_url}")
                print(f"\n Cannot connect to Chorus Engine at {config.chorus_api_url}")
                print("  Make sure Chorus Engine is running and CHORUS_API_URL is correct in .env")
                sys.exit(1)
            else:
                logger.info(" Connected to Chorus Engine")
                print(f" Connected to Chorus Engine: {config.chorus_api_url}\n")
            
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

