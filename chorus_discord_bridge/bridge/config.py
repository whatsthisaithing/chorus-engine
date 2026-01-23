"""Configuration management for Chorus Discord Bridge."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


@dataclass
class BotConfig:
    """Configuration for a single bot instance."""
    character_id: str
    bot_token: str
    enabled: bool = True
    
    def __repr__(self) -> str:
        """Return string representation (hiding token)."""
        return f"BotConfig(character='{self.character_id}', enabled={self.enabled})"


class BridgeConfig:
    """Main configuration class for the Discord bridge."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from file and environment."""
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from YAML file and environment variables."""
        # Load environment variables from .env file (look in same directory as config.yaml)
        env_path = self.config_path.parent / '.env'
        load_dotenv(env_path)
        
        # Load YAML configuration
        if not self.config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {self.config_path}\n"
                "Copy config.yaml.template to config.yaml and customize it."
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load sensitive values from environment variables."""
        # For backwards compatibility, check for single DISCORD_BOT_TOKEN
        default_token = os.getenv('DISCORD_BOT_TOKEN')
        
        # Load bot configurations
        if 'bots' in self._config and isinstance(self._config['bots'], list):
            # Multi-bot mode: Load token for each bot
            for bot_config in self._config['bots']:
                character_id = bot_config.get('character_id')
                token_env = bot_config.get('bot_token_env', f"DISCORD_BOT_TOKEN_{character_id.upper()}")
                
                # Try character-specific token first, fall back to default
                bot_token = os.getenv(token_env) or default_token
                
                if not bot_token:
                    raise ConfigError(
                        f"No Discord bot token found for character '{character_id}'.\n"
                        f"Set {token_env} or DISCORD_BOT_TOKEN in .env file."
                    )
                
                bot_config['bot_token'] = bot_token
        else:
            # Legacy single-bot mode: Convert to bots array
            if not default_token:
                raise ConfigError(
                    "DISCORD_BOT_TOKEN not found in environment.\n"
                    "Copy .env.template to .env and add your bot token."
                )
            
            character_id = self._config.get('chorus', {}).get('character_id', 'nova')
            
            if 'discord' not in self._config:
                self._config['discord'] = {}
            self._config['discord']['bot_token'] = default_token
            
            # Convert to bots array format
            self._config['bots'] = [{
                'character_id': character_id,
                'bot_token': default_token,
                'enabled': True
            }]
        
        # Chorus API URL (required)
        api_url = os.getenv('CHORUS_API_URL', 'http://localhost:5000')
        if 'chorus' not in self._config:
            self._config['chorus'] = {}
        self._config['chorus']['api_url'] = api_url
        
        # Chorus API Key (optional)
        api_key = os.getenv('CHORUS_API_KEY')
        if api_key:
            self._config['chorus']['api_key'] = api_key
    
    def _validate_config(self):
        """Validate that required configuration values are present."""
        # Validate bots configuration
        if 'bots' not in self._config or not self._config['bots']:
            raise ConfigError(
                "Missing 'bots' array in config or no bots configured.\n"
                "Add at least one bot to the 'bots' array."
            )
        
        # Validate each bot config
        for i, bot_config in enumerate(self._config['bots']):
            if 'character_id' not in bot_config:
                raise ConfigError(f"Bot #{i+1} missing 'character_id'")
            if 'bot_token' not in bot_config:
                raise ConfigError(f"Bot #{i+1} (character: {bot_config.get('character_id')}) missing bot_token")
        
        # Validate Chorus config
        if 'chorus' not in self._config:
            raise ConfigError("Missing 'chorus' section in config")
        
        if 'api_url' not in self._config['chorus']:
            raise ConfigError("Missing Chorus API URL")
        
        # Validate bridge config
        if 'bridge' not in self._config:
            self._config['bridge'] = {}
        
        # Set defaults for bridge config
        self._config['bridge'].setdefault('state_db_path', 'storage/state.db')
        self._config['bridge'].setdefault('log_level', 'INFO')
        self._config['bridge'].setdefault('log_file', 'storage/bridge.log')
    
    # Discord configuration properties
    @property
    def bots(self) -> List[BotConfig]:
        """Get list of bot configurations."""
        bot_configs = []
        for bot_data in self._config.get('bots', []):
            bot_configs.append(BotConfig(
                character_id=bot_data['character_id'],
                bot_token=bot_data['bot_token'],
                enabled=bot_data.get('enabled', True)
            ))
        return bot_configs
    
    @property
    def discord_bot_token(self) -> str:
        """Get Discord bot token."""
        return self._config['discord']['bot_token']
    
    @property
    def discord_command_prefix(self) -> str:
        """Get command prefix for bot commands."""
        return self._config['discord'].get('command_prefix', '!')
    
    @property
    def discord_per_user_cooldown(self) -> int:
        """Get per-user cooldown in seconds."""
        return self._config['discord'].get('rate_limit', {}).get('per_user_cooldown', 2)
    
    @property
    def discord_global_limit(self) -> int:
        """Get global message limit per minute."""
        return self._config['discord'].get('rate_limit', {}).get('global_limit', 10)
    
    @property
    def discord_max_history_fetch(self) -> int:
        """Get number of messages to fetch for context."""
        return self._config['discord'].get('max_history_fetch', 10)
    
    @property
    def discord_typing_timeout(self) -> int:
        """Get typing indicator timeout in seconds."""
        return self._config['discord'].get('typing_timeout', 10)
    
    # Chorus configuration properties
    @property
    def chorus_api_url(self) -> str:
        """Get Chorus Engine API URL."""
        return self._config['chorus']['api_url'].rstrip('/')
    
    @property
    def chorus_api_key(self) -> Optional[str]:
        """Get optional Chorus API key."""
        return self._config['chorus'].get('api_key')
    
    @property
    def chorus_character_id(self) -> str:
        """Get active character ID."""
        return self._config['chorus']['character_id']
    
    @property
    def chorus_timeout(self) -> int:
        """Get API request timeout in seconds."""
        return self._config['chorus'].get('timeout', 30)
    
    @property
    def chorus_retry_attempts(self) -> int:
        """Get number of retry attempts on failure."""
        return self._config['chorus'].get('retry_attempts', 3)
    
    @property
    def chorus_retry_delay(self) -> int:
        """Get seconds to wait between retries."""
        return self._config['chorus'].get('retry_delay', 2)
    
    # Bridge configuration properties
    @property
    def bridge_state_db_path(self) -> str:
        """Get state database path."""
        return self._config['bridge']['state_db_path']
    
    @property
    def bridge_log_level(self) -> str:
        """Get logging level."""
        return self._config['bridge']['log_level']
    
    @property
    def bridge_log_file(self) -> str:
        """Get log file path."""
        return self._config['bridge']['log_file']
    
    @property
    def bridge_log_rotation(self) -> bool:
        """Check if log rotation is enabled."""
        return self._config['bridge'].get('log_rotation', True)
    
    @property
    def bridge_log_max_bytes(self) -> int:
        """Get max log file size in bytes."""
        return self._config['bridge'].get('log_max_bytes', 10485760)
    
    @property
    def bridge_log_backup_count(self) -> int:
        """Get number of backup log files to keep."""
        return self._config['bridge'].get('log_backup_count', 5)
    
    @property
    def bridge_enable_typing_indicator(self) -> bool:
        """Check if typing indicator is enabled."""
        return self._config['bridge'].get('enable_typing_indicator', True)
    
    @property
    def bridge_enable_rate_limiting(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._config['bridge'].get('enable_rate_limiting', True)
    
    @property
    def bridge_enable_dm_support(self) -> bool:
        """Check if DM support is enabled."""
        return self._config['bridge'].get('enable_dm_support', True)
    
    @property
    def bridge_history_limit(self) -> int:
        """Get number of messages to fetch for history sync (Phase 3)."""
        return self._config['bridge'].get('history_limit', 10)
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
        self._validate_config()
    
    def __repr__(self) -> str:
        """Return string representation (without sensitive data)."""
        return (
            f"BridgeConfig("
            f"character='{self.chorus_character_id}', "
            f"api_url='{self.chorus_api_url}', "
            f"log_level='{self.bridge_log_level}'"
            f")"
        )
