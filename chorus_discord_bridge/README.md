# Chorus Discord Bridge

A standalone Discord bridge service that allows Chorus Engine characters to participate in Discord conversations.

## Features

- **@mention Support**: Characters respond when mentioned in Discord
- **Direct Messages**: Characters can handle DMs
- **Conversation Context**: Maintains full context within each channel
- **User Tracking**: Remembers users across messages
- **Hot-Swappable Characters**: Change active character without restarting
- **Rate Limiting**: Prevents spam and abuse
- **Graceful Error Handling**: Survives API failures and recovers automatically

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Running Chorus Engine instance
- Discord Bot Token (see [Creating a Discord Bot](#creating-a-discord-bot))

### Installation

**IMPORTANT**: The Discord Bridge uses the same Python environment as Chorus Engine to avoid dependency conflicts!

#### Windows

1. **Run the installation script** (from the bridge directory):
   ```bash
   install_bridge.bat
   ```
   This will install Discord dependencies into the embedded Python used by Chorus Engine.

2. **Edit configuration**:
   ```bash
   notepad .env
   # Add your Discord bot token
   ```

3. **Run the bridge**:
   ```bash
   start_bridge.bat
   ```

#### Linux/Mac

1. **Run the installation script** (from the bridge directory):
   ```bash
   chmod +x install_bridge.sh start_bridge.sh
   ./install_bridge.sh
   ```
   This will install Discord dependencies into the venv used by Chorus Engine.

2. **Edit configuration**:
   ```bash
   nano .env
   # Add your Discord bot token
   ```

3. **Run the bridge**:
   ```bash
   ./start_bridge.sh
   ```

## Creating a Discord Bot

See [scripts/create_bot.md](scripts/create_bot.md) for detailed instructions on creating a Discord bot and getting your token.

## Configuration

### Environment Variables (.env)

- `DISCORD_BOT_TOKEN`: Your Discord bot token (required)
- `CHORUS_API_URL`: Chorus Engine API URL (default: http://localhost:5000)
- `CHORUS_API_KEY`: Optional API key if Chorus requires authentication

### Application Settings (config.yaml)

- **discord.command_prefix**: Prefix for bot commands (default: "!")
- **discord.rate_limit**: Rate limiting settings
- **discord.max_history_fetch**: Number of messages to fetch for context (default: 10)
- **chorus.character_id**: Active character name (default: "nova")
- **chorus.timeout**: API request timeout in seconds (default: 30)
- **bridge.log_level**: Logging verbosity (default: "INFO")

## Usage

### In Discord Channels

Mention the bot to get a response:
```
@NovaBot what do you think about AI ethics?
```

### In Direct Messages

Just send a message directly to the bot:
```
Hello! Can you help me with something?
```

### Bot Commands

- `!status`: Show bot status and statistics
- `!character`: Show active character information
- `!reload`: Reload configuration (admin only)

## Architecture

The bridge is a standalone service that:
1. Connects to Discord via discord.py
2. Communicates with Chorus Engine via HTTP API
3. Maintains conversation state in SQLite database
4. Maps Discord channels to Chorus conversations
5. Tracks users and provides context to Chorus

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
chorus_discord_bridge/
├── bridge/           # Core bridge code
├── storage/          # State database and logs
├── tests/            # Unit and integration tests
├── scripts/          # Utility scripts
└── config.yaml       # Configuration file
```

## Troubleshooting

### Bot doesn't respond

1. Check that bot has proper permissions in Discord server
2. Verify `DISCORD_BOT_TOKEN` is correct in `.env`
3. Ensure Chorus Engine is running and accessible
4. Check logs in `storage/bridge.log`

### "Rate limit exceeded" errors

Adjust rate limit settings in `config.yaml`:
```yaml
discord:
  rate_limit:
    per_user_cooldown: 5  # Increase cooldown
    global_limit: 5       # Reduce global limit
```

### Connection to Chorus Engine fails

1. Verify `CHORUS_API_URL` in `.env` is correct
2. Test Chorus Engine directly: `curl http://localhost:5000/api/characters`
3. Check network connectivity and firewall settings

## License

See main Chorus Engine LICENSE.md

## Support

For issues and questions, see the main Chorus Engine documentation.
