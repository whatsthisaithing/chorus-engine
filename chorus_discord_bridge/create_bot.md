# Creating a Discord Bot

This guide walks you through creating a Discord bot application and getting your bot token.

## Step 1: Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Enter a name for your bot (e.g., "Chorus Nova Bot")
4. Click "Create"

## Step 2: Create Bot User

1. In your application, navigate to the "Bot" tab on the left sidebar
2. Click "Add Bot"
3. Confirm by clicking "Yes, do it!"
4. Your bot is now created

## Step 3: Get Bot Token

1. In the "Bot" tab, find the "TOKEN" section
2. Click "Reset Token" (or "Copy" if it's your first time)
3. **IMPORTANT**: Copy this token and save it securely
4. This token goes in your `.env` file as `DISCORD_BOT_TOKEN`
5. **NEVER share this token publicly or commit it to git**

## Step 4: Configure Bot Settings

### Privileged Gateway Intents

Enable these intents for the bot to work properly:

1. Scroll down to "Privileged Gateway Intents"
2. Enable:
   - ✅ **MESSAGE CONTENT INTENT** (required to read message content)
   - ✅ **SERVER MEMBERS INTENT** (optional, for user tracking)

### Bot Permissions

Calculate the permissions your bot needs:

1. Go to the "OAuth2" > "URL Generator" tab
2. Under "SCOPES", select:
   - ✅ `bot`
3. Under "BOT PERMISSIONS", select:
   - ✅ Send Messages
   - ✅ Read Messages/View Channels
   - ✅ Read Message History
   - ✅ Attach Files (for images)
   - ✅ Embed Links
   - ✅ Use External Emojis (optional)

## Step 5: Invite Bot to Server

1. Copy the generated URL from the "URL Generator" tab
2. Paste it in your browser
3. Select the server you want to add the bot to
4. Click "Authorize"
5. Complete the CAPTCHA if prompted

## Step 6: Configure Bridge

1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` and paste your bot token:
   ```
   DISCORD_BOT_TOKEN=your_actual_token_here
   CHORUS_API_URL=http://localhost:5000
   ```

3. Copy `config.yaml.template` to `config.yaml`:
   ```bash
   cp config.yaml.template config.yaml
   ```

4. Edit `config.yaml` and set your character:
   ```yaml
   chorus:
     character_id: "nova"  # Use your character's name
   ```

## Step 7: Test the Bot

1. Run the bridge:
   ```bash
   python -m bridge.main
   ```

2. In Discord, mention your bot:
   ```
   @YourBotName hello!
   ```

3. The bot should respond with a message from your Chorus character!

## Troubleshooting

### Bot appears offline
- Check that your bot token is correct in `.env`
- Ensure you've enabled MESSAGE CONTENT INTENT
- Verify Chorus Engine is running

### Bot doesn't respond to messages
- Make sure you @mention the bot
- Check bot has "Read Messages" and "Send Messages" permissions in the channel
- Look at logs in `storage/bridge.log`

### "Missing Permissions" error
- Go back to the OAuth2 URL Generator
- Ensure all required permissions are checked
- Re-invite the bot using the new URL

## Security Notes

- **Never** share your bot token
- **Never** commit `.env` to git (it's in `.gitignore`)
- Regenerate your token if it's ever exposed
- Use environment variables for secrets, not config files

## Next Steps

Once your bot is set up and responding:
1. Customize character settings in `config.yaml`
2. Adjust rate limiting as needed
3. Test in multiple channels and DMs
4. Monitor logs for any issues

For more help, see the main [README.md](../README.md).
