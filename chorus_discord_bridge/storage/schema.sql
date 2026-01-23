-- Discord Bridge State Database Schema
-- Version: 2.0
-- Created: January 20, 2026
-- Updated: January 22, 2026 - Multi-bot support

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Insert initial version
INSERT OR IGNORE INTO schema_version (version, description) 
VALUES (1, 'Initial schema: conversation_mappings and discord_users');

-- Insert multi-bot version
INSERT OR IGNORE INTO schema_version (version, description) 
VALUES (2, 'Multi-bot support: Added character_id to conversation_mappings');

-- Conversation Mappings: Discord channel/DM -> Chorus conversation
CREATE TABLE IF NOT EXISTS conversation_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_channel_id TEXT NOT NULL,          -- Discord channel ID or DM user ID
    discord_guild_id TEXT,                     -- Discord server ID (NULL for DMs)
    character_id TEXT NOT NULL,                -- Character ID for this conversation
    chorus_conversation_id TEXT NOT NULL,      -- Chorus Engine conversation UUID
    chorus_thread_id INTEGER NOT NULL,         -- Chorus Engine thread ID
    is_dm INTEGER NOT NULL DEFAULT 0,          -- 1 if DM, 0 if channel (SQLite boolean)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    UNIQUE(discord_channel_id, character_id)   -- Allow same channel for different characters
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_discord_channel_only ON conversation_mappings(discord_channel_id);
CREATE INDEX IF NOT EXISTS idx_character ON conversation_mappings(character_id);
CREATE INDEX IF NOT EXISTS idx_chorus_conversation ON conversation_mappings(chorus_conversation_id);
CREATE INDEX IF NOT EXISTS idx_last_message ON conversation_mappings(last_message_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_discord_channel_character 
ON conversation_mappings(discord_channel_id, character_id);

-- Discord User Tracking
CREATE TABLE IF NOT EXISTS discord_users (
    discord_user_id TEXT PRIMARY KEY,          -- Discord user ID
    username TEXT NOT NULL,                    -- Current Discord username
    display_name TEXT,                         -- Current display name/nickname
    known_aliases TEXT,                        -- JSON array of known aliases
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    last_username_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for username lookups
CREATE INDEX IF NOT EXISTS idx_username ON discord_users(username);
CREATE INDEX IF NOT EXISTS idx_last_seen ON discord_users(last_seen DESC);
