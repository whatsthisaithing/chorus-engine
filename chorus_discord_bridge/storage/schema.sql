-- Discord Bridge State Database Schema
-- Version: 1.0
-- Created: January 20, 2026

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Insert initial version
INSERT OR IGNORE INTO schema_version (version, description) 
VALUES (1, 'Initial schema: conversation_mappings and discord_users');

-- Conversation Mappings: Discord channel/DM -> Chorus conversation
CREATE TABLE IF NOT EXISTS conversation_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discord_channel_id TEXT NOT NULL UNIQUE,  -- Discord channel ID or DM user ID
    discord_guild_id TEXT,                     -- Discord server ID (NULL for DMs)
    chorus_conversation_id TEXT NOT NULL,      -- Chorus Engine conversation UUID
    chorus_thread_id INTEGER NOT NULL,         -- Chorus Engine thread ID
    is_dm INTEGER NOT NULL DEFAULT 0,          -- 1 if DM, 0 if channel (SQLite boolean)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_discord_channel ON conversation_mappings(discord_channel_id);
CREATE INDEX IF NOT EXISTS idx_chorus_conversation ON conversation_mappings(chorus_conversation_id);
CREATE INDEX IF NOT EXISTS idx_last_message ON conversation_mappings(last_message_at DESC);

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
