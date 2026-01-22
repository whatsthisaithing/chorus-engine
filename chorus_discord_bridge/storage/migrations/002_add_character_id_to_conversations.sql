-- Migration: Add character_id to conversation_mappings
-- Version: 2
-- Date: January 21, 2026
-- Purpose: Support multiple bots in same channel (multi-bot per-channel conversations)

-- Add character_id column
ALTER TABLE conversation_mappings ADD COLUMN character_id TEXT;

-- Update schema version
INSERT INTO schema_version (version, description) 
VALUES (2, 'Add character_id to conversation_mappings for multi-bot support');

-- Drop old unique constraint on discord_channel_id
DROP INDEX IF EXISTS idx_discord_channel;

-- Create new composite index for (discord_channel_id, character_id)
CREATE UNIQUE INDEX IF NOT EXISTS idx_discord_channel_character 
ON conversation_mappings(discord_channel_id, character_id);

-- Keep other indexes
CREATE INDEX IF NOT EXISTS idx_discord_channel_only ON conversation_mappings(discord_channel_id);
