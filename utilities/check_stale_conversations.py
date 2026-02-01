#!/usr/bin/env python3
"""
Diagnostic script to check which conversations are eligible for background analysis.

This helps debug the heartbeat conversation analysis system by showing:
1. All conversations in the database
2. Which ones meet the stale threshold criteria
3. Why conversations are being skipped
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Conversation, Thread, Message
from chorus_engine.config.loader import ConfigLoader


def format_timedelta(td: timedelta) -> str:
    """Format timedelta as human-readable string."""
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def check_conversations(
    stale_hours: float = 24.0,
    min_messages: int = 10,
    show_all: bool = False
):
    """
    Check all conversations against the stale threshold criteria.
    
    Args:
        stale_hours: Hours of inactivity to consider stale
        min_messages: Minimum message count required
        show_all: Show all conversations, not just eligible ones
    """
    db = SessionLocal()
    
    try:
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=stale_hours)
        
        print(f"\n{'='*80}")
        print("CONVERSATION ANALYSIS ELIGIBILITY CHECK")
        print(f"{'='*80}")
        print(f"Current time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stale threshold: {stale_hours} hours")
        print(f"Cutoff time: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Minimum messages: {min_messages}")
        print(f"{'='*80}\n")
        
        # Get all conversations
        conversations = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
        
        eligible = []
        ineligible_reasons = {
            "private": [],
            "too_few_messages": [],
            "too_recent": [],
            "recently_analyzed": [],
        }
        
        for conv in conversations:
            # Count messages across all threads
            message_count = 0
            thread_ids = []
            for thread in conv.threads:
                thread_ids.append(thread.id)
                count = db.query(Message).filter(Message.thread_id == thread.id).count()
                message_count += count
            
            # Calculate time since last activity
            time_since_activity = now - conv.updated_at if conv.updated_at else timedelta(days=999)
            
            # Calculate time since last analysis
            time_since_analysis = None
            if conv.last_analyzed_at:
                time_since_analysis = now - conv.last_analyzed_at
            
            # Check eligibility
            is_private = conv.is_private == "true"
            has_enough_messages = message_count >= min_messages
            is_stale = conv.updated_at and conv.updated_at < cutoff_time
            needs_analysis = conv.last_analyzed_at is None or conv.last_analyzed_at < cutoff_time
            
            conv_info = {
                "id": conv.id,
                "title": conv.title or "(untitled)",
                "character_id": conv.character_id,
                "message_count": message_count,
                "updated_at": conv.updated_at,
                "last_analyzed_at": conv.last_analyzed_at,
                "time_since_activity": time_since_activity,
                "time_since_analysis": time_since_analysis,
                "is_private": is_private,
            }
            
            if is_private:
                ineligible_reasons["private"].append(conv_info)
            elif not has_enough_messages:
                ineligible_reasons["too_few_messages"].append(conv_info)
            elif not is_stale:
                ineligible_reasons["too_recent"].append(conv_info)
            elif not needs_analysis:
                ineligible_reasons["recently_analyzed"].append(conv_info)
            else:
                eligible.append(conv_info)
        
        # Print eligible conversations
        print(f"✅ ELIGIBLE FOR ANALYSIS ({len(eligible)} conversations)")
        print("-" * 80)
        if eligible:
            for conv in eligible:
                analysis_status = "Never analyzed" if not conv["last_analyzed_at"] else f"Analyzed {format_timedelta(conv['time_since_analysis'])} ago"
                print(f"  {conv['id'][:8]}... | {conv['character_id']:15} | {conv['message_count']:4} msgs | "
                      f"Activity: {format_timedelta(conv['time_since_activity'])} ago | {analysis_status}")
                print(f"    Title: {conv['title'][:60]}")
        else:
            print("  (none)")
        
        # Print ineligible by reason
        if show_all or not eligible:
            print(f"\n❌ PRIVATE CONVERSATIONS ({len(ineligible_reasons['private'])})")
            print("-" * 80)
            for conv in ineligible_reasons["private"][:5]:
                print(f"  {conv['id'][:8]}... | {conv['character_id']:15} | {conv['message_count']:4} msgs")
            if len(ineligible_reasons["private"]) > 5:
                print(f"  ... and {len(ineligible_reasons['private']) - 5} more")
            
            print(f"\n⚠️  TOO FEW MESSAGES (need {min_messages}+) ({len(ineligible_reasons['too_few_messages'])})")
            print("-" * 80)
            for conv in ineligible_reasons["too_few_messages"][:5]:
                print(f"  {conv['id'][:8]}... | {conv['character_id']:15} | {conv['message_count']:4} msgs | {conv['title'][:40]}")
            if len(ineligible_reasons["too_few_messages"]) > 5:
                print(f"  ... and {len(ineligible_reasons['too_few_messages']) - 5} more")
            
            print(f"\n⏰ TOO RECENT (activity within {stale_hours}h) ({len(ineligible_reasons['too_recent'])})")
            print("-" * 80)
            for conv in ineligible_reasons["too_recent"][:5]:
                print(f"  {conv['id'][:8]}... | {conv['character_id']:15} | {conv['message_count']:4} msgs | "
                      f"Activity: {format_timedelta(conv['time_since_activity'])} ago")
            if len(ineligible_reasons["too_recent"]) > 5:
                print(f"  ... and {len(ineligible_reasons['too_recent']) - 5} more")
            
            print(f"\n🔄 RECENTLY ANALYZED (within {stale_hours}h) ({len(ineligible_reasons['recently_analyzed'])})")
            print("-" * 80)
            for conv in ineligible_reasons["recently_analyzed"][:5]:
                print(f"  {conv['id'][:8]}... | {conv['character_id']:15} | {conv['message_count']:4} msgs | "
                      f"Analyzed: {format_timedelta(conv['time_since_analysis'])} ago")
            if len(ineligible_reasons["recently_analyzed"]) > 5:
                print(f"  ... and {len(ineligible_reasons['recently_analyzed']) - 5} more")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total conversations: {len(conversations)}")
        print(f"Eligible for analysis: {len(eligible)}")
        print(f"Private (skipped): {len(ineligible_reasons['private'])}")
        print(f"Too few messages: {len(ineligible_reasons['too_few_messages'])}")
        print(f"Too recent activity: {len(ineligible_reasons['too_recent'])}")
        print(f"Recently analyzed: {len(ineligible_reasons['recently_analyzed'])}")
        print()
        
    finally:
        db.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check which conversations are eligible for background analysis"
    )
    parser.add_argument(
        "--stale-hours", "-s",
        type=float,
        default=24.0,
        help="Hours of inactivity to consider stale (default: 24)"
    )
    parser.add_argument(
        "--min-messages", "-m",
        type=int,
        default=10,
        help="Minimum messages required (default: 10)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all conversations, including ineligible ones"
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Use thresholds from system.yaml heartbeat config"
    )
    
    args = parser.parse_args()
    
    stale_hours = args.stale_hours
    min_messages = args.min_messages
    
    if args.from_config:
        try:
            loader = ConfigLoader()
            config = loader.load_system_config()
            if hasattr(config, 'heartbeat') and config.heartbeat:
                stale_hours = config.heartbeat.analysis_stale_hours
                min_messages = config.heartbeat.analysis_min_messages
                print(f"Using config from system.yaml:")
                print(f"  stale_hours: {stale_hours}")
                print(f"  min_messages: {min_messages}")
        except Exception as e:
            print(f"Warning: Could not load system config: {e}")
            print("Using default values")
    
    check_conversations(
        stale_hours=stale_hours,
        min_messages=min_messages,
        show_all=args.all
    )


if __name__ == "__main__":
    main()
