"""Quick script to check if metadata is being saved to messages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Message
from sqlalchemy import desc

db = SessionLocal()

# Get 5 most recent messages
recent_messages = db.query(Message).order_by(desc(Message.created_at)).limit(5).all()

print("\n=== 5 Most Recent Messages ===\n")
for msg in recent_messages:
    print(f"ID: {msg.id}")
    print(f"Thread: {msg.thread_id}")
    print(f"Role: {msg.role}")
    print(f"Content: {msg.content[:60]}...")
    print(f"Metadata: {msg.meta_data}")
    print(f"Created: {msg.created_at}")
    print("-" * 80)

db.close()
