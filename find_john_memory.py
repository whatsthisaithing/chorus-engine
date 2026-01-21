"""Find the John memory and its source."""
from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory, Message

db = SessionLocal()

# Find John memory
john_mem = db.query(Memory).filter(
    Memory.content == 'User name is John',
    Memory.id.like('ee756358%')
).first()

if john_mem:
    print(f"Memory ID: {john_mem.id}")
    print(f"Created: {john_mem.created_at}")
    print(f"Conversation: {john_mem.conversation_id}")
    print(f"Source messages: {john_mem.source_messages}")
    print(f"Vector ID: {john_mem.vector_id}")
    print(f"Status: {john_mem.status}")
    
    if john_mem.source_messages:
        msgs = db.query(Message).filter(Message.id.in_(john_mem.source_messages)).all()
        print(f"\nFound {len(msgs)} source messages:")
        for msg in msgs:
            print(f"  [{msg.role}] {msg.content}")
else:
    print("Memory not found!")

db.close()
