"""Check if recent memories have vector_id set."""
from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory

db = SessionLocal()
recent = db.query(Memory).order_by(Memory.created_at.desc()).limit(5).all()

print("Recent memories:")
for m in recent:
    vector_id_display = m.vector_id[:8] + "..." if m.vector_id else "NONE"
    print(f"  ID: {m.id[:8]}... vector_id: {vector_id_display} status: {m.status} content: {m.content[:40]}")

db.close()
