"""Debug memory retrieval to see what's happening."""
from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory, Conversation
from chorus_engine.db.vector_store import VectorStore
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.services.memory_retrieval import MemoryRetrievalService
from pathlib import Path

db = SessionLocal()

# Get the conversation from the log
conv = db.query(Conversation).filter(Conversation.id == '428b36e8-988a-4fbe-a32e-3cccfaa83399').first()
if conv:
    print(f"Conversation found: {conv.id}")
    print(f"  Source: {conv.source}")
    print(f"  Character: {conv.character_id}")
else:
    print("Conversation not found!")
    exit(1)

# Get recent name memories
name_memories = db.query(Memory).filter(Memory.content.like('%name is%')).order_by(Memory.created_at.desc()).limit(3).all()
print(f"\nFound {len(name_memories)} name memories:")
for m in name_memories:
    print(f"  ID: {m.id[:8]}... source: {m.source} vector_id: {m.vector_id[:8] if m.vector_id else 'NONE'} status: {m.status}")
    print(f"    Content: {m.content}")

# Try to retrieve memories
print("\n" + "="*80)
print("Testing memory retrieval...")
print("="*80)

vector_store = VectorStore(persist_directory=Path("data/vector_store"))
embedding_service = EmbeddingService()
memory_service = MemoryRetrievalService(db, vector_store, embedding_service)

# Test retrieval with the query "What's my name?"
query = "What's my name?"
print(f"\nQuery: '{query}'")
print(f"Conversation source: {conv.source}")

retrieved = memory_service.retrieve_memories(
    query=query,
    character_id='nova',
    conversation_source=conv.source,
    max_memories=10
)

print(f"\nRetrieved {len(retrieved)} memories:")
for mem in retrieved:
    print(f"  • [{mem.memory.source}] {mem.memory.content[:60]} (similarity: {mem.similarity:.3f})")

db.close()
