"""Quick test to verify the latest memory is retrievable."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Memory
from chorus_engine.services.memory_retrieval import MemoryRetrievalService
from chorus_engine.services.embedding_service import EmbeddingService
from chorus_engine.db.vector_store import VectorStore

db = SessionLocal()

# Get latest nova memory
latest = db.query(Memory).filter(Memory.character_id=='nova').order_by(Memory.created_at.desc()).first()
print(f"Latest memory: {latest.content}")
print(f"  ID: {latest.id}")
print(f"  vector_id: {latest.vector_id}")
print(f"  created: {latest.created_at}")
print()

# Initialize retrieval service
print("Initializing retrieval service...")
embedding_service = EmbeddingService()
vector_store = VectorStore(Path("data/vector_store"))
retrieval_service = MemoryRetrievalService(db, vector_store, embedding_service)

# Try to retrieve with query "What's my name?"
print("Testing retrieval with query: 'What's my name?'")
memories = retrieval_service.retrieve_memories(
    character_id="nova",
    query="What's my name?",
    conversation_id=latest.conversation_id,
    conversation_source="web"
)

print(f"\nRetrieved {len(memories)} memories:")
for mem in memories:
    print(f"  - {mem.memory.content[:60]} (similarity: {mem.similarity:.3f})")

if any(m.memory.id == latest.id for m in memories):
    print("\n✓ Latest memory IS retrievable!")
else:
    print("\n✗ Latest memory NOT in results")

db.close()
