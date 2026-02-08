"""
Extract conversation analysis data as JSON.

Usage:
    python utilities/extract_conversation_summary.py <conversation_id>
    
Example:
    python utilities/extract_conversation_summary.py b4a106a3-4ff3-45dd-917c-9b3e6f0501a4
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chorus_engine.db.database import SessionLocal
from chorus_engine.models.conversation import Conversation, ConversationSummary, Memory
from chorus_engine.db.conversation_summary_vector_store import ConversationSummaryVectorStore


def extract_conversation_analysis(conversation_id: str):
    """Extract complete conversation analysis including memories."""
    
    # Initialize database
    db = SessionLocal()
    
    try:
        # Get conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            print(f"Conversation {conversation_id} not found")
            return None
        
        print(f"Found conversation: {conversation.title}")
        print(f"Character: {conversation.character_id}")
        
        # Get most recent summary
        summary = db.query(ConversationSummary).filter(
            ConversationSummary.conversation_id == conversation_id
        ).order_by(ConversationSummary.created_at.desc()).first()
        
        if not summary:
            print(f"No analysis found for this conversation")
            return None
        
        print(f"Analysis date: {summary.created_at}")
        print(f"Messages analyzed: {summary.message_count}")
        
        # Get extracted memories from this conversation
        memories = db.query(Memory).filter(
            Memory.conversation_id == conversation_id,
            Memory.status.in_(["approved", "auto_approved"])
        ).order_by(Memory.created_at.desc()).all()
        
        print(f"Extracted memories: {len(memories)}")
        
        # Get themes/key topics from vector store (for cross-checking legacy metadata)
        vector_store = ConversationSummaryVectorStore(
            persist_directory=Path("data/vector_store")
        )
        
        themes = []
        vector_key_topics = []
        try:
            collection = vector_store.get_collection(conversation.character_id)
            if collection:
                # Query by conversation_id to get metadata
                result = collection.get(
                    ids=[conversation_id],
                    include=["metadatas"]
                )
                if result and result["metadatas"]:
                    metadata = result["metadatas"][0]
                    themes_raw = metadata.get("themes", [])
                    key_topics_raw = metadata.get("key_topics", [])
                    # Themes might be JSON string or list
                    if isinstance(themes_raw, str):
                        themes = json.loads(themes_raw)
                    else:
                        themes = themes_raw
                    if isinstance(key_topics_raw, str):
                        vector_key_topics = json.loads(key_topics_raw)
                    else:
                        vector_key_topics = key_topics_raw
        except Exception as e:
            print(f"Warning: Could not retrieve themes from vector store: {e}")
        
        print(f"Themes: {len(themes)}")
        print(f"Vector key topics: {len(vector_key_topics)}")
        
        # Build comprehensive JSON structure
        data = {
            "conversation": {
                "id": conversation.id,
                "character_id": conversation.character_id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
                "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
                "last_analyzed_at": conversation.last_analyzed_at.isoformat() if conversation.last_analyzed_at else None,
            },
            "analysis": {
                "id": summary.id,
                "analyzed_at": summary.created_at.isoformat() if summary.created_at else None,
                "message_count": summary.message_count,
                "message_range": {
                    "start": summary.message_range_start,
                    "end": summary.message_range_end
                },
                "summary": summary.summary,
                "themes": themes,  # From vector store (legacy)
                "tone": summary.tone,
                "emotional_arc": summary.emotional_arc,
                "key_topics": summary.key_topics if summary.key_topics else [],
                "vector_key_topics": vector_key_topics,
                "participants": summary.participants if summary.participants else [],
                "manual_analysis": summary.manual == "true"
            },
            "memories": [
                {
                    "id": mem.id,
                    "type": mem.memory_type.value,
                    "content": mem.content,
                    "confidence": mem.confidence,
                    "category": mem.category,
                    "status": mem.status,
                    "priority": mem.priority,
                    "created_at": mem.created_at.isoformat() if mem.created_at else None
                }
                for mem in memories
            ]
        }
        
        # Output JSON
        json_output = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Save to file
        output_dir = Path("data/exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"conversation_analysis_{conversation_id[:8]}_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_output)
        
        print(f"\n✓ Saved to: {output_file}")
        
        return data
        
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utilities/extract_conversation_summary.py <conversation_id>")
        print("Example: python utilities/extract_conversation_summary.py b4a106a3-4ff3-45dd-917c-9b3e6f0501a4")
        sys.exit(1)
    
    conversation_id = sys.argv[1]
    extract_conversation_analysis(conversation_id)
