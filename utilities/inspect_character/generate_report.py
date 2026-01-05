"""
Character Database Inspector

Generates a markdown report of all SQL database and vector store entries for a character.
Usage: python utilities/inspect_character/inspect.py <character_id>
Example: python utilities/inspect_character/inspect.py sarah_v1
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import chromadb

from chorus_engine.models.conversation import Conversation, Thread, Message, Memory, ConversationSummary

# Database paths
DATABASE_DIR = Path(__file__).parent.parent.parent / "data"
DATABASE_PATH = DATABASE_DIR / "chorus.db"
VECTOR_STORE_PATH = DATABASE_DIR / "vector_store"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "reports"


def format_datetime(dt):
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate(text, max_length=80):
    """Truncate text for display."""
    if text is None:
        return "N/A"
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def escape_markdown(text):
    """Escape special markdown characters."""
    if text is None:
        return "N/A"
    text = str(text)
    # Replace newlines with spaces (they break table rows)
    text = text.replace("\n", " ").replace("\r", " ")
    # Collapse multiple spaces into one
    while "  " in text:
        text = text.replace("  ", " ")
    # Escape pipe characters that would break tables
    text = text.replace("|", "\\|")
    return text


def inspect_sql_database(character_id: str, md_file):
    """Inspect SQL database and write to markdown file."""
    md_file.write(f"## SQL Database\n\n")
    
    # Connect to database
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # === Conversations ===
        conversations = session.query(Conversation).filter(
            Conversation.character_id == character_id
        ).all()
        
        md_file.write(f"### Conversations ({len(conversations)})\n\n")
        
        if conversations:
            md_file.write("| ID | Title | Messages | Created | Updated |\n")
            md_file.write("|---|---|---:|---|---|\n")
            
            for conv in conversations:
                # Count messages
                msg_count = session.query(Message).join(Thread).filter(
                    Thread.conversation_id == conv.id
                ).count()
                
                md_file.write(f"| `{conv.id[:8]}...` | {escape_markdown(conv.title)} | {msg_count} | {format_datetime(conv.created_at)} | {format_datetime(conv.updated_at)} |\n")
            
            md_file.write("\n")
        else:
            md_file.write("*No conversations found*\n\n")
        
        # === Threads ===
        threads = session.query(Thread).join(Conversation).filter(
            Conversation.character_id == character_id
        ).all()
        
        md_file.write(f"### Threads ({len(threads)})\n\n")
        
        if threads:
            md_file.write("| ID | Conversation | Title | Messages | Created |\n")
            md_file.write("|---|---|---|---:|---|\n")
            
            for thread in threads:
                # Count messages
                thread_msg_count = session.query(Message).filter(
                    Message.thread_id == thread.id
                ).count()
                
                md_file.write(f"| `{thread.id[:8]}...` | `{thread.conversation_id[:8]}...` | {escape_markdown(thread.title)} | {thread_msg_count} | {format_datetime(thread.created_at)} |\n")
            
            md_file.write("\n")
        else:
            md_file.write("*No threads found*\n\n")
        
        # === Messages ===
        messages = session.query(Message).join(Thread).join(Conversation).filter(
            Conversation.character_id == character_id
        ).order_by(Message.created_at.desc()).all()
        
        md_file.write(f"### Messages ({len(messages)})\n\n")
        
        if messages:
            # Show first 20 messages
            display_count = min(len(messages), 20)
            md_file.write("| ID | Thread | Role | Content | Created |\n")
            md_file.write("|---|---|---|---|---|\n")
            
            for msg in messages[:display_count]:
                content_preview = escape_markdown(truncate(msg.content, 60))
                md_file.write(f"| `{msg.id[:8]}...` | `{msg.thread_id[:8]}...` | {msg.role} | {content_preview} | {format_datetime(msg.created_at)} |\n")
            
            if len(messages) > display_count:
                md_file.write(f"\n*Showing {display_count} of {len(messages)} messages*\n")
            
            md_file.write("\n")
        else:
            md_file.write("*No messages found*\n\n")
        
        # === Memories ===
        memories = session.query(Memory).filter(
            Memory.character_id == character_id
        ).order_by(Memory.created_at.desc()).all()
        
        md_file.write(f"### Memories ({len(memories)})\n\n")
        
        if memories:
            # Group by type
            mem_types = {}
            for mem in memories:
                mem_type = mem.memory_type.value if hasattr(mem.memory_type, 'value') else str(mem.memory_type)
                mem_types[mem_type] = mem_types.get(mem_type, 0) + 1
            
            md_file.write("**Memory Types:**\n")
            for mem_type, count in sorted(mem_types.items()):
                md_file.write(f"- `{mem_type}`: {count}\n")
            md_file.write("\n")
            
            # Show first 30 memories
            display_count = min(len(memories), 30)
            md_file.write("| ID | Type | Content | Priority | Confidence | Conv ID | Created |\n")
            md_file.write("|---|---|---|---:|---:|---|---|\n")
            
            for mem in memories[:display_count]:
                mem_type = mem.memory_type.value if hasattr(mem.memory_type, 'value') else str(mem.memory_type)
                content_preview = escape_markdown(truncate(mem.content, 50))
                conf = f"{mem.confidence:.2f}" if mem.confidence else "N/A"
                conv_id = f"`{mem.conversation_id[:8]}...`" if mem.conversation_id else "*orphaned*"
                
                md_file.write(f"| `{mem.id[:8]}...` | `{mem_type}` | {content_preview} | {mem.priority or 0} | {conf} | {conv_id} | {format_datetime(mem.created_at)} |\n")
            
            if len(memories) > display_count:
                md_file.write(f"\n*Showing {display_count} of {len(memories)} memories*\n")
            
            md_file.write("\n")
        else:
            md_file.write("*No memories found*\n\n")
        
        # === Conversation Analyses ===
        analyses = session.query(ConversationSummary).join(Conversation).filter(
            Conversation.character_id == character_id
        ).order_by(ConversationSummary.created_at.desc()).all()
        
        if analyses:
            md_file.write(f"### Conversation Analyses ({len(analyses)})\n\n")
            md_file.write("| ID | Conversation | Summary | Tone | Manual | Created |\n")
            md_file.write("|---|---|---|---|:---:|---|\n")
            
            for analysis in analyses:
                summary_preview = escape_markdown(truncate(analysis.summary, 60))
                tone = escape_markdown(truncate(analysis.tone, 30)) if analysis.tone else "N/A"
                manual = "✓" if analysis.manual == "true" else ""
                
                md_file.write(f"| `{analysis.id[:8]}...` | `{analysis.conversation_id[:8]}...` | {summary_preview} | {tone} | {manual} | {format_datetime(analysis.created_at)} |\n")
            
            md_file.write("\n")
        
        # === Summary ===
        md_file.write("### Summary\n\n")
        md_file.write(f"- **Conversations:** {len(conversations)}\n")
        md_file.write(f"- **Threads:** {len(threads)}\n")
        md_file.write(f"- **Messages:** {len(messages)}\n")
        md_file.write(f"- **Memories:** {len(memories)}\n")
        md_file.write(f"- **Analyses:** {len(analyses)}\n")
        md_file.write("\n")
        
    except Exception as e:
        md_file.write(f"**Error inspecting SQL database:** {e}\n\n")
    finally:
        session.close()


def inspect_vector_store(character_id: str, md_file):
    """Inspect ChromaDB vector store and write to markdown file."""
    md_file.write(f"## Vector Store\n\n")
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
        
        # Try to get character collection
        collection_name = f"character_{character_id}"
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            md_file.write(f"*No vector store collection found: `{collection_name}`*\n\n")
            return
        
        # Get collection metadata
        count = collection.count()
        md_file.write(f"**Collection:** `{collection_name}`  \n")
        md_file.write(f"**Total Vectors:** {count}\n\n")
        
        if count == 0:
            md_file.write("*Collection is empty*\n\n")
            return
        
        # Get all entries (limit to first 50 for display)
        limit = min(count, 50)
        results = collection.get(
            limit=limit,
            include=["metadatas", "documents"]
        )
        
        # Type summary
        types = {}
        for metadata in results['metadatas']:
            mem_type = metadata.get('type', 'unknown')  # Note: stored as 'type', not 'memory_type'
            types[mem_type] = types.get(mem_type, 0) + 1
        
        if types:
            md_file.write("**Vector Types:**\n")
            for vec_type, vec_count in sorted(types.items()):
                md_file.write(f"- `{vec_type}`: {vec_count}\n")
            md_file.write("\n")
        
        # Vector table
        md_file.write("### Vector Entries\n\n")
        md_file.write("| Vector ID | Type | Content | Conversation | Category |\n")
        md_file.write("|---|---|---|---|---|\n")
        
        for i, vec_id in enumerate(results['ids']):
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            document = results['documents'][i] if results['documents'] else ""
            
            mem_type = metadata.get('type', 'unknown')  # Note: stored as 'type', not 'memory_type'
            content = escape_markdown(truncate(document, 60))
            conv_id = 'N/A'  # Note: conversation_id is NOT stored in vector metadata
            if conv_id and conv_id != 'N/A':
                conv_id = f"`{conv_id[:8]}...`"
            category = metadata.get('category', 'N/A')
            
            md_file.write(f"| `{vec_id[:8]}...` | `{mem_type}` | {content} | {conv_id} | {category} |\n")
        
        if count > limit:
            md_file.write(f"\n*Showing {limit} of {count} vectors*\n")
        
        md_file.write("\n")
        
    except Exception as e:
        md_file.write(f"**Error inspecting vector store:** {e}\n\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Error: Character ID required")
        print("\nUsage:")
        print("  python utilities/inspect_character/inspect.py <character_id>")
        print("\nExample:")
        print("  python utilities/inspect_character/inspect.py sarah_v1")
        sys.exit(1)
    
    character_id = sys.argv[1]
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{character_id}_{timestamp}.md"
    
    print(f"Inspecting character: {character_id}")
    print(f"Output file: {output_file}")
    
    # Generate markdown report
    with open(output_file, 'w', encoding='utf-8') as md_file:
        # Header
        md_file.write(f"# Character Database Report: {character_id}\n\n")
        md_file.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        md_file.write(f"**Database:** `{DATABASE_PATH}`  \n")
        md_file.write(f"**Vector Store:** `{VECTOR_STORE_PATH}`\n\n")
        md_file.write("---\n\n")
        
        # Inspect SQL database
        inspect_sql_database(character_id, md_file)
        
        # Inspect vector store
        inspect_vector_store(character_id, md_file)
        
        # Footer
        md_file.write("---\n\n")
        md_file.write(f"*Report generated by Character Database Inspector*\n")
    
    print(f"✓ Report generated successfully!")
    print(f"  Open: {output_file.absolute()}")


if __name__ == "__main__":
    main()
