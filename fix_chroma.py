"""Fix corrupted conversation summary collection in ChromaDB."""
import sqlite3
from pathlib import Path
import shutil

db_path = Path("data/vector_store/chroma.sqlite3")

if not db_path.exists():
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

coll_id = '19bc56ce-21c7-4d83-a239-398191ada8cc'
print(f"Cleaning up collection: {coll_id}")

# Delete segments associated with this collection
cursor.execute("SELECT id FROM segments WHERE collection = ?", (coll_id,))
segments = cursor.fetchall()
print(f"Found {len(segments)} segments to delete")

for seg in segments:
    seg_id = seg[0]
    # Delete segment metadata
    cursor.execute("DELETE FROM segment_metadata WHERE segment_id = ?", (seg_id,))
    print(f"  Deleted metadata for segment {seg_id}")

# Delete segments
cursor.execute("DELETE FROM segments WHERE collection = ?", (coll_id,))
print(f"Deleted segment records")

# Delete collection record if exists
cursor.execute("DELETE FROM collections WHERE id = ?", (coll_id,))

conn.commit()
conn.close()

print("\nCleanup complete. Restart server to recreate fresh collection.")
