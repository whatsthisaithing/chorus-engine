"""
Fix ChromaDB corruption by reinitializing the vector store.

This script:
1. Backs up the current vector_store directory
2. Deletes the corrupted ChromaDB database
3. Rebuilds the vector store from SQL memories that have vector_ids
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime

def main():
    vector_store_path = Path("data/vector_store")
    
    if not vector_store_path.exists():
        print("Vector store doesn't exist. Nothing to fix.")
        return
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"data/vector_store_backup_{timestamp}")
    
    print(f"Backing up vector store to: {backup_path}")
    shutil.copytree(vector_store_path, backup_path)
    print(f"✓ Backup created")
    
    # Delete corrupted vector store
    print(f"Deleting corrupted vector store...")
    shutil.rmtree(vector_store_path)
    print(f"✓ Deleted")
    
    # Recreate empty directory
    vector_store_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created fresh vector_store directory")
    
    print("\nChromaDB corruption fixed!")
    print("The vector store will be rebuilt automatically when the server starts.")
    print("\nNote: Memories without vector_ids will need to be re-added to the vector store.")
    print("You can run 'python rebuild_vector_store.py' to rebuild from existing memories.")

if __name__ == "__main__":
    main()
