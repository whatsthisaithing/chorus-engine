#!/usr/bin/env python
"""
Pre-update diagnostics for Chorus Engine.
Checks for and optionally fixes ChromaDB collection configuration issues.

This script runs BEFORE pulling new code so users can backup if needed.
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_chromadb_collections() -> Tuple[bool, List[str], Dict[str, int]]:
    """
    Check ChromaDB collections for configuration issues.
    
    Returns:
        Tuple of (has_issues, collection_names_with_issues, stats)
    """
    chroma_db_path = Path("data/vector_store/chroma.sqlite3")
    
    if not chroma_db_path.exists():
        return False, [], {"total": 0, "with_issues": 0}
    
    try:
        conn = sqlite3.connect(chroma_db_path)
        cursor = conn.cursor()
        
        # Get all collections and their configs
        cursor.execute("SELECT id, name, config_json_str FROM collections")
        collections = cursor.fetchall()
        
        conn.close()
        
        # Check for empty configs (missing _type field)
        collections_with_issues = []
        for coll_id, name, config_json in collections:
            if config_json == "{}":
                collections_with_issues.append(name)
        
        stats = {
            "total": len(collections),
            "with_issues": len(collections_with_issues)
        }
        
        return len(collections_with_issues) > 0, collections_with_issues, stats
        
    except Exception as e:
        print(f"⚠️  Warning: Could not check ChromaDB: {e}")
        return False, [], {"total": 0, "with_issues": 0}


def fix_chromadb_collections(dry_run: bool = False) -> bool:
    """
    Fix ChromaDB collections with empty configuration.
    
    Args:
        dry_run: If True, only report what would be fixed
    
    Returns:
        True if successful (or would be successful in dry_run mode)
    """
    chroma_db_path = Path("data/vector_store/chroma.sqlite3")
    
    if not chroma_db_path.exists():
        return True  # Nothing to fix
    
    # Proper HNSW configuration
    proper_config = {
        "hnsw_configuration": {
            "space": "l2",
            "ef_construction": 100,
            "ef_search": 10,
            "num_threads": 20,
            "M": 16,
            "resize_factor": 1.2,
            "batch_size": 100,
            "sync_threshold": 1000,
            "_type": "HNSWConfigurationInternal"
        },
        "_type": "CollectionConfigurationInternal"
    }
    
    try:
        conn = sqlite3.connect(chroma_db_path)
        cursor = conn.cursor()
        
        # Find collections with empty config
        cursor.execute("SELECT id, name, config_json_str FROM collections")
        collections = cursor.fetchall()
        
        collections_to_fix = [(coll_id, name) for coll_id, name, config_json in collections if config_json == "{}"]
        
        if not collections_to_fix:
            conn.close()
            return True  # Nothing to fix
        
        if dry_run:
            conn.close()
            print(f"\n  Would fix {len(collections_to_fix)} collections:")
            for _, name in collections_to_fix:
                print(f"    - {name}")
            return True
        
        # Apply fixes
        proper_config_str = json.dumps(proper_config)
        for coll_id, name in collections_to_fix:
            cursor.execute(
                "UPDATE collections SET config_json_str = ? WHERE id = ?",
                (proper_config_str, coll_id)
            )
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error fixing collections: {e}")
        return False


def check_pending_memories() -> Dict[str, int]:
    """
    Check for pending memories (informational, not an error).
    
    Returns:
        Dictionary with pending memory counts by character
    """
    try:
        from sqlalchemy import text
        from chorus_engine.db.database import engine
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT character_id, COUNT(*) as count
                FROM memories
                WHERE status = 'pending'
                GROUP BY character_id
            """))
            
            pending = {row.character_id: row.count for row in result}
            return pending
            
    except Exception as e:
        # Silent fail - this is just informational
        return {}


def run_diagnostics(fix_issues: bool = False) -> int:
    """
    Run all pre-update diagnostics.
    
    Args:
        fix_issues: If True, automatically fix detected issues
    
    Returns:
        Exit code (0 = success, 1 = issues found, 2 = critical error)
    """
    print("=" * 80)
    print("CHORUS ENGINE PRE-UPDATE DIAGNOSTICS")
    print("=" * 80)
    
    exit_code = 0
    
    # Check ChromaDB collections
    print("\n🔍 Checking ChromaDB collections...")
    has_issues, collections_with_issues, stats = check_chromadb_collections()
    
    if stats["total"] == 0:
        print("  ℹ️  No ChromaDB collections found (fresh install)")
    elif not has_issues:
        print(f"  ✅ All {stats['total']} collections are properly configured")
    else:
        print(f"  ⚠️  Found {stats['with_issues']} collection(s) with configuration issues:")
        for name in collections_with_issues:
            print(f"      - {name}")
        
        if fix_issues:
            print("\n  🔧 Fixing collections...")
            if fix_chromadb_collections(dry_run=False):
                print("  ✅ Collections fixed successfully")
            else:
                print("  ❌ Failed to fix collections")
                exit_code = 2
        else:
            print("\n  ⚠️  These collections may cause ChromaDB access errors.")
            print("  💡 Run with '--fix' to automatically repair them.")
            print("  💡 Or manually run: python fix_chromadb_collections.py")
            exit_code = 1
    
    # Check pending memories (informational only)
    print("\n📝 Checking for pending memories...")
    pending = check_pending_memories()
    
    if not pending:
        print("  ✅ No pending memories")
    else:
        total_pending = sum(pending.values())
        print(f"  ℹ️  Found {total_pending} pending memory(ies):")
        for char_id, count in pending.items():
            print(f"      - {char_id}: {count} pending")
        print("\n  💡 Pending memories are waiting for approval in the web UI.")
        print("  💡 They will receive vector embeddings once approved.")
    
    # Summary
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ DIAGNOSTICS COMPLETE - NO ISSUES FOUND")
        if fix_issues and has_issues:
            print("   All detected issues have been automatically fixed.")
    elif exit_code == 1:
        print("⚠️  DIAGNOSTICS COMPLETE - ISSUES FOUND")
        print("   Review the issues above and consider backing up your data.")
        print("   Run 'python check_before_update.py --fix' to fix automatically.")
    else:
        print("❌ DIAGNOSTICS COMPLETE - CRITICAL ERROR")
        print("   Could not fix detected issues. Manual intervention required.")
    
    print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-update diagnostics for Chorus Engine")
    parser.add_argument("--fix", action="store_true", help="Automatically fix detected issues")
    args = parser.parse_args()
    
    exit_code = run_diagnostics(fix_issues=args.fix)
    sys.exit(exit_code)
