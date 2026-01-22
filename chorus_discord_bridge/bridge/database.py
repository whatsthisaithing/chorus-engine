"""
Database initialization and management for Discord Bridge state.

Handles SQLite database creation, migrations, and connection management.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:
    """Manages SQLite database connection and initialization."""
    
    def __init__(self, db_path: str = "storage/state.db"):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        
    def connect(self) -> sqlite3.Connection:
        """Get or create database connection.
        
        Returns:
            SQLite connection with row factory enabled
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Allow multi-threaded access
            )
            # Return rows as dictionaries
            self._connection.row_factory = sqlite3.Row
            
        return self._connection
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def init_database(self) -> bool:
        """Initialize database schema from schema.sql.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Look for schema.sql in the storage directory (same as db file)
            schema_path = self.db_path.parent / "schema.sql"
            
            # If not found there, try relative to this module's location
            if not schema_path.exists():
                module_dir = Path(__file__).parent.parent
                schema_path = module_dir / "storage" / "schema.sql"
            
            if not schema_path.exists():
                logger.error(f"Schema file not found: {schema_path}")
                logger.error(f"Tried: {self.db_path.parent / 'schema.sql'} and {module_dir / 'storage' / 'schema.sql'}")
                return False
            
            logger.info(f"Loading schema from: {schema_path}")
            
            # Read schema SQL
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Execute schema
            conn = self.connect()
            conn.executescript(schema_sql)
            conn.commit()
            
            # Verify schema version
            version = self.get_schema_version()
            logger.info(f"Database initialized successfully (schema version {version})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            return False
    
    def get_schema_version(self) -> int:
        """Get current schema version.
        
        Returns:
            Schema version number, or 0 if not initialized
        """
        try:
            conn = self.connect()
            cursor = conn.execute(
                "SELECT MAX(version) as version FROM schema_version"
            )
            row = cursor.fetchone()
            return row['version'] if row and row['version'] else 0
            
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return 0
        except Exception as e:
            logger.error(f"Failed to get schema version: {e}")
            return 0
    
    def migrate(self, target_version: Optional[int] = None) -> bool:
        """Run database migrations.
        
        Args:
            target_version: Target schema version (None = latest)
            
        Returns:
            True if successful, False otherwise
        """
        current_version = self.get_schema_version()
        
        if target_version is None:
            target_version = 2  # Current latest version (added character_id)
        
        if current_version >= target_version:
            logger.info(f"Database already at version {current_version}")
            return True
        
        logger.info(f"Migrating database from v{current_version} to v{target_version}")
        
        # Run migrations in order
        for version in range(current_version + 1, target_version + 1):
            if not self._run_migration(version):
                logger.error(f"Migration to v{version} failed")
                return False
        
        return True
    
    def _run_migration(self, version: int) -> bool:
        """Run a specific migration.
        
        Args:
            version: Migration version to run
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Look for migration file
            migration_file = self.db_path.parent / "migrations" / f"{version:03d}_*.sql"
            migration_files = list(self.db_path.parent.glob(f"migrations/{version:03d}_*.sql"))
            
            # If not found in db path, try module location
            if not migration_files:
                module_dir = Path(__file__).parent.parent
                migration_files = list(module_dir.glob(f"storage/migrations/{version:03d}_*.sql"))
            
            if not migration_files:
                logger.error(f"Migration file for v{version} not found")
                return False
            
            migration_path = migration_files[0]
            logger.info(f"Running migration: {migration_path.name}")
            
            # Read and execute migration
            with open(migration_path, 'r', encoding='utf-8') as f:
                migration_sql = f.read()
            
            conn = self.connect()
            conn.executescript(migration_sql)
            conn.commit()
            
            logger.info(f"Migration to v{version} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run migration v{version}: {e}", exc_info=True)
            return False
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query and return cursor.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Cursor with results
        """
        conn = self.connect()
        return conn.execute(query, params)
    
    def execute_many(self, query: str, params_list: list) -> None:
        """Execute query with multiple parameter sets.
        
        Args:
            query: SQL query
            params_list: List of parameter tuples
        """
        conn = self.connect()
        conn.executemany(query, params_list)
        conn.commit()
    
    def commit(self):
        """Commit current transaction."""
        if self._connection:
            self._connection.commit()


# Global database instance
_db_instance: Optional[Database] = None


def get_database(db_path: str = "storage/state.db") -> Database:
    """Get or create global database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Database instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = Database(db_path)
        
        # Initialize if needed
        if _db_instance.get_schema_version() == 0:
            logger.info("Database not initialized, running init_database()")
            _db_instance.init_database()
    
    return _db_instance


def init_database(db_path: str = "storage/state.db") -> bool:
    """Initialize database (convenience function).
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if successful
    """
    db = get_database(db_path)
    return db.init_database()
