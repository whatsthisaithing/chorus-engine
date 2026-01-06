"""
Database migration management for Chorus Engine.

Handles both fresh installs and incremental migrations for existing databases.
Uses Alembic for migration tracking and execution.
"""

import logging
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
from alembic.config import Config
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Manage database migrations for Chorus Engine.
    
    Supports two scenarios:
    1. Fresh install: Create schema directly (no migrations needed)
    2. Existing install: Apply incremental migrations from current version
    """
    
    def __init__(self, db_url: str, migrations_dir: Optional[Path] = None):
        """
        Initialize migration manager.
        
        Args:
            db_url: SQLAlchemy database URL (e.g., "sqlite:///data/chorus.db")
            migrations_dir: Path to migrations directory (defaults to alembic/)
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        
        # Default migrations directory
        if migrations_dir is None:
            migrations_dir = Path(__file__).parent.parent.parent / "alembic"
        
        self.migrations_dir = migrations_dir
        
        # Alembic configuration
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", str(self.migrations_dir))
        self.alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        self.alembic_cfg.attributes["configure_logger"] = False  # Use our logger
    
    def is_fresh_database(self) -> bool:
        """
        Check if this is a fresh database (no tables).
        
        Returns:
            True if database has no tables, False otherwise
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        
        if not tables:
            logger.info("Fresh database detected - no tables exist")
            return True
        
        logger.info(f"Existing database detected - {len(tables)} tables found")
        return False
    
    def get_current_revision(self) -> Optional[str]:
        """
        Get current migration revision from database.
        
        Returns:
            Current revision ID or None if no migrations applied
        """
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                
                if current_rev:
                    logger.info(f"Current migration revision: {current_rev}")
                else:
                    logger.info("No migration revision found (might be fresh install)")
                
                return current_rev
        except Exception as e:
            logger.warning(f"Could not determine current revision: {e}")
            return None
    
    def get_head_revision(self) -> str:
        """
        Get the latest migration revision from migration scripts.
        
        Returns:
            Latest revision ID
        """
        script = ScriptDirectory.from_config(self.alembic_cfg)
        head = script.get_current_head()
        logger.info(f"Latest migration revision: {head}")
        return head
    
    def has_pending_migrations(self) -> bool:
        """
        Check if there are migrations that haven't been applied.
        
        Returns:
            True if migrations are pending, False otherwise
        """
        current = self.get_current_revision()
        head = self.get_head_revision()
        
        if current is None:
            # No version tracking - might be pre-migration database
            return True
        
        if current != head:
            logger.info(f"Pending migrations detected: {current} -> {head}")
            return True
        
        logger.info("Database is up to date - no pending migrations")
        return False
    
    def initialize_fresh_database(self):
        """
        Initialize a fresh database with the latest schema.
        
        This creates all tables using SQLAlchemy's create_all() and stamps
        the database with the current migration version.
        """
        from chorus_engine.db.database import Base
        
        logger.info("Initializing fresh database with latest schema...")
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        logger.info("All tables created successfully")
        
        # Stamp database with current migration version (no migrations needed)
        logger.info("Stamping database with current migration version...")
        command.stamp(self.alembic_cfg, "head")
        logger.info("Database stamped - ready to use")
    
    def apply_migrations(self):
        """
        Apply pending migrations to bring database up to date.
        
        Uses Alembic to incrementally apply migrations from current version
        to latest version.
        """
        if not self.has_pending_migrations():
            logger.info("No migrations to apply")
            return
        
        logger.info("Applying pending migrations...")
        
        try:
            # Run migrations to head
            command.upgrade(self.alembic_cfg, "head")
            logger.info("All migrations applied successfully")
        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to apply migrations: {e}")
    
    def has_alembic_version_table(self) -> bool:
        """
        Check if the alembic_version table exists.
        
        Returns:
            True if alembic_version table exists, False otherwise
        """
        inspector = inspect(self.engine)
        return 'alembic_version' in inspector.get_table_names()
    
    def ensure_database_ready(self):
        """
        Ensure database is ready for use.
        
        This is the main entry point - call this on application startup.
        
        Handles three scenarios:
        1. Fresh install: No tables exist - creates schema and stamps version
        2. Existing pre-migration database: Has tables but no alembic_version - stamps baseline and applies migrations
        3. Existing migrated database: Has alembic_version - applies pending migrations
        """
        logger.info("Checking database state...")
        
        if self.is_fresh_database():
            # Fresh install - create schema directly
            logger.info("Fresh install detected - creating schema...")
            self.initialize_fresh_database()
        elif not self.has_alembic_version_table():
            # Existing database without migration tracking - stamp at base then apply migrations
            logger.info("Existing database without migration tracking detected...")
            logger.info("This database needs to be brought under migration management")
            logger.info("Stamping database at 'base' (before any migrations)...")
            command.stamp(self.alembic_cfg, "base")
            logger.info("Database stamped at baseline")
            
            # Now apply all migrations to add new features
            logger.info("Applying migrations to add new tables...")
            self.apply_migrations()
        else:
            # Existing database with migrations - check for pending migrations
            logger.info("Existing database with migration tracking - checking for updates...")
            self.apply_migrations()
        
        logger.info("Database is ready")
    
    def create_migration(self, message: str, autogenerate: bool = True):
        """
        Create a new migration script.
        
        Args:
            message: Description of the migration
            autogenerate: Whether to auto-detect schema changes
        
        Usage:
            manager.create_migration("add document tables")
        """
        logger.info(f"Creating new migration: {message}")
        
        try:
            if autogenerate:
                # Auto-detect changes by comparing model to database
                command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                # Create empty migration template
                command.revision(
                    self.alembic_cfg,
                    message=message
                )
            
            logger.info("Migration created successfully")
            logger.info("Review the generated migration file before applying!")
        except Exception as e:
            logger.error(f"Failed to create migration: {e}", exc_info=True)
            raise
    
    def downgrade(self, revision: str = "-1"):
        """
        Downgrade database to a previous revision.
        
        Args:
            revision: Target revision ("-1" for previous, "base" for initial)
        
        Warning: Use with caution - may result in data loss
        """
        logger.warning(f"Downgrading database to revision: {revision}")
        
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info("Downgrade completed")
        except Exception as e:
            logger.error(f"Downgrade failed: {e}", exc_info=True)
            raise
    
    def show_current_state(self):
        """
        Display current database migration state.
        
        Useful for debugging migration issues.
        """
        logger.info("=== Database Migration State ===")
        logger.info(f"Database URL: {self.db_url}")
        logger.info(f"Migrations directory: {self.migrations_dir}")
        
        if self.is_fresh_database():
            logger.info("Status: Fresh database (no tables)")
        else:
            current = self.get_current_revision()
            head = self.get_head_revision()
            
            logger.info(f"Current revision: {current}")
            logger.info(f"Latest revision: {head}")
            
            if self.has_pending_migrations():
                logger.info("Status: Pending migrations")
            else:
                logger.info("Status: Up to date")


# Convenience function for application startup
def ensure_database_ready(db_url: str):
    """
    Ensure database is ready for use (call this on startup).
    
    Args:
        db_url: SQLAlchemy database URL
    
    Example:
        ensure_database_ready("sqlite:///data/chorus.db")
    """
    manager = MigrationManager(db_url)
    manager.ensure_database_ready()
