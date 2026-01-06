# Database Migration System

This document explains how Chorus Engine's database migration system works and how to use it.

## Overview

Chorus Engine uses **Alembic** (the standard SQLAlchemy migration tool) to manage database schema changes. The migration system automatically handles:

1. **Fresh installs**: Creates the latest schema directly
2. **Existing databases**: Applies incremental migrations
3. **Version tracking**: Keeps database in sync with code

## How It Works

### Automatic Migration on Startup

The migration system runs automatically every time Chorus Engine starts:

```python
# In chorus_engine/api/app.py
from chorus_engine.db.migrations import ensure_database_ready
from chorus_engine.db.database import DATABASE_URL

ensure_database_ready(DATABASE_URL)
```

### Three Scenarios

**1. Fresh Install (No database exists)**
- Creates all tables using `Base.metadata.create_all()`
- Stamps database with current migration version
- No migrations are applied (already at latest schema)

**2. Existing Pre-Migration Database (Has tables but no alembic_version)**
- Stamps database at "base" (before any migrations)
- Applies all migrations to add new features
- Example: Adding document analysis tables to existing Chorus Engine install

**3. Existing Migrated Database (Has alembic_version table)**
- Checks for pending migrations
- Applies only the new migrations needed
- Keeps database in sync with code

## Creating a New Migration

When you add new database models or modify existing ones, create a migration:

### Method 1: Manual Migration (Recommended for complex changes)

Create a file in `alembic/versions/` with this naming pattern:
```
YYYYMMDD_HHMM_<revision>_<slug>.py
```

Example: `001_add_documents.py`

```python
"""Add document analysis tables

Revision ID: 001_add_documents
Revises: 
Create Date: 2025-01-28 10:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = '001_add_documents'
down_revision: Union[str, None] = None  # Previous migration (None = first migration)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new tables/columns."""
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('filename', sa.String(length=500), nullable=False),
        # ... more columns
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Remove tables/columns (for rollback)."""
    op.drop_table('documents')
```

### Method 2: Autogenerate Migration (For simple changes)

**WARNING**: Autogenerate can fail if your database has foreign key issues or custom types.

```python
from chorus_engine.db.migrations import MigrationManager
from chorus_engine.db.database import DATABASE_URL

manager = MigrationManager(DATABASE_URL)
manager.create_migration("add user preferences table", autogenerate=True)
```

Then review and edit the generated migration file in `alembic/versions/`.

## Database Models

All database models must:

1. Inherit from `Base` (defined in `chorus_engine.db.database`)
2. Be imported in `chorus_engine/models/__init__.py`
3. Be imported in `alembic/env.py` (so Alembic can see them)

Example model:

```python
# chorus_engine/models/document.py
from chorus_engine.db.database import Base
from sqlalchemy import Column, Integer, String, Text, DateTime

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
```

Then register it:

```python
# chorus_engine/models/__init__.py
from .document import Document

__all__ = ["Document", ...]
```

```python
# alembic/env.py
import chorus_engine.models.document  # Add this import
```

## Migration System Architecture

```
chorus_engine/
├── db/
│   ├── database.py          # Base, engine, session management
│   ├── migrations.py        # MigrationManager class
│   └── vector_store.py
├── models/
│   ├── __init__.py          # All models imported here
│   ├── conversation.py
│   ├── document.py
│   └── workflow.py
alembic/
├── env.py                   # Migration environment config
├── script.py.mako           # Migration template
└── versions/
    └── 001_add_documents.py # Migration scripts
alembic.ini                  # Alembic configuration
```

### Key Files

**`chorus_engine/db/migrations.py`**
- `MigrationManager` class - orchestrates all migration operations
- `ensure_database_ready()` - main entry point (called on startup)
- `is_fresh_database()` - detects if database exists
- `has_alembic_version_table()` - checks migration tracking
- `apply_migrations()` - applies pending migrations
- `initialize_fresh_database()` - creates fresh schema

**`alembic/env.py`**
- Imports all models to populate SQLAlchemy metadata
- Configures Alembic runtime environment
- Handles online/offline migration modes

**`alembic/versions/`**
- Contains all migration scripts
- Named with revision ID and timestamp
- Each migration has `upgrade()` and `downgrade()` functions

## Common Operations

### Check Current Migration Status

```bash
python -c "from chorus_engine.db.migrations import MigrationManager; from chorus_engine.db.database import DATABASE_URL; m = MigrationManager(DATABASE_URL); print(f'Current: {m.get_current_revision()}'); print(f'Latest: {m.get_head_revision()}')"
```

### Apply Migrations Manually

```python
from chorus_engine.db.migrations import ensure_database_ready
from chorus_engine.db.database import DATABASE_URL

ensure_database_ready(DATABASE_URL)
```

### Rollback to Previous Migration

```python
from chorus_engine.db.migrations import MigrationManager
from chorus_engine.db.database import DATABASE_URL

manager = MigrationManager(DATABASE_URL)
manager.downgrade("-1")  # Rollback one migration
```

**WARNING**: Downgrades can lose data! Only use for development/testing.

### View Migration History

```bash
alembic history
```

### View Current Migration

```bash
alembic current
```

## Testing Migrations

After creating a migration, test both paths:

### Test 1: Fresh Install
```bash
# Delete database
rm data/chorus.db

# Start server (will create schema and stamp version)
python -m chorus_engine.main

# Verify tables exist
python -c "from chorus_engine.db.database import engine; from sqlalchemy import inspect; print('Tables:', inspect(engine).get_table_names())"
```

### Test 2: Existing Database Upgrade
```bash
# Use database with old schema (before your migration)
# Start server (will apply your migration)
python -m chorus_engine.main

# Verify new tables/columns exist
python test_document_schema.py
```

### Test 3: Idempotency
```bash
# Start server twice in a row
python -m chorus_engine.main
# (Ctrl+C to stop)
python -m chorus_engine.main

# Should see "No pending migrations" in logs
```

## Migration Best Practices

1. **Always test migrations on a backup database first**
2. **Write descriptive migration messages**
3. **Keep migrations small and focused** (one feature per migration)
4. **Test both upgrade() and downgrade() functions**
5. **Never modify existing migrations** (create new ones instead)
6. **Review autogenerated migrations carefully** (they can be wrong)
7. **Add indexes for foreign keys and frequently queried columns**
8. **Use nullable=True for new columns** (easier to add to existing data)

## Troubleshooting

### Error: "table already exists"
- Migration tried to create a table that exists
- Solution: Check if you're running migrations on an already-migrated database
- Fix: Manually stamp database to correct version

### Error: "no such table: characters"
- Foreign key references non-existent table
- Solution: Create manual migration (don't use autogenerate)
- Fix: See `alembic/versions/001_add_documents.py` for example

### Error: "migration revision not found"
- Alembic version table has invalid revision ID
- Solution: Check `alembic_version` table in database
- Fix: Update to valid revision or stamp with correct version

### Migrations not applying on startup
- Check logs for migration errors
- Verify `ensure_database_ready()` is called in app.py
- Ensure all models are imported in alembic/env.py

## Phase 1 Document Analysis Migration

The first migration (`001_add_documents`) adds four tables:

1. **documents**: Stores document metadata (filename, type, processing status)
2. **document_chunks**: Stores document chunks for vector retrieval
3. **document_access_logs**: Tracks when documents are accessed in conversations
4. **code_execution_logs**: Logs code execution attempts (for Phase 7)

### Migration Strategy
- Existing Chorus Engine installations: Migration applied automatically on first startup
- Fresh installations: Tables created directly (no migration applied)
- Both paths result in identical schema

## Future Migrations

When adding Phase 2-7 features:

1. Create new migration with `down_revision` = previous migration
2. Test on both fresh and existing databases
3. Document any breaking changes
4. Provide upgrade guide if data migration needed

Example:
```python
# 002_add_code_sandbox.py
revision = '002_add_code_sandbox'
down_revision = '001_add_documents'  # Points to previous migration
```

## Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- Chorus Engine Migration Code: `chorus_engine/db/migrations.py`
