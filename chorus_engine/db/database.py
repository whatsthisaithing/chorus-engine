"""Database configuration and session management."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Base class for all models
Base = declarative_base()

# Database configuration
DATABASE_DIR = Path(__file__).parent.parent.parent / "data"
DATABASE_PATH = DATABASE_DIR / "chorus.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """
    Get a database session.
    
    Usage in FastAPI endpoints:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db here
            pass
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize the database.
    
    Creates all tables if they don't exist.
    Should be called on application startup.
    """
    # Ensure data directory exists
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Import all models so they're registered with Base
    from chorus_engine.models import conversation  # noqa: F401
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Database initialized at: {DATABASE_PATH}")
