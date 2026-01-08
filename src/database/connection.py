"""
Database Connection Management

Handles PostgreSQL connections with proper pooling and error handling.

FIXED: SQLAlchemy 2.0 compatibility (text() wrapper for SQL strings)

Design Decisions:
-----------------
1. Connection pooling: Reuse connections for performance
2. Context managers: Automatic transaction handling
3. Pre-ping: Verify connections before use
4. Pool recycle: Refresh stale connections
5. Error handling: Proper exception management

Features:
------------------------
- Production-tested connection patterns
- Proper resource cleanup
- Thread-safe operations
- Performance optimized
- Error recovery
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def create_db_engine():
    """
    Create SQLAlchemy engine with production settings.
    
    Configuration:
    - Pool size: 10 connections (handles 10 concurrent requests)
    - Max overflow: 20 additional connections (total 30 under load)
    - Pre-ping: Test connection before use (detect disconnects)
    - Pool recycle: Refresh connections every hour (prevent stale)
    
    Returns:
        Configured SQLAlchemy engine
    """
    engine = create_engine(
        settings.DATABASE_URL,
        poolclass=QueuePool,  # Production pool (reuses connections)
        pool_size=10,  # Base pool size
        max_overflow=20,  # Additional connections under load
        pool_pre_ping=True,  # Verify connection before use
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False,  # Set to True for SQL debugging
        future=True  # Use SQLAlchemy 2.0 style
    )
    
    # Log pool events (useful for monitoring)
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine, "close")
    def receive_close(dbapi_conn, connection_record):
        logger.debug("Database connection closed")
    
    return engine


# Global engine instance
engine = create_db_engine()

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,  # Explicit commit required
    autoflush=False,  # Manual flush control
    future=True  # SQLAlchemy 2.0 style
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup.
    
    Usage pattern (context manager):
```python
    with get_db() as db:
        session = db.query(ResearchSession).filter(...).first()
        db.commit()  # Explicit commit
    # Automatically closed
```
    
    Features:
    - Automatic session cleanup
    - Transaction rollback on error
    - Proper exception propagation
    
    Yields:
        Database session
        
    Example:
        >>> with get_db() as db:
        ...     session = ResearchSession(query="Test")
        ...     db.add(session)
        ...     db.commit()
    """
    db = SessionLocal()
    try:
        yield db
        # Commit happens in calling code (explicit control)
    except Exception as e:
        db.rollback()
        logger.error(f"Database error, rolling back: {e}")
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    
    Creates all tables defined in models.py if they don't exist.
    Idempotent - safe to run multiple times.
    
    Usage:
        >>> from src.database.connection import init_db
        >>> init_db()
        Database tables created successfully
    """
    from src.database.models import Base
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def drop_all_tables():
    """
    Drop all database tables.
    
    WARNING: This deletes ALL data! Use only in development/testing.
    
    Usage:
        >>> from src.database.connection import drop_all_tables
        >>> drop_all_tables()  # BE CAREFUL!
    """
    from src.database.models import Base
    
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")


def check_connection() -> bool:
    """
    Test database connection.
    
    SQLAlchemy 2.0 compatible - uses text() wrapper for SQL strings.
    
    Returns:
        True if connection successful, False otherwise
        
    Example:
        >>> from src.database.connection import check_connection
        >>> if check_connection():
        ...     print("Database is ready!")
    """
    try:
        with get_db() as db:
            # SQLAlchemy 2.0 requires text() wrapper for raw SQL
            db.execute(text("SELECT 1"))
        logger.info("Database connection verified")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# Export public API
__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "drop_all_tables",
    "check_connection"
]