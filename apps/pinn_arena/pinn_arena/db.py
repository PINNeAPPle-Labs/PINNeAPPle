"""SQLite database of the PINNeAPPle Arena (SQLAlchemy, sync API): users, experiments, architecture feedback.
The file is ``<PINN_ARENA_DATA_DIR>/arena.sqlite``; it starts empty (no stored leaderboard data is shipped)."""
from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from pinn_arena.settings import DB_PATH

logger = logging.getLogger(__name__)

_DB_PATH = str(DB_PATH)
DATABASE_URL = f"sqlite:///{_DB_PATH}"

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """Yield a database session and close it on completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the tables (idempotent)."""
    from pinn_arena import orm  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at: %s", _DB_PATH)
