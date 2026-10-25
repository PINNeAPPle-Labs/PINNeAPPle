"""SQLAlchemy ORM models of the PINNeAPPle Arena."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from pinn_arena.db import Base


class User(Base):
    """A user authenticated via GitHub OAuth."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String, nullable=False)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    experiments = relationship("ExperimentRecord", back_populates="user")


class ExperimentRecord(Base):
    """Persisted record of an arena experiment."""

    __tablename__ = "experiments"

    id = Column(String, primary_key=True)  # UUID from ExperimentStore
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    problem_id = Column(String, nullable=True)
    status = Column(String, default="PENDING")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    config = Column(Text, nullable=True)          # JSON: experiment config
    result_summary = Column(Text, nullable=True)  # JSON: high-level summary
    results = Column(Text, nullable=True)         # JSON: full model_results list
    model_statuses = Column(Text, nullable=True)  # JSON: per-arch training state
    logs = Column(Text, nullable=True)            # JSON: list of log strings
    current_model = Column(String, nullable=True)
    winner = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    user = relationship("User", back_populates="experiments")


class ArchitectureFeedback(Base):
    """User feedback (thumbs up/down) for a neural architecture."""

    __tablename__ = "architecture_feedback"

    id = Column(Integer, primary_key=True, index=True)
    arch_id = Column(String, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_positive = Column(Integer, nullable=False)  # 1 = thumbs up, 0 = thumbs down
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
