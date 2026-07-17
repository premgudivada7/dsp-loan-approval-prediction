"""
Database layer (SQLAlchemy ORM).

Two tables:
  * predictions      – every prediction served by the API (webapp or scheduled).
  * ingestion_stats  – per-file data-quality summary written by the ingestion DAG
                       (summary statistics only, structured for the Grafana
                       data-quality dashboard).

The connection string is read from the DATABASE_URL environment variable
(never hardcoded); a local SQLite file is used as a fallback for dev/tests.

Uses the psycopg (v3) driver, which ships wheels for Python 3.13+. Any legacy
``postgresql://`` / ``postgresql+psycopg2://`` URL is normalised to
``postgresql+psycopg://`` automatically so existing env vars keep working.
"""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./loan_mlops.db")

# Normalise the driver to psycopg3 (postgresql+psycopg://).
if DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql+psycopg2://", "postgresql+psycopg://", 1
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# check_same_thread only matters for SQLite; ignored by Postgres.
_connect_args = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(20), default="webapp", index=True)  # webapp | scheduled
    prediction = Column(String(10), nullable=False)  # Approved | Rejected
    probability = Column(Float, nullable=False)
    cibil_score = Column(Integer)
    loan_amount = Column(Float)
    income_annum = Column(Float)
    features = Column(JSON, nullable=False)  # full validated input row


class IngestionStat(Base):
    __tablename__ = "ingestion_stats"

    id = Column(Integer, primary_key=True, index=True)
    ingested_at = Column(DateTime, default=datetime.utcnow, index=True)
    file_name = Column(String(255), nullable=False)
    total_rows = Column(Integer, nullable=False)
    valid_rows = Column(Integer, nullable=False)
    invalid_rows = Column(Integer, nullable=False)
    invalid_pct = Column(Float, nullable=False)
    criticality = Column(String(10), nullable=False)  # High | Medium | Low | None
    error_summary = Column(JSON, nullable=False)  # {error_type: count}
    report_name = Column(String(255))  # GX Data Docs report file


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialised tables at {DATABASE_URL}")
