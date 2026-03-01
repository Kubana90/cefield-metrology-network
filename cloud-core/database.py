import os
from datetime import UTC, datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/cefield")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Organization(Base):
    """
    A paying B2B customer (e.g., Munich Quantum Lab, Bosch MEMS).
    Links API key to Stripe Customer ID for billing.
    """

    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    stripe_customer_id = Column(String, unique=True, nullable=True)
    subscription_active = Column(Boolean, default=False)
    subscription_tier = Column(String, default="freemium")  # 'freemium' | 'enterprise'
    created_at = Column(DateTime, nullable=True)


class Node(Base):
    """A single edge hardware device inside an organization."""

    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    node_id = Column(String, unique=True, index=True)
    lab_name = Column(String, nullable=True)
    hardware_type = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    last_q_factor = Column(Float, nullable=True)
    last_alert = Column(Boolean, default=False)


class Measurement(Base):
    """
    A single RF vector measurement from a Node.
    pattern_type encodes position in failure lifecycle:
      'normal'        — routine measurement
      'precursor_72h' — taken 24-72h before a confirmed failure
      'precursor_24h' — taken 0-24h before a confirmed failure (highest signal)
      'failure'       — the failure event itself
    """

    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"))
    f0 = Column(Float)
    q_factor = Column(Float)
    signature_vector = Column(Vector(128))
    is_anomaly = Column(Boolean, default=False)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    pattern_type = Column(String(20), default="normal", nullable=False)


class NodeBaseline(Base):
    """Cached statistical baseline per node — updated on every measurement."""

    __tablename__ = "node_baselines"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), unique=True, index=True)
    mean_q = Column(Float, nullable=True)
    std_q = Column(Float, nullable=True)
    slope_per_day = Column(Float, nullable=True)  # Negative = degrading
    n_samples = Column(Integer, default=0)
    updated_at = Column(DateTime, nullable=True)


def _create_indexes():
    """Create HNSW index for fast O(log n) pgvector similarity search."""
    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS measurements_vector_hnsw_idx
            ON measurements
            USING hnsw (signature_vector vector_l2_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        )
        # Dedicated high-quality index for pre-failure patterns only
        conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS measurements_precursor_hnsw_idx
            ON measurements
            USING hnsw (signature_vector vector_l2_ops)
            WITH (m = 32, ef_construction = 128)
            WHERE pattern_type IN ('precursor_72h', 'precursor_24h')
        """)
        )
        conn.commit()


def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    _create_indexes()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
