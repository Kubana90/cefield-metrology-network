import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# We pull DB_URL from environment, fallback to a local docker-compose default
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/cefield")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Organization(Base):
    """
    Represents a paying B2B Customer (e.g., Munich Quantum Lab, Bosch MEMS).
    Links an API key to a Stripe Customer ID for billing.
    """
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    
    # Stripe Billing details
    stripe_customer_id = Column(String, unique=True, nullable=True)
    subscription_active = Column(Boolean, default=False)
    subscription_tier = Column(String, default="freemium") # 'freemium' or 'enterprise'
    
    created_at = Column(DateTime, nullable=True)


class Node(Base):
    """
    A single edge hardware device inside an organization.
    """
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
    A single RF vector measurement recorded by a Node.
    """
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"))
    
    f0 = Column(Float)
    q_factor = Column(Float)
    
    # The 128-dimensional envelope shape
    signature_vector = Column(Vector(128))
    
    is_anomaly = Column(Boolean, default=False)


# Initialize pgvector BEFORE creating tables
def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

# Dependency to inject DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
