from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from database import Base

class Node(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String, unique=True, index=True, nullable=False)
    lab_name = Column(String, nullable=True)
    hardware_type = Column(String, nullable=True)
    
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), nullable=True)
    
    last_q_factor = Column(Float, nullable=True)
    last_alert = Column(Boolean, default=False)


class Measurement(Base):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    f0 = Column(Float)
    q_factor = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    
    # The latent fingerprint of the physical resonator (128-dim)
    signature_vector = Column(Vector(128))
