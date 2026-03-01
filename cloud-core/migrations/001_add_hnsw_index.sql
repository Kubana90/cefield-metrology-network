-- CEFIELD Migration 001: HNSW Vector Index + Schema Extensions
-- Run manually if upgrading from v2.x without full re-init
-- For fresh deploys, init_db() handles this automatically.

-- Ensure pgvector extension is active
CREATE EXTENSION IF NOT EXISTS vector;

-- Add timestamp column to existing measurements (safe for existing data)
ALTER TABLE measurements
  ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL;

-- Add pattern_type column (pre-failure lifecycle classification)
ALTER TABLE measurements
  ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(20) DEFAULT 'normal' NOT NULL;

-- Create NodeBaseline cache table
CREATE TABLE IF NOT EXISTS node_baselines (
    id          SERIAL PRIMARY KEY,
    node_id     INTEGER UNIQUE NOT NULL REFERENCES nodes(id),
    mean_q      FLOAT,
    std_q       FLOAT,
    slope_per_day FLOAT,
    n_samples   INTEGER DEFAULT 0,
    updated_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_node_baselines_node_id ON node_baselines(node_id);

-- General HNSW index (all measurements)
-- Replaces sequential O(n) scan with O(log n) approximate nearest neighbour
CREATE INDEX IF NOT EXISTS measurements_vector_hnsw_idx
    ON measurements
    USING hnsw (signature_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

-- Dedicated high-precision index for pre-failure patterns only
-- Higher m + ef_construction = better recall for critical predictive queries
CREATE INDEX IF NOT EXISTS measurements_precursor_hnsw_idx
    ON measurements
    USING hnsw (signature_vector vector_l2_ops)
    WITH (m = 32, ef_construction = 128)
    WHERE pattern_type IN ('precursor_72h', 'precursor_24h');

-- Pattern type index for fast filtering
CREATE INDEX IF NOT EXISTS idx_measurements_pattern_type
    ON measurements(pattern_type)
    WHERE pattern_type != 'normal';

-- Timestamp index for baseline window queries
CREATE INDEX IF NOT EXISTS idx_measurements_node_ts
    ON measurements(node_id, timestamp DESC);
