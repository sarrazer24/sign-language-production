-- ============================================
-- ASL App - Database Schema
-- Run: psql -U postgres -d asl_db -f schema.sql
-- ============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── USERS ────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  full_name  VARCHAR(100) NOT NULL,
  email      VARCHAR(100) UNIQUE NOT NULL,
  password   VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- ── SIGNS (Dictionnaire ASL) ─────────────────
CREATE TABLE IF NOT EXISTS signs (
  id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  letter     VARCHAR(10),
  word       VARCHAR(100),
  category   VARCHAR(20) CHECK (category IN ('alphabet', 'numbers', 'phrases')),
  image_url  VARCHAR(255),
  video_url  VARCHAR(255),
  created_at TIMESTAMP DEFAULT NOW()
);

-- ── GENERATIONS ──────────────────────────────
CREATE TABLE IF NOT EXISTS generations (
  id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
  input_text TEXT NOT NULL,
  video_url  VARCHAR(255),
  status     VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'ready', 'failed')),
  created_at TIMESTAMP DEFAULT NOW()
);

-- ── NOTIFICATIONS ─────────────────────────────
CREATE TABLE IF NOT EXISTS notifications (
  id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
  title      VARCHAR(100) NOT NULL,
  message    TEXT NOT NULL,
  type       VARCHAR(20) CHECK (type IN ('success', 'info', 'error')),
  is_read    BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW()
);

-- ── SEED: Alphabet Signs ──────────────────────
INSERT INTO signs (letter, category, image_url) VALUES
  ('A', 'alphabet', '/images/asl_a.jpg'),
  ('B', 'alphabet', '/images/asl_b.jpg'),
  ('C', 'alphabet', '/images/asl_c.jpg'),
  ('D', 'alphabet', '/images/asl_d.jpg'),
  ('E', 'alphabet', '/images/asl_e.jpg'),
  ('F', 'alphabet', '/images/asl_f.jpg'),
  ('G', 'alphabet', '/images/asl_g.jpg'),
  ('H', 'alphabet', '/images/asl_h.jpg'),
  ('I', 'alphabet', '/images/asl_i.jpg')
ON CONFLICT DO NOTHING;