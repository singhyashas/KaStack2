-- ============================================================
-- KaStack Hackathon — Checkpoint & Events Schema
-- ============================================================

-- 1. TOPIC CHECKPOINTS
CREATE TABLE topic_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) DEFAULT NULL,
    topic_id INT NOT NULL,
    start_msg_idx INT NOT NULL,
    end_msg_idx INT NOT NULL,
    summary TEXT NOT NULL,
    keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. 100-MESSAGE CHECKPOINTS
CREATE TABLE message_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) DEFAULT NULL,
    checkpoint_id INT NOT NULL,
    start_msg_idx INT NOT NULL,
    end_msg_idx INT NOT NULL,
    message_count INT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 3. DAY-WISE CHECKPOINTS (encrypted, backed up to R2)
CREATE TABLE day_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) DEFAULT NULL,
    day_index INT NOT NULL,
    r2_backup_key TEXT NOT NULL,         -- path/key in Cloudflare R2
    encryption_iv TEXT NOT NULL,         -- base64 IV used for AES encryption
    summary TEXT,
    message_count INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 4. MOOD CHECKPOINTS
CREATE TABLE mood_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) DEFAULT NULL,
    day_index INT NOT NULL,
    segment_start_idx INT NOT NULL,
    segment_end_idx INT NOT NULL,
    mood_label TEXT NOT NULL,             -- e.g. 'happy', 'anxious'
    mood_score FLOAT NOT NULL,            -- confidence/intensity score
    tone_label TEXT,                      -- e.g. 'casual', 'formal'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 5. EVENTS MEMORY (discrete events table)
CREATE TABLE events_memory (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) DEFAULT NULL,
    event_type TEXT NOT NULL,             -- e.g. 'mentioned_grocery_list', 'mood_drop'
    description TEXT NOT NULL,
    day_index INT,
    msg_idx INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 6. QUERY LOGS (for router decision tracing — useful for Opik cross-reference)
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) DEFAULT NULL,
    query_text TEXT NOT NULL,
    route_chosen TEXT NOT NULL,           -- 'semantic' | 'keyword' | 'both'
    route_reason TEXT,
    sources_used JSONB DEFAULT '[]',
    answer TEXT,
    latency_ms FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for common lookups
CREATE INDEX idx_topic_checkpoints_user ON topic_checkpoints(user_id);
CREATE INDEX idx_message_checkpoints_user ON message_checkpoints(user_id);
CREATE INDEX idx_day_checkpoints_user ON day_checkpoints(user_id);
CREATE INDEX idx_mood_checkpoints_user ON mood_checkpoints(user_id);
CREATE INDEX idx_events_memory_user ON events_memory(user_id);
CREATE INDEX idx_events_memory_type ON events_memory(event_type);
CREATE INDEX idx_query_logs_user ON query_logs(user_id);

-- Enable Row Level Security (RLS) — users can only see their own data
ALTER TABLE topic_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE day_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE mood_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE events_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies: users can only access their own rows
CREATE POLICY "Users access own topic checkpoints" ON topic_checkpoints
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users access own message checkpoints" ON message_checkpoints
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users access own day checkpoints" ON day_checkpoints
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users access own mood checkpoints" ON mood_checkpoints
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users access own events" ON events_memory
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users access own query logs" ON query_logs
    FOR ALL USING (auth.uid() = user_id);
