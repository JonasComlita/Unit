-- PostgreSQL Schema for Game Data

-- Games table
CREATE TABLE games (
    game_id VARCHAR(50) PRIMARY KEY,
    start_time BIGINT NOT NULL,
    end_time BIGINT,
    winner VARCHAR(20),
    total_moves INTEGER,
    platform VARCHAR(10),
    game_duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    -- ✅ NEW: Store initial state once per game (not per move!)
    -- Used for state reconstruction during training when state_serialization='none'
    initial_state TEXT,  -- JSON or base64-encoded binary
    -- Optional: Track which serialization strategy was used
    state_serialization_strategy VARCHAR(20) DEFAULT 'none'  -- 'none', 'json', 'binary', 'delta'
);

CREATE INDEX idx_games_winner ON games(winner);
CREATE INDEX idx_games_platform ON games(platform);
CREATE INDEX idx_games_created ON games(created_at);
CREATE INDEX idx_games_start_time ON games(start_time);  -- ✅ NEW: For time-based queries

-- Moves table (normalized for efficient querying)
CREATE TABLE moves (
    move_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) REFERENCES games(game_id) ON DELETE CASCADE,  -- ✅ Added CASCADE
    move_number INTEGER NOT NULL,
    player_id VARCHAR(20) NOT NULL,
    action_type VARCHAR(20) NOT NULL, -- 'place', 'infuse', 'move', 'attack', 'pincer', 'endTurn'
    action_data JSONB, -- Full action details
    thinking_time_ms INTEGER,
    timestamp BIGINT,
    
    -- Denormalized state info for fast querying
    pieces_player1 INTEGER,
    pieces_player2 INTEGER,
    energy_player1 INTEGER,
    energy_player2 INTEGER,
    reinforcements_player1 INTEGER,
    reinforcements_player2 INTEGER,
    
    UNIQUE(game_id, move_number)
);

CREATE INDEX idx_moves_game ON moves(game_id);
CREATE INDEX idx_moves_action_type ON moves(action_type);
CREATE INDEX idx_moves_player ON moves(player_id);  -- ✅ NEW: For player-specific queries

-- ⚠️ IMPORTANT DESIGN DECISION: game_states table
-- With state_serialization='none', we DON'T store per-move states here anymore.
-- Instead, states are reconstructed on-demand from initial_state + moves.
-- This table can be kept for legacy data or removed entirely.

-- Option 1: Keep but mark as deprecated
CREATE TABLE game_states (
    state_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) REFERENCES games(game_id) ON DELETE CASCADE,
    move_number INTEGER,
    state_hash VARCHAR(64), -- Hash of board position for duplicate detection
    state_data JSONB, -- Full compressed state (DEPRECATED - use reconstruction instead)
    evaluation FLOAT, -- Will be filled by AI later
    
    UNIQUE(game_id, move_number)
);

-- Add comment explaining the deprecation
COMMENT ON TABLE game_states IS 
'DEPRECATED: With state_serialization=none, states are reconstructed from 
games.initial_state + moves. This table is kept for backward compatibility.';

CREATE INDEX idx_states_hash ON game_states(state_hash);
CREATE INDEX idx_states_eval ON game_states(evaluation);

-- Option 2: Remove game_states entirely and use a view instead
-- DROP TABLE IF EXISTS game_states;

-- ✅ NEW: Metadata table for tracking training runs
CREATE TABLE game_metadata (
    game_id VARCHAR(50) PRIMARY KEY REFERENCES games(game_id) ON DELETE CASCADE,
    metadata JSONB,  -- Stores seed, starting_player, exploration_rate, etc.
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_game_metadata_jsonb ON game_metadata USING GIN(metadata);

-- Opening book table (like chess openings)
CREATE TABLE openings (
    opening_id SERIAL PRIMARY KEY,
    move_sequence TEXT[], -- Array of first N moves
    occurrence_count INTEGER DEFAULT 1,
    win_rate_player1 FLOAT,
    win_rate_player2 FLOAT,
    avg_game_length INTEGER,
    last_seen TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_openings_sequence ON openings USING GIN(move_sequence);

-- Player statistics (if you add user accounts later)
CREATE TABLE player_stats (
    player_id VARCHAR(50) PRIMARY KEY,
    games_played INTEGER DEFAULT 0,
    games_won INTEGER DEFAULT 0,
    avg_thinking_time_ms INTEGER,
    favorite_opening TEXT[],
    elo_rating INTEGER DEFAULT 1500,
    last_active TIMESTAMP DEFAULT NOW()
);

-- ✅ NEW: Training dataset tracking
CREATE TABLE training_datasets (
    dataset_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    game_count INTEGER,
    start_game_id VARCHAR(50),  -- First game in dataset
    end_game_id VARCHAR(50),    -- Last game in dataset
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Metadata about the dataset
    total_positions INTEGER,  -- Total training examples
    state_serialization_strategy VARCHAR(20),
    board_layout INTEGER[],  -- e.g., {3, 5, 7, 5, 3}
    game_version VARCHAR(20)
);

-- ✅ NEW: Model training runs
CREATE TABLE model_training_runs (
    run_id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES training_datasets(dataset_id),
    model_architecture VARCHAR(50),  -- 'fc', 'gcn', 'transformer'
    hyperparameters JSONB,
    training_start TIMESTAMP DEFAULT NOW(),
    training_end TIMESTAMP,
    best_loss FLOAT,
    best_epoch INTEGER,
    model_path TEXT,  -- Path to saved checkpoint
    notes TEXT
);

-- Analytics views
CREATE VIEW game_statistics AS
SELECT 
    platform,
    winner,
    COUNT(*) as game_count,
    AVG(total_moves) as avg_moves,
    AVG(game_duration_ms) as avg_duration_ms,
    MIN(total_moves) as shortest_game,
    MAX(total_moves) as longest_game
FROM games
WHERE winner IS NOT NULL
GROUP BY platform, winner;

CREATE VIEW move_frequencies AS
SELECT 
    action_type,
    COUNT(*) as frequency,
    AVG(thinking_time_ms) as avg_think_time
FROM moves
GROUP BY action_type
ORDER BY frequency DESC;

-- ✅ NEW: View for training data statistics
CREATE VIEW training_data_stats AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    state_serialization_strategy,
    COUNT(*) as games_generated,
    SUM(total_moves) as total_positions,
    AVG(total_moves) as avg_moves_per_game,
    AVG(game_duration_ms) as avg_duration_ms,
    -- Estimate storage size
    CASE 
        WHEN state_serialization_strategy = 'none' THEN 
            -- initial_state (~2KB) + moves (~100 bytes each)
            SUM(2048 + total_moves * 100)
        WHEN state_serialization_strategy = 'binary' THEN
            -- Compressed states
            SUM(total_moves * 1024)
        ELSE
            -- Full JSON states
            SUM(total_moves * 10240)
    END as estimated_bytes
FROM games
WHERE platform = 'selfplay'
GROUP BY DATE_TRUNC('day', created_at), state_serialization_strategy
ORDER BY date DESC;

-- ✅ NEW: Function to reconstruct game state at any move
-- This demonstrates how to use initial_state + moves for training
CREATE OR REPLACE FUNCTION get_state_at_move(
    p_game_id VARCHAR(50),
    p_move_number INTEGER
) RETURNS JSONB AS $$
DECLARE
    initial_state_data TEXT;
    reconstructed_state JSONB;
BEGIN
    -- Get initial state from games table
    SELECT initial_state INTO initial_state_data
    FROM games
    WHERE game_id = p_game_id;
    
    IF initial_state_data IS NULL THEN
        RAISE EXCEPTION 'Game % not found or has no initial state', p_game_id;
    END IF;
    
    -- Parse initial state
    reconstructed_state := initial_state_data::JSONB;
    
    -- TODO: Apply moves sequentially up to p_move_number
    -- This would call your game engine's apply_move logic
    -- For now, return a placeholder
    RETURN jsonb_build_object(
        'note', 'State reconstruction not yet implemented in SQL',
        'game_id', p_game_id,
        'move_number', p_move_number,
        'initial_state', reconstructed_state
    );
END;
$$ LANGUAGE plpgsql;

-- ✅ NEW: Useful queries for training pipeline

-- Query 1: Get all games in a date range for training
-- SELECT game_id, initial_state, winner, total_moves
-- FROM games 
-- WHERE platform = 'selfplay' 
--   AND created_at BETWEEN '2024-01-01' AND '2024-01-31'
-- ORDER BY start_time;

-- Query 2: Get moves for a specific game (for reconstruction)
-- SELECT move_number, action_data
-- FROM moves
-- WHERE game_id = 'selfplay_123_456'
-- ORDER BY move_number;

-- Query 3: Estimate storage savings with different strategies
-- SELECT 
--     state_serialization_strategy,
--     COUNT(*) as game_count,
--     pg_size_pretty(SUM(
--         CASE state_serialization_strategy
--             WHEN 'none' THEN 2048 + total_moves * 100
--             WHEN 'binary' THEN total_moves * 1024
--             ELSE total_moves * 10240
--         END
--     )::BIGINT) as estimated_size
-- FROM games
-- WHERE platform = 'selfplay'
-- GROUP BY state_serialization_strategy;

-- ✅ NEW: Maintenance queries

-- Clean up old test games
-- DELETE FROM games WHERE platform = 'test' AND created_at < NOW() - INTERVAL '7 days';

-- Vacuum and analyze for performance
-- VACUUM ANALYZE games;
-- VACUUM ANALYZE moves;
-- VACUUM ANALYZE game_metadata;