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
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_games_winner ON games(winner);
CREATE INDEX idx_games_platform ON games(platform);
CREATE INDEX idx_games_created ON games(created_at);

-- Moves table (normalized for efficient querying)
CREATE TABLE moves (
    move_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) REFERENCES games(game_id),
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

-- Game states table (for training data)
CREATE TABLE game_states (
    state_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) REFERENCES games(game_id),
    move_number INTEGER,
    state_hash VARCHAR(64), -- Hash of board position for duplicate detection
    state_data JSONB NOT NULL, -- Full compressed state
    evaluation FLOAT, -- Will be filled by AI later
    
    UNIQUE(game_id, move_number)
);

CREATE INDEX idx_states_hash ON game_states(state_hash);
CREATE INDEX idx_states_eval ON game_states(evaluation);

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
