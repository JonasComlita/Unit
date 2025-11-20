# Human Game Recording API

## Overview
The backend now records human gameplay for training data collection. Games are stored in the same format as self-play data, enabling seamless integration with the ML training pipeline.

## API Endpoints

### 1. Create Game
**POST** `/api/game/create`

Start recording a new game.

**Request Body:**
```json
{
  "gameId": "unique-game-id-123",
  "initialState": { /* full game state object */ },
  "player1Device": "device-id-123",
  "player2Device": "device-id-456",  // or "AI" for vs AI games
  "gameMode": "pvp"  // 'pvp' (player vs player) or 'pva' (player vs AI)
}
```

**Response:** `201 Created`
```json
{
  "game_id": "unique-game-id-123",
  "platform": "human",
  "player1_device": "device-id-123",
  "player2_device": "device-id-456",
  "game_mode": "pvp"
}
```

---

### 2. Record Move
**POST** `/api/game/{gameId}/move`

Record a move in an ongoing game.

**Request Body:**
```json
{
  "moveNumber": 0,
  "playerId": "Player1",
  "action": {
    "type": "place",
    "vertexId": "v0"
  },
  "stateBefore": { /* optional: full game state before this move */ }
}
```

**Response:** `200 OK`
```json
{
  "status": "recorded"
}
```

**Note:** Including `stateBefore` is **highly recommended** for training quality!

---

### 3. Complete Game
**POST** `/api/game/{gameId}/complete`

Mark a game as finished.

**Request Body:**
```json
{
  "winner": "Player1",  // or "Player2" or null for draw
  "totalMoves": 42
}
```

**Response:** `200 OK`
```json
{
  "status": "completed"
}
```

---

### 4. Export Training Data (Admin)
**GET** `/api/training/export-games?limit=1000&compress=true`

Export recorded human games for training.

**Response:**
```json
{
  "count": 150,
  "games": [/* array of game objects in training format */]
}
```

---

## Frontend Integration

### Example: TypeScript/React

```typescript
// 1. When game starts
async function onGameStart(gameState: GameState, deviceId: string) {
  const gameId = crypto.randomUUID();
  
  await fetch(`${API_URL}/api/game/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gameId,
      initialState: gameState,
      player1Device: deviceId,
      player2Device: 'AI',  // or another device ID
      gameMode: 'pva'  // or 'pvp'
    })
  });
  
  return gameId;
}

// 2. After each move
async function onMoveMade(
  gameId: string, 
  moveNumber: number, 
  playerId: string, 
  action: Action,
  stateBefore: GameState
) {
  await fetch(`${API_URL}/api/game/${gameId}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      moveNumber,
      playerId,
      action,
      stateBefore  // Include for better training quality
    })
  });
}

// 3. When game ends
async function onGameEnd(gameId: string, winner: string | null, totalMoves: number) {
  await fetch(`${API_URL}/api/game/${gameId}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      winner,
      totalMoves
    })
  });
}
```

---

## Training Pipeline Integration

### Export Human Games to Parquet
```python
from game_recorder import export_games_to_training_format
import pandas as pd

# Export human games
games = export_games_to_training_format(limit=1000, compress_moves=True)

# Convert to DataFrame
df = pd.DataFrame(games)

# Save as parquet
df.to_parquet('shards/human_games/human_data.parquet')
```

### Train on Combined Data
```bash
# Option 1: Train on human games only
python -m self_play.training_pipeline train \
  --data-dir shards/human_games \
  --epochs 100 \
  --batch-size 256

# Option 2: Combine with self-play data
# Copy human games to existing shard directory
python -m self_play.training_pipeline train \
  --data-dir shards/combined_data \
  --epochs 100 \
  --batch-size 256
```

---

## Database Schema

**games** table:
- `game_id` (TEXT, PK): Unique game identifier
- `platform` (TEXT): Always 'human' for user games
- `start_time` (TIMESTAMP): Game start time
- `end_time` (TIMESTAMP): Game completion time
- `winner` (TEXT): 'Player1', 'Player2', or NULL
- `total_moves` (INTEGER): Total moves made
- `initial_state` (TEXT): JSON of initial game state
- `player1_device`, `player2_device` (TEXT): Device IDs
- `game_mode` (TEXT): 'pvp' or 'pva'

**moves** table:
- `id` (INTEGER, PK): Auto-increment
- `game_id` (TEXT, FK): References games table
- `move_number` (INTEGER): Sequential move number
- `player_id` (TEXT): 'Player1' or 'Player2'
- `action_type` (TEXT): Move type ('place', 'move', etc.)
- `action_data` (TEXT): JSON of full action object
- `state_before` (TEXT): JSON of game state before move

---

## Benefits

1. **Diverse Training Data** - Human games provide strategic patterns that greedy AI can't generate
2. **Supervised Learning** - Learn from real player decisions
3. **Quality Filtering** - Can filter by player rating, game length, etc.
4. **Seamless Integration** - Same format as self-play data
5. **Privacy-Friendly** - Only stores game states and moves, no personal data

---

## Next Steps

1. **Integrate into Frontend** - Add the 3 API calls (create, move, complete)
2. **Test Recording** - Play a few games and verify data is captured
3. **Export & Train** - Export human games and train a model
4. **Compare Performance** - Evaluate model trained on human data vs self-play
