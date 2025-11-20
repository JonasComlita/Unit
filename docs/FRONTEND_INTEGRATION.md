# Frontend Integration Guide for Game Recording

## Quick Integration Steps

### Step 1: Update `useGame.ts` Hook

Add game recording to the `useGame` hook:

```typescript
// src/hooks/useGame.ts
import { gameRecorder } from '../services/gameRecordingService';

export const useGame = () => {
    const [gameState, setGameState] = useState<GameState>(() => {
        // ... existing initialization ...
        const initialState = initializeGameState();
        const validActions = calculateValidActions(initialState);
        
        // ✨ START RECORDING
        gameRecorder.startRecording(initialState, 'pva')  // 'pva' = player vs AI
            .catch(err => console.warn('Failed to start recording:', err));
        
        return { ...initialState, ...validActions };
    });

    const handleAction = useCallback((action: PlayerAction) => {
        setGameState(prev => {
            // ✨ RECORD MOVE (before applying it)
            if (gameRecorder.isRecording()) {
                gameRecorder.recordMove(
                    prev.currentPlayerId,
                    action,
                    prev  // State BEFORE the move
                ).catch(err => console.warn('Failed to record move:', err));
            }

            // ... existing action handling code ...
            const nextState = applyAction(prev, action);
            
            // ✨ CHECK FOR GAME END
            const winner = checkWinner(nextState);
            if (winner) {
                const totalMoves = nextState.turn.turnNumber;
                gameRecorder.completeRecording(winner, totalMoves)
                    .catch(err => console.warn('Failed to complete recording:', err));
            }

            return nextState;
        });
    }, []);

    // ... rest of hook ...
};
```

### Step 2: Add Recording Toggle UI (Optional)

Add a toggle in `UnitGame.tsx` or `MobileHUD.tsx`:

```typescript
import { gameRecorder } from '../services/gameRecordingService';

function RecordingIndicator() {
    const isRecording = gameRecorder.isRecording();
    const info = gameRecorder.getCurrentRecording();
    
    if (!isRecording) return null;
    
    return (
        <div style={{
            position: 'absolute',
            top: 10,
            right: 10,
            background: 'rgba(255, 0, 0, 0.8)',
            color: 'white',
            padding: '5px 10px',
            borderRadius: 5,
            fontSize: 12
        }}>
            ● REC ({info?.moveNumber} moves)
        </div>
    );
}
```

### Step 3: Handle Game Resets

Update any "New Game" or "Reset" buttons:

```typescript
const handleNewGame = () => {
    // Cancel current recording
    gameRecorder.cancelRecording();
    
    // Initialize new game
    const newState = initializeGameState();
    setGameState(newState);
    
    // Start new recording
    gameRecorder.startRecording(newState, 'pva')
        .catch(err => console.warn('Failed to start recording:', err));
};
```

---

## Full Example: Modified `useGame.ts`

Here's a complete minimal integration:

```typescript
// src/hooks/useGame.ts
import { useState, useCallback, useEffect } from 'react';
import { GameState, PlayerAction } from '../game/types';
import { initializeGameState, calculateValidActions, checkWinner } from '../game/gameLogic';
import { gameRecorder } from '../services/gameRecordingService';

export const useGame = () => {
    const [gameState, setGameState] = useState<GameState>(() => {
        const initialState = initializeGameState();
        const validActions = calculateValidActions(initialState);
        
        // Start recording
        gameRecorder.startRecording(initialState, 'pva')
            .catch(err => console.warn('Recording failed to start:', err));
        
        return { ...initialState, ...validActions };
    });

    const [moveHistory, setMoveHistory] = useState<GameState[]>([]);

    const handleAction = useCallback((action: PlayerAction) => {
        setGameState(prev => {
            // Record move BEFORE applying it
            if (gameRecorder.isRecording()) {
                gameRecorder.recordMove(prev.currentPlayerId, action, prev)
                    .catch(err => console.warn('Move recording failed:', err));
            }

            // Apply action (existing code)
            const nextState = applyActionLogic(prev, action);
            
            // Check for game end
            const winner = checkWinner(nextState);
            if (winner && gameRecorder.isRecording()) {
                const totalMoves = nextState.turn.turnNumber;
                gameRecorder.completeRecording(winner, totalMoves)
                    .catch(err => console.warn('Failed to complete recording:', err));
            }

            return nextState;
        });
    }, []);

    const resetGame = useCallback(() => {
        gameRecorder.cancelRecording();
        const newState = initializeGameState();
        setGameState(newState);
        setMoveHistory([]);
        
        gameRecorder.startRecording(newState, 'pva')
            .catch(err => console.warn('Recording failed to start:', err));
    }, []);

    return { gameState, handleAction, resetGame, moveHistory };
};
```

---

## Testing the Integration

### 1. Check Browser Console

You should see:
```
🎮 Started recording game: abc-123-def (pva)
✅ Game recording completed: abc-123-def, Winner: Player1
```

### 2. Check Database

```bash
# In backend directory
sqlite3 game.db "SELECT * FROM games ORDER BY start_time DESC LIMIT 5"
sqlite3 game.db "SELECT COUNT(*) FROM moves"
```

### 3. Export Training Data

```bash
# Backend
curl http://localhost:3000/api/training/export-games?limit=10
```

---

## Advanced: Multiplayer (PvP) Support

For player-vs-player games, you'll need to:

1. **Match players** (outside scope - could use WebSockets, Firebase, etc.)
2. **Share game state** between clients
3. **Record from both perspectives**:

```typescript
// Player 1's client
gameRecorder.startRecording(
    initialState,
    'pvp',  // Player vs Player
    player2DeviceId  // Other player's device ID
);

// Player 2's client (same game_id!)
gameRecorder.startRecording(
    initialState,
    'pvp',
    player1DeviceId
);
```

For now, **PvA (Player vs AI) is recommended** as it's simpler and still provides valuable training data!

---

## Troubleshooting

**Problem: No games in database**
- Check browser console for errors
- Verify API_URL is correct in gameRecordingService.ts
- Check backend is running on port 3000

**Problem: Moves not recorded**
- Ensure `stateBefore` is being passed
- Check network tab in browser dev tools
- Verify game is not ending prematurely

**Problem: Recording never completes**
- Check that `checkWinner()` is being called
- Verify winner detection logic
- Add manual completion on page unload:
  ```typescript
  useEffect(() => {
      return () => {
          if (gameRecorder.isRecording()) {
              gameRecorder.cancelRecording();
          }
      };
  }, []);
  ```

---

## Next Steps

1. ✅ Integrate recording into `useGame.ts`
2. ✅ Add visual indicator (optional)  
3. ✅ Test with a few games
4. ✅ Export data and verify quality
5. ✅ Train model on human data
6. ✅ Compare with self-play baseline
