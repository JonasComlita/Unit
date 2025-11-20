// src/hooks/useGame.ts
import { useState, useCallback, useEffect } from 'react';
import { Vector3 } from '@babylonjs/core';
import { GameState, PlayerAction, Piece } from '../game/types';
import {
    initializeGameState,
    calculateValidActions,
    getForce,
    checkWinner,
    resolveCombat,
    isOccupied
} from '../game/gameLogic';
import { gameLogger } from '../services/gameLogger';
import { Capacitor } from '@capacitor/core';
import { GAME_RULES } from '../game/constants';
import { multiplayerService, MatchInfo, OpponentMove } from '../services/multiplayerService';

export type ActionPhase = 'placement' | 'infusion' | 'movement';

export const useGame = (matchInfo?: MatchInfo | null) => {
    // Load saved game from localStorage if available, else initialize
    const [gameState, setGameState] = useState<GameState>(() => {
        // If multiplayer, always start fresh
        if (matchInfo) {
            const initialState = initializeGameState();
            // If we are Player 1, we initialize the game state on the server
            if (matchInfo.playerId === 'Player1') {
                multiplayerService.initializeGameState(initialState);
            }
            const validActions = calculateValidActions(initialState);
            // Start logging
            const platform = Capacitor.isNativePlatform()
                ? Capacitor.getPlatform() as 'ios' | 'android'
                : 'web';
            gameLogger.startGame(platform);
            return { ...initialState, ...validActions };
        }

        try {
            const raw = localStorage.getItem('currentGame');
            if (raw) {
                const parsed = JSON.parse(raw) as GameState & { vertices: Record<string, any> };
                // Rehydrate Vector3 positions
                Object.values(parsed.vertices).forEach((v: any) => {
                    v.position = new Vector3(v.position.x, v.position.y, v.position.z);
                });
                // Start logging
                const platform = Capacitor.isNativePlatform()
                    ? Capacitor.getPlatform() as 'ios' | 'android'
                    : 'web';
                gameLogger.startGame(platform);
                const validActions = calculateValidActions(parsed as GameState);
                return { ...(parsed as GameState), ...validActions };
            }
        } catch (e) {
            console.warn('Failed to load saved game, initializing new game', e);
        }

        const initialState = initializeGameState();
        const validActions = calculateValidActions(initialState);
        // Start logging
        const platform = Capacitor.isNativePlatform()
            ? Capacitor.getPlatform() as 'ios' | 'android'
            : 'web';
        gameLogger.startGame(platform);
        return { ...initialState, ...validActions };
    });

    // History for undo
    const [moveHistory, setMoveHistory] = useState<GameState[]>([]);

    // Helper to deep-clone GameState while preserving Vector3
    const cloneGameState = (s: GameState): GameState => ({
        ...s,
        vertices: Object.fromEntries(Object.entries(s.vertices).map(([id, v]) => [
            id,
            {
                ...v,
                position: v.position.clone(),
                stack: v.stack.map(p => ({ ...p })),
                adjacencies: [...v.adjacencies]
            }
        ])),
        players: { ...s.players },
        currentPlayerId: s.currentPlayerId,
        turn: { ...s.turn },
        homeCorners: { Player1: [...s.homeCorners.Player1], Player2: [...s.homeCorners.Player2] },
        winner: s.winner,
        selectedVertexId: s.selectedVertexId,
        validPlacementVertices: [...s.validPlacementVertices],
        validInfusionVertices: [...s.validInfusionVertices],
        validAttackTargets: [...s.validAttackTargets],
        validPincerTargets: { ...s.validPincerTargets },
        validMoveOrigins: [...s.validMoveOrigins],
        validMoveTargets: [...s.validMoveTargets],
    });

    // Undo function exposed to UI
    const undo = () => {
        setMoveHistory(prev => {
            if (prev.length === 0) return prev;
            const previousState = prev[prev.length - 1];

            // Only allow undo if the previous state was the same player
            // (prevents undoing opponent's moves)
            if (previousState.currentPlayerId !== gameState.currentPlayerId) {
                console.log('Cannot undo opponent\'s move');
                return prev;
            }

            setGameState(previousState);
            return prev.slice(0, -1);
        });
    };

    const handleAction = useCallback((action: PlayerAction, fromMultiplayer = false) => {
        // Multiplayer check: prevent moves if not our turn (unless it's an update from the server)
        if (matchInfo && !fromMultiplayer) {
            if (gameState.currentPlayerId !== matchInfo.playerId) {
                console.log('Not your turn!');
                return;
            }
        }

        setGameState(prev => {
            // Push snapshot for undo (limit history to 50)
            try {
                setMoveHistory(h => {
                    const next = [...h, cloneGameState(prev)];
                    const MAX = 50;
                    if (next.length > MAX) next.splice(0, next.length - MAX);
                    return next;
                });
            } catch (e) {
                console.warn('Failed to push move history', e);
            }

            // FIXED: Proper deep copy that preserves Vector3 objects
            const nextState: GameState = {
                vertices: Object.fromEntries(
                    Object.entries(prev.vertices).map(([id, vertex]) => [
                        id,
                        {
                            ...vertex,
                            position: vertex.position.clone(), // Proper Vector3 clone
                            stack: [...vertex.stack.map(p => ({ ...p }))],
                            adjacencies: [...vertex.adjacencies]
                        }
                    ])
                ),
                players: {
                    Player1: { ...prev.players.Player1 },
                    Player2: { ...prev.players.Player2 }
                },
                currentPlayerId: prev.currentPlayerId,
                turn: { ...prev.turn },
                homeCorners: {
                    Player1: [...prev.homeCorners.Player1],
                    Player2: [...prev.homeCorners.Player2]
                },
                winner: prev.winner,
                selectedVertexId: prev.selectedVertexId,
                validPlacementVertices: [...prev.validPlacementVertices],
                validInfusionVertices: [...prev.validInfusionVertices],
                validAttackTargets: [...prev.validAttackTargets],
                validPincerTargets: { ...prev.validPincerTargets },
                validMoveOrigins: [...prev.validMoveOrigins],
                validMoveTargets: [...prev.validMoveTargets],
            };

            switch (action.type) {
                case 'select':
                    nextState.selectedVertexId = action.vertexId;
                    break;

                case 'place':
                    // Validate place directly
                    if (!prev.turn.hasPlaced &&
                        prev.players[prev.currentPlayerId].reinforcements > 0 &&
                        action.vertexId) {

                        // Check if vertex is a home corner for current player
                        const isHomeCorner = prev.homeCorners[prev.currentPlayerId].includes(action.vertexId);

                        if (isHomeCorner) {
                            const newPiece: Piece = {
                                id: `p-${Date.now()}`, player: prev.currentPlayerId
                            };
                            nextState.vertices[action.vertexId].stack.push(newPiece);
                            nextState.players[prev.currentPlayerId].reinforcements -= 1;
                            nextState.turn.hasPlaced = true;
                        } else {
                            console.log('Place validation failed: not a home corner');
                        }
                    }
                    break;

                case 'infuse':
                    // Validate infuse directly
                    if (!prev.turn.hasInfused && action.vertexId) {
                        const vertex = nextState.vertices[action.vertexId];

                        // Check if vertex has friendly pieces
                        const hasFriendlyPieces = vertex && vertex.stack.length > 0 &&
                            vertex.stack[0].player === prev.currentPlayerId;

                        // Check if energy won't exceed force cap after infusion
                        const currentForce = getForce(vertex);
                        const potentialForce = (vertex.stack.length * (vertex.energy + 1)) /
                            (vertex.layer !== undefined ? [1.0, 2.0, 3.0, 2.0, 1.0][vertex.layer] : 1.0);
                        const withinForceCap = potentialForce <= GAME_RULES.forceCapMax;

                        if (hasFriendlyPieces && withinForceCap) {
                            nextState.vertices[action.vertexId].energy += 1;
                            nextState.turn.hasInfused = true;
                        } else {
                            console.log('Infuse validation failed:', { hasFriendlyPieces, withinForceCap });
                        }
                    }
                    break;

                case 'move':
                    // Validate move directly instead of relying on validMoveTargets
                    // (which requires a selected vertex that AI doesn't set)
                    if (!prev.turn.hasMoved && action.fromId && action.toId) {
                        const source = nextState.vertices[action.fromId];
                        const target = nextState.vertices[action.toId];

                        // Check if source has pieces and belongs to current player
                        const sourceValid = source && source.stack.length > 0 &&
                            source.stack[0].player === prev.currentPlayerId;

                        // Check if target is adjacent to source
                        const isAdjacent = source && source.adjacencies.includes(action.toId);

                        // Check if target is not enemy-occupied
                        const targetNotEnemy = !target.stack.length ||
                            target.stack[0].player === prev.currentPlayerId;

                        // Check if source meets occupancy requirements for target layer
                        const meetsOccupancy = isOccupied(source, target.layer);

                        if (sourceValid && isAdjacent && targetNotEnemy && meetsOccupancy) {
                            if (target.stack.length > 0 && target.stack[0].player === nextState.currentPlayerId) {
                                // Stacking: combine stacks and sum energy
                                target.stack = [...target.stack, ...source.stack];
                                target.energy += source.energy;
                            } else {
                                // Move to empty vertex
                                target.stack = source.stack;
                                target.energy = source.energy;
                            }
                            source.stack = [];
                            source.energy = 0;
                            nextState.turn.hasMoved = true;
                            nextState.selectedVertexId = null;
                        } else {
                            console.log('Move validation failed:', {
                                sourceValid, isAdjacent, targetNotEnemy, meetsOccupancy,
                                fromId: action.fromId, toId: action.toId
                            });
                        }
                    }
                    break;

                case 'attack':
                    // Validate attack directly
                    if (action.vertexId && action.targetId) {
                        const attackerV = nextState.vertices[action.vertexId];
                        const defenderV = nextState.vertices[action.targetId];

                        // Check if attacker has friendly pieces and is occupied
                        const attackerValid = attackerV && attackerV.stack.length > 0 &&
                            attackerV.stack[0].player === prev.currentPlayerId &&
                            isOccupied(attackerV);

                        // Check if target is adjacent
                        const isAdjacent = attackerV && attackerV.adjacencies.includes(action.targetId);

                        // Check if defender has enemy pieces
                        const defenderIsEnemy = defenderV && defenderV.stack.length > 0 &&
                            defenderV.stack[0].player !== prev.currentPlayerId;

                        if (attackerValid && isAdjacent && defenderIsEnemy) {
                            // Capture owner before clearing stack
                            const defenderOwner = defenderV.stack[0]?.player;

                            const result = resolveCombat(attackerV, defenderV);

                            if (result.outcome === 'attacker_win') {
                                // Attacker moves to defender vertex
                                defenderV.stack = [];
                                for (let i = 0; i < result.attacker.pieces; i++) {
                                    defenderV.stack.push({
                                        id: `p-conquer-${i}-${Date.now()}`,
                                        player: prev.currentPlayerId
                                    });
                                }
                                defenderV.energy = result.attacker.energy;

                                // Source becomes empty
                                attackerV.stack = [];
                                attackerV.energy = 0;
                            } else if (result.outcome === 'defender_win') {
                                // Defender stays, attacker destroyed
                                defenderV.stack = [];
                                for (let i = 0; i < result.defender.pieces; i++) {
                                    defenderV.stack.push({
                                        id: `p-defend-${i}-${Date.now()}`,
                                        player: defenderOwner
                                    });
                                }
                                defenderV.energy = result.defender.energy;

                                // Attacker destroyed
                                attackerV.stack = [];
                                attackerV.energy = 0;
                            } else {
                                // Draw: Both stay at original positions with reduced stats
                                // Update Attacker at Source
                                if (result.attacker.pieces > 0) {
                                    attackerV.stack = [];
                                    for (let i = 0; i < result.attacker.pieces; i++) {
                                        attackerV.stack.push({
                                            id: `p-draw-att-${i}-${Date.now()}`,
                                            player: prev.currentPlayerId
                                        });
                                    }
                                    attackerV.energy = result.attacker.energy;
                                } else {
                                    attackerV.stack = [];
                                    attackerV.energy = 0;
                                }

                                // Update Defender at Target
                                if (result.defender.pieces > 0) {
                                    defenderV.stack = [];
                                    for (let i = 0; i < result.defender.pieces; i++) {
                                        defenderV.stack.push({
                                            id: `p-draw-def-${i}-${Date.now()}`,
                                            player: defenderOwner
                                        });
                                    }
                                    defenderV.energy = result.defender.energy;
                                } else {
                                    defenderV.stack = [];
                                    defenderV.energy = 0;
                                }
                            }

                            // End turn
                            nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                            nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                            nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (prev.turn?.turnNumber || 0) + 1 };
                            nextState.selectedVertexId = null;
                            setMoveHistory([]);
                        } else {
                            console.log('Attack validation failed:', { attackerValid, isAdjacent, defenderIsEnemy });
                        }
                    }
                    break;

                case 'pincer':
                    // Determine origin IDs: use provided ones, or fallback to all valid origins for this target
                    const targetId = action.targetId;
                    const validPincerTargets = prev.validPincerTargets || {};

                    let originIds = action.originIds;
                    if (!originIds && targetId && validPincerTargets[targetId]) {
                        originIds = validPincerTargets[targetId];
                    }

                    if (targetId &&
                        Array.isArray(originIds) &&
                        Object.keys(validPincerTargets).includes(targetId)) {

                        const allowedOrigins = validPincerTargets[targetId] || [];

                        const allValid = originIds.every(id =>
                            allowedOrigins.includes(id) &&
                            prev.vertices[id].stack.length > 0 &&
                            prev.vertices[id].stack[0].player === prev.currentPlayerId
                        );

                        if (allValid) {
                            const defenderV = nextState.vertices[targetId];
                            const originVerts = originIds.map(id => nextState.vertices[id]);

                            let attackerForce = originVerts.map(getForce).reduce((a, b) => a * b, 1);
                            attackerForce = Math.min(attackerForce, GAME_RULES.forceCapMax);
                            const defenderForce = getForce(defenderV);

                            const attackerPieces = originVerts.reduce((acc, v) => acc + v.stack.length, 0);
                            const defenderPieces = defenderV.stack.length;
                            const attackerEnergy = originVerts.reduce((acc, v) => acc + v.energy, 0);
                            const defenderEnergy = defenderV.energy;

                            const newPieces = Math.abs(attackerPieces - defenderPieces);
                            const newEnergy = Math.abs(attackerEnergy - defenderEnergy);

                            if (attackerForce > defenderForce) {
                                defenderV.stack = [];
                                for (let i = 0; i < newPieces; i++) {
                                    defenderV.stack.push({
                                        id: `p-conquer-${i}-${Date.now()}`,
                                        player: prev.currentPlayerId
                                    });
                                }
                                defenderV.energy = newEnergy;
                            } else {
                                const defenderOwner = defenderV.stack[0]?.player ?? prev.currentPlayerId;
                                defenderV.stack = [];
                                for (let i = 0; i < newPieces; i++) {
                                    defenderV.stack.push({
                                        id: `p-defend-${i}-${Date.now()}`,
                                        player: defenderOwner
                                    });
                                }
                                defenderV.energy = newEnergy;
                            }

                            // Clear all origin vertices
                            originVerts.forEach(v => {
                                v.stack = [];
                                v.energy = 0;
                            });

                            // End turn
                            nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                            nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                            nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (prev.turn?.turnNumber || 0) + 1 };
                            nextState.selectedVertexId = null;
                            setMoveHistory([]);
                        }
                    }
                    break;

                case 'endTurn':
                    if (prev.turn.hasPlaced && prev.turn.hasInfused && prev.turn.hasMoved) {
                        nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                        nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                        nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (prev.turn?.turnNumber || 0) + 1 };
                        nextState.selectedVertexId = null;
                        setMoveHistory([]);
                    }
                    break;
            }

            // Log the move
            gameLogger.logMove(action, prev.currentPlayerId);

            // Check for winner
            nextState.winner = checkWinner(nextState);

            // End game if there's a winner
            if (nextState.winner) {
                gameLogger.endGame(nextState.winner);
            }

            // Auto-end turn if all actions are complete
            if (nextState.turn.hasPlaced && nextState.turn.hasInfused && nextState.turn.hasMoved && !nextState.winner) {
                nextState.currentPlayerId = nextState.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (nextState.turn?.turnNumber || 0) + 1 };
                nextState.selectedVertexId = null;
                // Clear move history when turn ends
                setMoveHistory([]);
            }

            // Calculate valid actions for next state
            const validActions = calculateValidActions(nextState);

            // If multiplayer and this was a local move, send to server
            if (matchInfo && !fromMultiplayer) {
                multiplayerService.makeMove(action, prev);

                // If game over, notify server
                if (nextState.winner) {
                    multiplayerService.gameOver(nextState.winner, nextState.turn.turnNumber || 0);
                }
            }

            return { ...nextState, ...validActions };
        });
    }, [matchInfo, gameState.currentPlayerId]);

    // Sync pending games on mount
    useEffect(() => {
        gameLogger.syncPendingGames();
    }, []);

    // Persist gameState to localStorage on change (serialize Vector3)
    useEffect(() => {
        try {
            const serializable = {
                ...gameState,
                vertices: Object.fromEntries(
                    Object.entries(gameState.vertices).map(([id, v]) => [
                        id,
                        { ...v, position: { x: v.position.x, y: v.position.y, z: v.position.z } }
                    ])
                )
            };
            localStorage.setItem('currentGame', JSON.stringify(serializable));
        } catch (e) {
            console.warn('Failed to save game to localStorage', e);
        }
    }, [gameState]);

    // Multiplayer Event Listeners
    useEffect(() => {
        if (!matchInfo) return;

        const handleOpponentMove = (data: OpponentMove) => {
            handleAction(data.action, true);
        };

        const handleGameEnded = (data: { winner: string | null }) => {
            // Handle game end if needed (already handled by state update but good for sync)
            console.log('Multiplayer game ended:', data.winner);
        };

        multiplayerService.on('opponent_move', handleOpponentMove);
        multiplayerService.on('game_ended', handleGameEnded);

        return () => {
            multiplayerService.off('opponent_move', handleOpponentMove);
            multiplayerService.off('game_ended', handleGameEnded);
        };
    }, [matchInfo, handleAction]);

    // Difficulty state
    const [difficulty, setDifficulty] = useState<'very_easy' | 'easy' | 'medium' | 'hard' | 'very_hard'>('medium');
    const [isAiThinking, setIsAiThinking] = useState(false);

    // AI Turn Logic
    useEffect(() => {
        const performAiMove = async () => {
            // AI is Player 2 (Only in Single Player)
            if (!matchInfo && gameState.currentPlayerId === 'Player2' && !gameState.winner && !isAiThinking) {
                setIsAiThinking(true);
                try {
                    // Small delay for better UX
                    await new Promise(resolve => setTimeout(resolve, 500));

                    // Import apiClient dynamically to avoid circular dependencies if any
                    const { apiClient } = await import('../services/apiClient');

                    const action = await apiClient.getAIMove(gameState, difficulty as any);

                    console.log('AI returned action:', action);
                    if (action) {
                        console.log('Executing AI action:', action.type, action);
                        handleAction(action);
                    }
                } catch (error) {
                    console.error("AI Move Error:", error);
                    // Fallback to end turn if AI fails to prevent stuck game
                    handleAction({ type: 'endTurn' });
                } finally {
                    setIsAiThinking(false);
                }
            }
        };

        performAiMove();
    }, [gameState, difficulty, handleAction]); // Added gameState to deps to trigger on turn updates

    return { gameState, handleAction, undo, moveHistory, setDifficulty, isAiThinking };
};