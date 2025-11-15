// src/hooks/useGame.ts
import { useState, useCallback, useEffect } from 'react';
import { Vector3 } from '@babylonjs/core';
import { GameState, PlayerAction, Piece } from '../game/types';
import { 
    initializeGameState, 
    calculateValidActions, 
    getForce, 
    checkWinner 
} from '../game/gameLogic';
import { gameLogger } from '../services/gameLogger';
import { Capacitor } from '@capacitor/core';
import { GAME_RULES } from '../game/constants';

export type ActionPhase = 'placement' | 'infusion' | 'movement';

export const useGame = () => {
    // Load saved game from localStorage if available, else initialize
    const [gameState, setGameState] = useState<GameState>(() => {
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
            setGameState(previousState);
            return prev.slice(0, -1);
        });
    };

    const handleAction = useCallback((action: PlayerAction) => {
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
                    if (!prev.turn.hasPlaced && 
                        prev.validPlacementVertices.includes(action.vertexId) && 
                        prev.players[prev.currentPlayerId].reinforcements > 0) {
                        
                        const newPiece: Piece = { 
                            id: `p-${Date.now()}`, 
                            player: prev.currentPlayerId 
                        };
                        nextState.vertices[action.vertexId].stack.push(newPiece);
                        nextState.players[prev.currentPlayerId].reinforcements -= 1;
                        nextState.turn.hasPlaced = true;
                    }
                    break;
                
                case 'infuse':
                    if (!prev.turn.hasInfused && 
                        prev.validInfusionVertices.includes(action.vertexId)) {
                        
                        nextState.vertices[action.vertexId].energy += 1;
                        nextState.turn.hasInfused = true;
                    }
                    break;

                case 'move':
                    if (!prev.turn.hasMoved && 
                        prev.validMoveOrigins.includes(action.fromId) && 
                        prev.validMoveTargets.includes(action.toId)) {
                        const source = nextState.vertices[action.fromId];
                        const target = nextState.vertices[action.toId];
                        if (target.stack.length > 0 && target.stack[0].player === nextState.currentPlayerId) {
                            // Stacking: combine stacks and sum energy
                            target.stack = [...target.stack, ...source.stack];
                            target.energy += source.energy;
                        } else {
                            // Move to empty or enemy vertex (shouldn't happen for enemy due to validation)
                            target.stack = source.stack;
                            target.energy = source.energy;
                        }
                        source.stack = [];
                        source.energy = 0;
                        nextState.turn.hasMoved = true;
                        nextState.selectedVertexId = null;
                    }
                    break;

                case 'attack':
                    if (action.vertexId && 
                        action.targetId && 
                        prev.validAttackTargets.includes(action.targetId)) {
                        
                        const attackerV = nextState.vertices[action.vertexId];
                        const defenderV = nextState.vertices[action.targetId];
                        const attackerForce = getForce(attackerV);
                        const defenderForce = getForce(defenderV);

                        const attackerPieces = attackerV.stack.length;
                        const defenderPieces = defenderV.stack.length;
                        const attackerEnergy = attackerV.energy;
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
                        
                        attackerV.stack = [];
                        attackerV.energy = 0;

                        // End turn
                        nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                        nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                        nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (prev.turn?.turnNumber || 0) + 1 };
                        nextState.selectedVertexId = null;
                    }
                    break;

                case 'pincer':
                    if (action.targetId && 
                        Array.isArray(action.originIds) && 
                        Object.keys(prev.validPincerTargets || {}).includes(action.targetId)) {
                        
                        const allowedOrigins = prev.validPincerTargets[action.targetId] || [];
                        const allValid = action.originIds.every(id => 
                            allowedOrigins.includes(id) && 
                            prev.vertices[id].stack.length > 0 && 
                            prev.vertices[id].stack[0].player === prev.currentPlayerId
                        );
                        
                        if (allValid) {
                            const defenderV = nextState.vertices[action.targetId];
                            const originVerts = action.originIds.map(id => nextState.vertices[id]);

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

                            originVerts.forEach(v => { 
                                v.stack = []; 
                                v.energy = 0; 
                            });

                            // End turn
                            nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                            nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                            nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (prev.turn?.turnNumber || 0) + 1 };
                            nextState.selectedVertexId = null;
                        }
                    }
                    break;

                case 'endTurn':
                    if (prev.turn.hasPlaced && prev.turn.hasInfused && prev.turn.hasMoved) {
                        nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                        nextState.players[nextState.currentPlayerId].reinforcements += GAME_RULES.reinforcementsPerTurn;
                        nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false, turnNumber: (prev.turn?.turnNumber || 0) + 1 };
                        nextState.selectedVertexId = null;
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

            // Calculate valid actions for next state
            const validActions = calculateValidActions(nextState);
            return { ...nextState, ...validActions };
        });
    }, []);

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

    return { gameState, handleAction, undo, moveHistory };
};