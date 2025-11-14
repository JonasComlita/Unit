// unitEngine.ts - Classical game engine using minimax + alpha-beta pruning

import { GameState, PlayerAction, PlayerId, Vertex } from './types';

interface EvaluatedMove {
    action: PlayerAction;
    score: number;
    principalVariation: PlayerAction[]; // Best continuation
}

class UnitGameEngine {
    private transpositionTable: Map<string, { score: number; depth: number }> = new Map();
    private nodesEvaluated = 0;

    /**
     * Find the best move for the current position
     * @param state Current game state
     * @param depth Search depth (higher = stronger but slower)
     * @returns Best move with evaluation
     */
    getBestMove(state: GameState, depth: number = 4): EvaluatedMove {
        this.nodesEvaluated = 0;
        const startTime = Date.now();

        const bestMove = this.minimax(state, depth, -Infinity, Infinity, true);
        
        const elapsed = Date.now() - startTime;
        console.log(`Evaluated ${this.nodesEvaluated} positions in ${elapsed}ms`);
        console.log(`Nodes per second: ${Math.round(this.nodesEvaluated / (elapsed / 1000))}`);

        return bestMove;
    }

    private minimax(
        state: GameState, 
        depth: number, 
        alpha: number, 
        beta: number, 
        isMaximizing: boolean
    ): EvaluatedMove {
        this.nodesEvaluated++;

        // Terminal conditions
        if (depth === 0 || state.winner) {
            return {
                action: { type: 'endTurn' },
                score: this.evaluatePosition(state),
                principalVariation: []
            };
        }

        // Check transposition table
        const stateHash = this.hashState(state);
        const cached = this.transpositionTable.get(stateHash);
        if (cached && cached.depth >= depth) {
            return {
                action: { type: 'endTurn' },
                score: cached.score,
                principalVariation: []
            };
        }

        const legalMoves = this.generateLegalMoves(state);
        
        if (legalMoves.length === 0) {
            return {
                action: { type: 'endTurn' },
                score: this.evaluatePosition(state),
                principalVariation: []
            };
        }

        // Order moves for better pruning (best moves first)
        const orderedMoves = this.orderMoves(legalMoves, state);

        let bestMove: EvaluatedMove = {
            action: orderedMoves[0],
            score: isMaximizing ? -Infinity : Infinity,
            principalVariation: []
        };

        for (const move of orderedMoves) {
            const newState = this.applyMove(state, move);
            const evaluation = this.minimax(newState, depth - 1, alpha, beta, !isMaximizing);

            if (isMaximizing) {
                if (evaluation.score > bestMove.score) {
                    bestMove = {
                        action: move,
                        score: evaluation.score,
                        principalVariation: [move, ...evaluation.principalVariation]
                    };
                }
                alpha = Math.max(alpha, evaluation.score);
            } else {
                if (evaluation.score < bestMove.score) {
                    bestMove = {
                        action: move,
                        score: evaluation.score,
                        principalVariation: [move, ...evaluation.principalVariation]
                    };
                }
                beta = Math.min(beta, evaluation.score);
            }

            // Alpha-beta pruning
            if (beta <= alpha) {
                break;
            }
        }

        // Store in transposition table
        this.transpositionTable.set(stateHash, { score: bestMove.score, depth });

        return bestMove;
    }

    /**
     * Evaluate the current position (positive = good for current player)
     */
    private evaluatePosition(state: GameState): number {
        if (state.winner === state.currentPlayerId) return 10000;
        if (state.winner) return -10000;

        let score = 0;
        const player = state.currentPlayerId;
        const opponent = player === 'Player1' ? 'Player2' : 'Player1';

        // Material: Count pieces and energy
        let playerPieces = 0, opponentPieces = 0;
        let playerEnergy = 0, opponentEnergy = 0;
        let playerForce = 0, opponentForce = 0;
        let playerOccupiedVertices = 0, opponentOccupiedVertices = 0;

        Object.values(state.vertices).forEach(vertex => {
            if (vertex.stack.length > 0) {
                const owner = vertex.stack[0].player;
                const force = this.getForce(vertex);
                
                if (owner === player) {
                    playerPieces += vertex.stack.length;
                    playerEnergy += vertex.energy;
                    playerForce += force;
                    if (this.isOccupied(vertex)) playerOccupiedVertices++;
                } else {
                    opponentPieces += vertex.stack.length;
                    opponentEnergy += vertex.energy;
                    opponentForce += force;
                    if (this.isOccupied(vertex)) opponentOccupiedVertices++;
                }
            }
        });

        // Weighted evaluation
        score += (playerPieces - opponentPieces) * 10;        // Piece count
        score += (playerEnergy - opponentEnergy) * 15;        // Energy advantage
        score += (playerForce - opponentForce) * 20;          // Force is most important
        score += (playerOccupiedVertices - opponentOccupiedVertices) * 30; // Territory control

        // Bonus for controlling home corners
        const opponentCorners = state.homeCorners[opponent];
        const controlledCorners = opponentCorners.filter(cornerId => {
            const vertex = state.vertices[cornerId];
            return vertex.stack.length > 0 && vertex.stack[0].player === player;
        }).length;
        score += controlledCorners * 500; // Huge bonus for corner control

        // Mobility: Number of valid moves available
        const mobility = this.generateLegalMoves(state).length;
        score += mobility * 2;

        return score;
    }

    /**
     * Generate all legal moves for current player
     */
    private generateLegalMoves(state: GameState): PlayerAction[] {
        const moves: PlayerAction[] = [];

        // Placement moves
        if (!state.turn.hasPlaced && state.players[state.currentPlayerId].reinforcements > 0) {
            state.validPlacementVertices.forEach(vId => {
                moves.push({ type: 'place', vertexId: vId });
            });
        }

        // Infusion moves
        if (!state.turn.hasInfused) {
            state.validInfusionVertices.forEach(vId => {
                moves.push({ type: 'infuse', vertexId: vId });
            });
        }

        // Movement moves
        if (!state.turn.hasMoved) {
            state.validMoveOrigins.forEach(fromId => {
                const vertex = state.vertices[fromId];
                vertex.adjacencies.forEach(toId => {
                    const target = state.vertices[toId];
                    if (target.stack.length === 0 && this.isOccupied(vertex, target.layer)) {
                        moves.push({ type: 'move', fromId, toId });
                    }
                });
            });
        }

        // Attack moves
        Object.values(state.vertices).forEach(vertex => {
            if (vertex.stack[0]?.player === state.currentPlayerId && this.isOccupied(vertex)) {
                vertex.adjacencies.forEach(targetId => {
                    const target = state.vertices[targetId];
                    if (target.stack.length > 0 && target.stack[0].player !== state.currentPlayerId) {
                        moves.push({ type: 'attack', vertexId: vertex.id, targetId });
                    }
                });
            }
        });

        // Pincer moves
        Object.entries(state.validPincerTargets).forEach(([targetId, originIds]) => {
            if (originIds.length >= 2) {
                moves.push({ type: 'pincer', targetId, originIds });
            }
        });

        // End turn (always available if mandatory actions done)
        if (state.turn.hasPlaced && state.turn.hasInfused && state.turn.hasMoved) {
            moves.push({ type: 'endTurn' });
        }

        return moves;
    }

    /**
     * Order moves for better alpha-beta pruning
     * Most promising moves first
     */
    private orderMoves(moves: PlayerAction[], state: GameState): PlayerAction[] {
        return moves.sort((a, b) => {
            // Prioritize attacks and pincers
            if (a.type === 'attack' || a.type === 'pincer') return -1;
            if (b.type === 'attack' || b.type === 'pincer') return 1;
            
            // Then infusion (builds strength)
            if (a.type === 'infuse') return -1;
            if (b.type === 'infuse') return 1;
            
            // Then movement
            if (a.type === 'move') return -1;
            if (b.type === 'move') return 1;
            
            return 0;
        });
    }

    /**
     * Apply a move to state (returns new state, doesn't mutate)
     */
    private applyMove(state: GameState, action: PlayerAction): GameState {
        // Deep clone state
        const newState: GameState = JSON.parse(JSON.stringify(state));
        
        // Apply action logic (simplified - reuse your existing handleAction logic)
        // ... (you'd integrate your full game logic here)
        
        return newState;
    }

    /**
     * Hash game state for transposition table
     */
    private hashState(state: GameState): string {
        // Simple hash: serialize occupied vertices
        const occupied = Object.entries(state.vertices)
            .filter(([_, v]) => v.stack.length > 0)
            .map(([id, v]) => `${id}:${v.stack[0].player}:${v.stack.length}:${v.energy}`)
            .sort()
            .join('|');
        
        return `${occupied}_${state.currentPlayerId}_${state.turn.hasPlaced}_${state.turn.hasInfused}_${state.turn.hasMoved}`;
    }

    // Helper methods (reuse from your game logic)
    private getForce(vertex: Vertex): number {
        if (vertex.stack.length === 0) return 0;
        const layerGravity = [1.0, 2.0, 3.0, 2.0, 1.0];
        return Math.min((vertex.stack.length * vertex.energy) / layerGravity[vertex.layer], 10);
    }

    private isOccupied(vertex: Vertex, requirementLayer?: number): boolean {
        const occupationRequirements = [
            { minPieces: 1, minEnergy: 1, minForce: 1 },
            { minPieces: 1, minEnergy: 1, minForce: 4 },
            { minPieces: 1, minEnergy: 1, minForce: 9 },
        ];
        const getReq = (layer: number) => {
            if (layer === 0 || layer === 4) return occupationRequirements[0];
            if (layer === 1 || layer === 3) return occupationRequirements[1];
            return occupationRequirements[2];
        };
        
        const layer = requirementLayer ?? vertex.layer;
        const req = getReq(layer);
        const force = this.getForce(vertex);
        return vertex.stack.length >= req.minPieces && vertex.energy >= req.minEnergy && force >= req.minForce;
    }
}

// Export singleton
export const unitEngine = new UnitGameEngine();