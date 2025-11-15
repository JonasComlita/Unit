// src/game/gameLogic.ts
import { Vector3 } from '@babylonjs/core';
import { GameState, Vertex, PlayerId } from './types';
import { BOARD_CONFIG, GAME_RULES, getOccupationRequirement } from './constants';

const { layout: boardLayoutRaw, layerSpacing, boardWidth, layerGravity } = BOARD_CONFIG;
const boardLayout: number[] = Array.from(boardLayoutRaw);

export const getForce = (vertex: Vertex | undefined): number => {
    if (!vertex || vertex.stack.length === 0) return 0;
    const gravityDivider = layerGravity[vertex.layer];
    const force = vertex.stack.length * vertex.energy / gravityDivider;
    return Math.min(force, GAME_RULES.forceCapMax);
};

export const isOccupied = (vertex: Vertex, requirementLayer?: number): boolean => {
    const layerToCheck = requirementLayer ?? vertex.layer;
    const req = getOccupationRequirement(layerToCheck);
    const force = getForce(vertex);
    return vertex.stack.length >= req.minPieces && 
           vertex.energy >= req.minEnergy && 
           force >= req.minForce;
};

// Replace the getAdjacencies function in gameLogic.ts

export const getAdjacencies = (
    vertex: Vertex,
    layerGrid: Vertex[][][],
    boardLayout: number[]
): string[] => {
    const { layer, position, vertexSpacing } = vertex;
    const size = boardLayout[layer];
    const offset = (size - 1) / 2;
    const x = vertexSpacing > 0 ? Math.round(position.x / vertexSpacing + offset) : 0;
    const z = vertexSpacing > 0 ? Math.round(position.z / vertexSpacing + offset) : 0;
    const adj: string[] = [];

    // --- 1. Same layer neighbors (4-connected grid) ---
    const neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    neighbors.forEach(([dx, dz]) => {
        const nx = x + dx;
        const nz = z + dz;
        // Check bounds
        if (nx >= 0 && nx < size && nz >= 0 && nz < size) {
            adj.push(layerGrid[layer][nx][nz].id);
        }
    });

    // --- 2. Adjacent layers (find closest vertex by world position) ---
    [layer - 1, layer + 1].forEach(targetLayer => {
        if (targetLayer >= 0 && targetLayer < boardLayout.length) {
            // Find the vertex on targetLayer that is closest in XZ plane
            let closestVertex: Vertex | null = null;
            let minDistance = Infinity;

            // Iterate through the 2D grid properly
            const targetSize = boardLayout[targetLayer];
            for (let tx = 0; tx < targetSize; tx++) {
                for (let tz = 0; tz < targetSize; tz++) {
                    const targetVertex = layerGrid[targetLayer][tx][tz];
                    
                    // Calculate XZ distance (ignore Y since layers are at different heights)
                    const dx = targetVertex.position.x - position.x;
                    const dz = targetVertex.position.z - position.z;
                    const distance = Math.sqrt(dx * dx + dz * dz);

                    if (distance < minDistance) {
                        minDistance = distance;
                        closestVertex = targetVertex;
                    }
                }
            }

            // Only add if the closest vertex is very close (essentially directly above/below)
            // Use a very tight threshold - only connect if nearly aligned
            const threshold = 0.1; // Very small - only vertically aligned vertices
            if (closestVertex && minDistance < threshold) {
                adj.push(closestVertex.id);
            }
        }
    });

    return Array.from(new Set(adj)); // Remove duplicates
};

export const initializeGameState = (): GameState => {
    const initialVertices: Record<string, Vertex> = {};
    const layerGrid: Vertex[][][] = boardLayout.map(size => 
        Array(size).fill(0).map(() => [])
    );
    let vertexIdCounter = 0;

    // Create all vertices
    boardLayout.forEach((size, layerIndex) => {
        const layerY = (layerIndex - Math.floor(boardLayout.length / 2)) * layerSpacing;
        const offset = (size - 1) / 2;
        const vertexSpacing = size > 1 ? boardWidth / (size - 1) : 0;

        for (let x = 0; x < size; x++) {
            for (let z = 0; z < size; z++) {
                const id = `vertex-${vertexIdCounter++}`;
                const pos = new Vector3(
                    (x - offset) * vertexSpacing, 
                    layerY, 
                    (z - offset) * vertexSpacing
                );
                // Debug log: print vertex initialization info
                console.log(`[initVertex] ${id} L${layerIndex} x=${x} z=${z} pos=(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`);
                const newVertex: Vertex = {
                    id,
                    position: pos,
                    layer: layerIndex,
                    layerSize: size,
                    adjacencies: [],
                    stack: [],
                    vertexSpacing,
                    energy: 0
                };
                initialVertices[id] = newVertex;
                layerGrid[layerIndex][x][z] = newVertex;
            }
        }
    });

    // Set up adjacencies
    Object.values(initialVertices).forEach(v => {
        v.adjacencies = getAdjacencies(v, layerGrid, boardLayout);
    });

    // Find home corners
    const bottomLayer = Object.values(initialVertices).filter(v => v.layer === 0);
    const topLayer = Object.values(initialVertices).filter(v => v.layer === boardLayout.length - 1);
    const minX = Math.min(...bottomLayer.map(v => v.position.x));
    const minZ = Math.min(...bottomLayer.map(v => v.position.z));
    const maxX = Math.max(...bottomLayer.map(v => v.position.x));
    const maxZ = Math.max(...bottomLayer.map(v => v.position.z));
    
    const p1BottomCorner = bottomLayer.find(v => v.position.x === minX && v.position.z === minZ);
    const p2BottomCorner = bottomLayer.find(v => v.position.x === maxX && v.position.z === maxZ);
    const p1TopCorner = topLayer.find(v => v.position.x === minX && v.position.z === minZ);
    const p2TopCorner = topLayer.find(v => v.position.x === maxX && v.position.z === maxZ);

    return {
        vertices: initialVertices,
        players: { 
            Player1: { id: 'Player1', reinforcements: GAME_RULES.initialReinforcements }, 
            Player2: { id: 'Player2', reinforcements: GAME_RULES.initialReinforcements } 
        },
        currentPlayerId: 'Player1',
        turn: { hasPlaced: false, hasInfused: false, hasMoved: false },
        homeCorners: { 
            Player1: [p1BottomCorner!.id, p1TopCorner!.id], 
            Player2: [p2BottomCorner!.id, p2TopCorner!.id] 
        },
        winner: null,
        selectedVertexId: null,
        validPlacementVertices: [],
        validInfusionVertices: [],
        validAttackTargets: [],
        validPincerTargets: {},
        validMoveOrigins: [],
        validMoveTargets: [],
    };
};

export const calculateValidActions = (state: GameState): Partial<GameState> => {
    const { vertices, currentPlayerId, turn, selectedVertexId, players, homeCorners } = state;
    const updates: Partial<GameState> = {
        validPlacementVertices: [],
        validInfusionVertices: [],
        validAttackTargets: [],
        validPincerTargets: {},
        validMoveOrigins: [],
        validMoveTargets: [],
    };

    // Placement
    if (!turn.hasPlaced && players[currentPlayerId].reinforcements > 0) {
        updates.validPlacementVertices = homeCorners[currentPlayerId];
    }

    // Infusion
    if (!turn.hasInfused) {
        updates.validInfusionVertices = Object.values(vertices)
            .filter(v => v.stack.length > 0 && v.stack[0].player === currentPlayerId)
            .filter(v => {
                const potentialForce = (v.stack.length * (v.energy + 1)) / layerGravity[v.layer];
                return potentialForce <= GAME_RULES.forceCapMax;
            })
            .map(v => v.id);
    }

    // Movement
    if (!turn.hasMoved) {
        updates.validMoveOrigins = Object.values(vertices)
            .filter(v => v.stack.length > 0 && v.stack[0].player === currentPlayerId)
            .map(v => v.id);
    }
    
    // Attack & Move Targets
    if (selectedVertexId) {
        const selectedVertex = vertices[selectedVertexId];
        if (selectedVertex.stack[0]?.player === currentPlayerId) {
            // Attack targets
            if (isOccupied(selectedVertex)) {
                updates.validAttackTargets = selectedVertex.adjacencies.filter(id => 
                    vertices[id].stack.length > 0 && 
                    vertices[id].stack[0].player !== currentPlayerId
                );
            }
            // Move targets
            if (!turn.hasMoved && (updates.validMoveOrigins ?? []).includes(selectedVertexId)) {
                updates.validMoveTargets = selectedVertex.adjacencies.filter(id => {
                    const targetVertex = vertices[id];
                    if (targetVertex.stack.length > 0) return false;
                    return isOccupied(selectedVertex, targetVertex.layer);
                });
            }
        }
    }

    // Pincer targets
    const pincerMap: Record<string, string[]> = {};
    Object.values(vertices).forEach(targetV => {
        const friendlyOrigins = targetV.adjacencies
            .map(id => vertices[id])
            .filter(orig => orig && orig.stack.length > 0 && orig.stack[0].player === currentPlayerId)
            .map(orig => orig.id);
        if (friendlyOrigins.length >= 2) {
            pincerMap[targetV.id] = friendlyOrigins.slice(0, GAME_RULES.maxPincerParticipants);
        }
    });
    updates.validPincerTargets = pincerMap;

    return updates;
};

export const checkWinner = (state: GameState): PlayerId | null => {
    const playerIds: PlayerId[] = ['Player1', 'Player2'];
    for (const pid of playerIds) {
        const opponent = pid === 'Player1' ? 'Player2' : 'Player1';
        const opponentCorners = state.homeCorners[opponent] || [];
        const controlsAll = opponentCorners.length > 0 && opponentCorners.every(cid => {
            const cv = state.vertices[cid];
            return cv && cv.stack.length > 0 && cv.stack[0].player === pid;
        });
        if (controlsAll) {
            return pid;
        }
    }
    return null;
};