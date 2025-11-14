import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { Engine, Scene, useScene } from 'react-babylonjs';
import { Vector3, Color3, Quaternion, PointerInfo, PointerEventTypes, Mesh } from '@babylonjs/core';
import './unit.css';
import { GameState, PlayerId, Vertex, Piece } from '../game/types';

// Action types for user interaction
type PlayerAction = 
    | { type: 'select'; vertexId: string | null }
    | { type: 'place'; vertexId: string }
    | { type: 'infuse'; vertexId: string }
    | { type: 'move'; fromId: string; toId: string }
    | { type: 'attack'; vertexId: string; targetId: string }
    | { type: 'pincer'; targetId: string; originIds: string[] }
    | { type: 'endTurn' };

// --- CONSTANTS & CONFIG from rules.json ---
const boardLayout = [3, 5, 7, 5, 3];
const layerGravity = [1.0, 2.0, 3.0, 2.0, 1.0];
const layerSpacing = 2.5;
const boardWidth = 10;

const occupationRequirements = [
    { minPieces: 1, minEnergy: 1, minForce: 1 }, // 3x3 layers
    { minPieces: 1, minEnergy: 1, minForce: 4 }, // 5x5 layers
    { minPieces: 1, minEnergy: 1, minForce: 9 }, // 7x7 layer
];

const getOccupationRequirement = (layer: number) => {
    if (layer === 0 || layer === 4) return occupationRequirements[0];
    if (layer === 1 || layer === 3) return occupationRequirements[1];
    return occupationRequirements[2];
};

// --- UTILITIES ---
const getForce = (vertex: Vertex | undefined): number => {
    if (!vertex || vertex.stack.length === 0) return 0;
    const gravityDivider = layerGravity[vertex.layer];
    const force = vertex.stack.length * vertex.energy / gravityDivider;
    return Math.min(force, 10); // Force is capped at 10
};

const isOccupied = (vertex: Vertex, requirementLayer?: number): boolean => {
    const layerToCheck = requirementLayer ?? vertex.layer;
    const req = getOccupationRequirement(layerToCheck);
    const force = getForce(vertex);
    return vertex.stack.length >= req.minPieces && vertex.energy >= req.minEnergy && force >= req.minForce;
};

const getAdjacencies = (vertex: Vertex, layerGrid: Vertex[][][], boardLayout: number[]): string[] => {
    const { layer, position, vertexSpacing } = vertex;
    const size = boardLayout[layer];
    const offset = (size - 1) / 2;
    const x = vertexSpacing > 0 ? Math.round(position.x / vertexSpacing + offset) : 0;
    const z = vertexSpacing > 0 ? Math.round(position.z / vertexSpacing + offset) : 0;
    const adj: string[] = [];

    const neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    neighbors.forEach(([dx, dz]) => {
        const nx = (x + dx + size) % size;
        const nz = (z + dz + size) % size;
        adj.push(layerGrid[layer][nx][nz].id);
    });

    [layer - 1, layer + 1].forEach(nextLayer => {
        if (nextLayer >= 0 && nextLayer < boardLayout.length) {
            const nextSize = boardLayout[nextLayer];
            const scale = nextSize / size;
            const nx = Math.round(x * scale);
            const nz = Math.round(z * scale);
            if (layerGrid[nextLayer]?.[nx]?.[nz]) {
                 adj.push(layerGrid[nextLayer][nx][nz].id);
            }
        }
    });
    return Array.from(new Set(adj));
};

// --- GAME LOGIC HOOK ---
const useGame = () => {
  const [gameState, setGameState] = useState<GameState>(() => {
    const initialVertices: Record<string, Vertex> = {};
    const layerGrid: Vertex[][][] = boardLayout.map(size => Array(size).fill(0).map(() => []));
    let vertexIdCounter = 0;

    boardLayout.forEach((size, layerIndex) => {
      const layerY = (layerIndex - Math.floor(boardLayout.length / 2)) * layerSpacing;
      const offset = (size - 1) / 2;
      const vertexSpacing = size > 1 ? boardWidth / (size - 1) : 0;

      for (let x = 0; x < size; x++) {
        for (let z = 0; z < size; z++) {
          const id = `vertex-${vertexIdCounter++}`;
          const pos = new Vector3((x - offset) * vertexSpacing, layerY, (z - offset) * vertexSpacing);
          const newVertex: Vertex = { id, position: pos, layer: layerIndex, layerSize: size, adjacencies: [], stack: [], vertexSpacing, energy: 0 };
          initialVertices[id] = newVertex;
          layerGrid[layerIndex][x][z] = newVertex;
        }
      }
    });

    Object.values(initialVertices).forEach(v => {
        v.adjacencies = getAdjacencies(v, layerGrid, boardLayout);
    });

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

    const initialState: GameState = {
      vertices: initialVertices,
      players: { 
        Player1: { id: 'Player1', reinforcements: 1 }, 
        Player2: { id: 'Player2', reinforcements: 1 } 
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
    
    // Initial valid actions
    const validActions = calculateValidActions(initialState);
    return { ...initialState, ...validActions };
  });

  const calculateValidActions = useCallback((state: GameState): Partial<GameState> => {
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
                return potentialForce <= 10; // Force Cap
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
                    vertices[id].stack.length > 0 && vertices[id].stack[0].player !== currentPlayerId
                );
            }
            // Move targets
            if (!turn.hasMoved && (updates.validMoveOrigins ?? []).includes(selectedVertexId)) {
                updates.validMoveTargets = selectedVertex.adjacencies.filter(id => {
                    const targetVertex = vertices[id];
                    if (targetVertex.stack.length > 0) return false; // Must be empty
                    // Source must meet occupation requirements for the TARGET layer
                    return isOccupied(selectedVertex, targetVertex.layer);
                });
            }
        }
    }

    // Pincer targets: a vertex is a valid pincer target when two or more
    // friendly origin vertices share it as an adjacent vertex. We allow up to
    // 6 origins to participate in a pincer.
    const pincerMap: Record<string, string[]> = {};
    Object.values(vertices).forEach(targetV => {
        const friendlyOrigins = targetV.adjacencies
            .map(id => vertices[id])
            .filter(orig => orig && orig.stack.length > 0 && orig.stack[0].player === currentPlayerId)
            .map(orig => orig.id);
        if (friendlyOrigins.length >= 2) {
            pincerMap[targetV.id] = friendlyOrigins.slice(0, 6);
        }
    });
    updates.validPincerTargets = pincerMap;

    return updates;
  }, []);

  const handleAction = useCallback((action: PlayerAction) => {
    setGameState(prev => {
        let nextState: GameState = JSON.parse(JSON.stringify(prev));
        
        switch (action.type) {
            case 'select':
                nextState.selectedVertexId = action.vertexId;
                break;

            case 'place':
                if (!prev.turn.hasPlaced && prev.validPlacementVertices.includes(action.vertexId) && prev.players[prev.currentPlayerId].reinforcements > 0) {
                    const newPiece: Piece = { id: `p-${Date.now()}`, player: prev.currentPlayerId };
                    nextState.vertices[action.vertexId].stack.push(newPiece);
                    nextState.players[prev.currentPlayerId].reinforcements -= 1;
                    nextState.turn.hasPlaced = true;
                }
                break;
            
            case 'infuse':
                if (!prev.turn.hasInfused && prev.validInfusionVertices.includes(action.vertexId)) {
                    nextState.vertices[action.vertexId].energy += 1;
                    nextState.turn.hasInfused = true;
                }
                break;

            case 'move':
                // Moves always transfer the entire stack and the full energy value
                // from source -> target. Partial moves or splitting a stack are not
                // supported by the game rules.
                if (!prev.turn.hasMoved && prev.validMoveOrigins.includes(action.fromId) && prev.validMoveTargets.includes(action.toId)) {
                    const source = nextState.vertices[action.fromId];
                    const target = nextState.vertices[action.toId];
                    // Transfer whole stack and energy
                    target.stack = source.stack;
                    target.energy = source.energy;
                    // Empty source
                    source.stack = [];
                    source.energy = 0;
                    nextState.turn.hasMoved = true;
                    nextState.selectedVertexId = null;
                }
                break;

            case 'attack':
                if (action.vertexId && action.targetId && prev.validAttackTargets.includes(action.targetId)) {
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

                    if (attackerForce > defenderForce) { // Attacker wins
                        defenderV.stack = [];
                        for (let i = 0; i < newPieces; i++) {
                            defenderV.stack.push({ id: `p-conquer-${i}-${Date.now()}`, player: prev.currentPlayerId });
                        }
                        defenderV.energy = newEnergy;
                    } else { // Defender wins or draw
                        defenderV.stack = [];
                        for (let i = 0; i < newPieces; i++) {
                            defenderV.stack.push({ id: `p-defend-${i}-${Date.now()}`, player: defenderV.stack[0].player });
                        }
                        defenderV.energy = newEnergy;
                    }
                    
                    // Attacker's original vertex is always emptied
                    attackerV.stack = [];
                    attackerV.energy = 0;

                    // Attack ends the turn
                    nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                    nextState.players[nextState.currentPlayerId].reinforcements += 1;
                    nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false };
                    nextState.selectedVertexId = null;
                }
                break;

            case 'pincer':
                if (action.targetId && Array.isArray(action.originIds) && Object.keys(prev.validPincerTargets || {}).includes(action.targetId)) {
                    const allowedOrigins = prev.validPincerTargets[action.targetId] || [];
                    // Ensure all requested origins are allowed and belong to the current player
                    const allValid = action.originIds.every(id => allowedOrigins.includes(id) && prev.vertices[id].stack.length > 0 && prev.vertices[id].stack[0].player === prev.currentPlayerId);
                    if (allValid) {
                        const defenderV = nextState.vertices[action.targetId];
                        const originVerts = action.originIds.map(id => nextState.vertices[id]);

                        // Multiply forces of all participating origins (cap to 10)
                        let attackerForce = originVerts.map(getForce).reduce((a, b) => a * b, 1);
                        attackerForce = Math.min(attackerForce, 10);
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
                                defenderV.stack.push({ id: `p-conquer-${i}-${Date.now()}`, player: prev.currentPlayerId });
                            }
                            defenderV.energy = newEnergy;
                        } else {
                            const defenderOwner = defenderV.stack[0]?.player ?? prev.currentPlayerId;
                            defenderV.stack = [];
                            for (let i = 0; i < newPieces; i++) {
                                defenderV.stack.push({ id: `p-defend-${i}-${Date.now()}`, player: defenderOwner });
                            }
                            defenderV.energy = newEnergy;
                        }

                        // Empty origin vertices (they collapsed)
                        originVerts.forEach(v => { v.stack = []; v.energy = 0; });

                        // Pincer ends the turn (same as attack)
                        nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                        nextState.players[nextState.currentPlayerId].reinforcements += 1;
                        nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false };
                        nextState.selectedVertexId = null;
                    }
                }
                break;

            case 'endTurn':
                if (prev.turn.hasPlaced && prev.turn.hasInfused && prev.turn.hasMoved) {
                    nextState.currentPlayerId = prev.currentPlayerId === 'Player1' ? 'Player2' : 'Player1';
                    nextState.players[nextState.currentPlayerId].reinforcements += 1;
                    nextState.turn = { hasPlaced: false, hasInfused: false, hasMoved: false };
                    nextState.selectedVertexId = null;
                }
                break;
        }

        // Winner logic: A player wins when they successfully move onto (i.e. control)
        // both of the opponent's home corners. Check both players against their
        // opponent's home corners on the resulting state.
        nextState.winner = null;
        const playerIds: PlayerId[] = ['Player1', 'Player2'];
        for (const pid of playerIds) {
            const opponent = pid === 'Player1' ? 'Player2' : 'Player1';
            const opponentCorners = nextState.homeCorners[opponent] || [];
            const controlsAll = opponentCorners.length > 0 && opponentCorners.every(cid => {
                const cv = nextState.vertices[cid];
                return cv && cv.stack.length > 0 && cv.stack[0].player === pid;
            });
            if (controlsAll) {
                nextState.winner = pid;
                break;
            }
        }

        const validActions = calculateValidActions(nextState);
        return { ...nextState, ...validActions };
    });
  }, [calculateValidActions]);

  return { gameState, handleAction };
};

// --- UI COMPONENTS ---
const GameBoard: React.FC<{ gameState: GameState; onVertexClick: (id: string) => void }> = ({ gameState, onVertexClick }) => {
    const { vertices, validPlacementVertices, validInfusionVertices, validAttackTargets, validPincerTargets, validMoveOrigins, validMoveTargets, selectedVertexId } = gameState;
    const [viewMode, setViewMode] = useState<'stacked' | 'side-by-side'>('stacked');
    const scene = useScene();
    // drawnLines should be local to the memo so it doesn't change across renders
    // and won't break the useMemo dependency analysis.

    const getVertexPosition = useCallback((vertex: Vertex) => {
        if (viewMode === 'stacked') {
            return vertex.position;
        }
        // Side-by-side view
        const layerXOffset = (vertex.layer - Math.floor(boardLayout.length / 2)) * (boardWidth + 4);
        return new Vector3(vertex.position.x + layerXOffset, 0, vertex.position.z);
    }, [viewMode]);

    useEffect(() => {
        const pointerDownListener = (pointerInfo: PointerInfo) => {
            if (pointerInfo.type === PointerEventTypes.POINTERDOWN) {
                const pickInfo = pointerInfo.pickInfo;
                if (pickInfo && pickInfo.hit && pickInfo.pickedMesh) {
                    const clickedMesh = pickInfo.pickedMesh as Mesh;
                    if (clickedMesh.name.startsWith('vertex-')) {
                        onVertexClick(clickedMesh.name);
                    }
                }
            }
        };
        scene?.onPointerObservable.add(pointerDownListener);
        return () => {
            scene?.onPointerObservable.removeCallback(pointerDownListener);
        }
    }, [scene, onVertexClick]);

    const boardElements = useMemo(() => {
        const drawnLines = new Set<string>();
        return Object.values(vertices).map(vertex => {
            const pos = getVertexPosition(vertex);
            const force = getForce(vertex);
            const maxRadius = (vertex.vertexSpacing / 2) * 0.9;
            const radius = (force / 10) * maxRadius;
            const controller = vertex.stack.length > 0 ? vertex.stack[0].player : null;
            const color = controller === 'Player1' ? new Color3(0.2, 0.5, 1) : new Color3(1, 0.2, 0.2);
            const sphereDiameter = Math.max(0.3, vertex.vertexSpacing / 10);

            const isMoveOrigin = validMoveOrigins.includes(vertex.id);
            const isMoveTarget = validMoveTargets.includes(vertex.id);

            return (
                <React.Fragment key={vertex.id}>
                    {/* Adjacency Lines */}
                    {vertex.adjacencies.map(adjId => {
                        const adjVertex = vertices[adjId];
                        const lineId = [vertex.id, adjId].sort().join('-');
                        if (drawnLines.has(lineId)) return null;
                        drawnLines.add(lineId);
                        return (
                            <lines key={lineId} name={`line-${lineId}`} points={[pos, getVertexPosition(adjVertex)]} color={new Color3(0.6, 0.6, 0.8)} alpha={0.5} />
                        );
                    })}

                    {/* Base Vertex Sphere (Clickable) */}
                    <sphere name={vertex.id} diameter={sphereDiameter} segments={12} position={pos}>
                        <standardMaterial name={`mat-vertex-${vertex.id}`} 
                            diffuseColor={new Color3(0.8, 0.8, 1)} 
                            emissiveColor={
                                validPlacementVertices.includes(vertex.id) ? Color3.Green() :
                                validInfusionVertices.includes(vertex.id) ? Color3.Yellow() :
                                (validPincerTargets && validPincerTargets[vertex.id]) ? new Color3(0.6, 0.2, 0.8) : // Purple for pincer targets
                                validAttackTargets.includes(vertex.id) ? Color3.Red() :
                                isMoveTarget ? Color3.Blue() :
                                selectedVertexId === vertex.id ? Color3.White() :
                                isMoveOrigin ? new Color3(0.5, 0.5, 1) : // Light blue for move origins
                                Color3.Black()
                            }
                            specularColor={Color3.Black()} />
                    </sphere>
                    
                    {force > 0 && controller &&
                        <sphere name={`radius-${vertex.id}`} diameter={radius * 2} segments={16} position={pos}>
                            <standardMaterial name={`mat-radius-${vertex.id}`} diffuseColor={color} alpha={0.2} />
                        </sphere>
                    }

                    {vertex.stack.map((piece, stackIndex) => {
                        const stackOffset = new Vector3(0, (stackIndex + 1) * 0.4, 0);
                        const piecePosition = pos.add(stackOffset);
                        return (
                            <sphere key={piece.id} name={`piece-${piece.id}`} diameter={0.35} segments={12} position={piecePosition}>
                                <standardMaterial name={`mat-piece-${piece.id}`} diffuseColor={color} specularColor={Color3.Black()} />
                            </sphere>
                        );
                    })}
                </React.Fragment>
            );
        });
    }, [vertices, getVertexPosition, validPlacementVertices, validInfusionVertices, validAttackTargets, validPincerTargets, validMoveOrigins, validMoveTargets, selectedVertexId]);

    return (
        <>
            <div style={{ position: 'absolute', top: '20px', right: '20px', zIndex: 10 }}>
                <button onClick={() => setViewMode(v => v === 'stacked' ? 'side-by-side' : 'stacked')}>Toggle View</button>
            </div>
            <transformNode name="board-transform" rotationQuaternion={viewMode === 'stacked' ? Quaternion.RotationAxis(Vector3.Up(), Math.PI / 4) : undefined}>
                {boardElements}
            </transformNode>
        </>
    );
};

const Hud: React.FC<{ gameState: GameState; onEndTurn: () => void }> = ({ gameState, onEndTurn }) => {
    const { turn, currentPlayerId, winner, players } = gameState;
    const allMandatoryDone = turn.hasPlaced && turn.hasInfused && turn.hasMoved;

    return (
        <div style={{ position: 'absolute', top: '20px', left: '20px', zIndex: 10, fontFamily: 'sans-serif', color: 'white', backgroundColor: 'rgba(0,0,0,0.5)', padding: '10px', borderRadius: '8px', minWidth: '220px' }}>
            {winner ? (
                <h2 style={{ margin: 0, color: '#FFD700' }}>{winner} Wins!</h2>
            ) : (
                <>
                    <h2 style={{ margin: 0, paddingBottom: '10px', borderBottom: '1px solid #555' }}>
                        Turn: <span style={{ color: currentPlayerId === 'Player1' ? '#4A90E2' : '#D0021B', fontWeight: 'bold' }}>{currentPlayerId}</span>
                    </h2>
                    <div style={{ paddingTop: '10px' }}>
                        <p style={{ margin: '5px 0' }}>Reinforcements: {players[currentPlayerId].reinforcements}</p>
                        <h4 style={{marginTop: '15px', marginBottom: '5px'}}>Mandatory Actions:</h4>
                        <p style={{ margin: '5px 0', color: turn.hasPlaced ? 'lightgreen' : 'white' }}>Place Piece: {turn.hasPlaced ? 'âœ“' : 'Pending'}</p>
                        <p style={{ margin: '5px 0', color: turn.hasInfused ? 'lightgreen' : 'white' }}>Infuse Energy: {turn.hasInfused ? 'âœ“' : 'Pending'}</p>
                        <p style={{ margin: '5px 0', color: turn.hasMoved ? 'lightgreen' : 'white' }}>Move Stack: {turn.hasMoved ? 'âœ“' : 'Pending'}</p>
                    </div>
                    <button onClick={onEndTurn} style={{ width: '100%', marginTop: '10px', padding: '8px', fontSize: '16px' }} disabled={!allMandatoryDone || !!winner}>End Turn</button>
                </>
            )}
        </div>
    );
};

// --- MAIN GAME COMPONENT ---
const UnitGame: React.FC = () => {
  const { gameState, handleAction } = useGame();

  const handleVertexClick = useCallback((vertexId: string) => {
    const { selectedVertexId, validPlacementVertices, validInfusionVertices, validAttackTargets, validPincerTargets, validMoveTargets, currentPlayerId, vertices, turn } = gameState;
    const clickedVertex = vertices[vertexId];

    if (selectedVertexId) {
        if (validAttackTargets.includes(vertexId)) {
            handleAction({ type: 'attack', vertexId: selectedVertexId, targetId: vertexId });
        } else if (validMoveTargets.includes(vertexId)) {
            handleAction({ type: 'move', fromId: selectedVertexId, toId: vertexId });
        } else if (validPincerTargets && validPincerTargets[vertexId] && validPincerTargets[vertexId].includes(selectedVertexId)) {
            // Execute pincer using all available friendly origins for this target
            handleAction({ type: 'pincer', targetId: vertexId, originIds: validPincerTargets[vertexId] });
        } else {
            handleAction({ type: 'select', vertexId: selectedVertexId === vertexId ? null : vertexId });
        }
    } else {
        if (!turn.hasPlaced && validPlacementVertices.includes(vertexId)) {
            handleAction({ type: 'place', vertexId });
        } else if (!turn.hasInfused && validInfusionVertices.includes(vertexId)) {
            handleAction({ type: 'infuse', vertexId });
        } else if (clickedVertex.stack.length > 0 && clickedVertex.stack[0].player === currentPlayerId) {
            handleAction({ type: 'select', vertexId });
        }
    }
  }, [gameState, handleAction]);

  return (
    <div className="unit-game-container">
      <Hud gameState={gameState} onEndTurn={() => handleAction({ type: 'endTurn' })} />
      <Engine antialias adaptToDeviceRatio canvasId="babylonJS">
        <Scene>
          <arcRotateCamera name="camera1" target={Vector3.Zero()} alpha={-Math.PI / 2.5} beta={Math.PI / 3} radius={40} minZ={0.001} wheelPrecision={50} lowerRadiusLimit={20} upperRadiusLimit={80} useAutoRotationBehavior />
          <hemisphericLight name="light1" intensity={0.9} direction={Vector3.Up()} />
          <hemisphericLight name="light2" intensity={0.4} direction={Vector3.Down()} />
          <GameBoard gameState={gameState} onVertexClick={handleVertexClick} />
        </Scene>
      </Engine>
    </div>
  );
};

export default UnitGame;