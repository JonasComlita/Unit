// src/App.tsx
import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
    Engine,
    Scene,
    ArcRotateCamera,
    HemisphericLight,
    Vector3,
    Color3,
    Mesh,
    StandardMaterial,
    MeshBuilder
} from '@babylonjs/core';
import { useGame } from './hooks/useGame';
import { ActionPhase } from './game/types';
import { COLORS, MOBILE_CONFIG } from './game/constants';
import { getForce } from './game/gameLogic';
import './App.css';

const App: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { gameState, handleAction } = useGame();
    const [activePhase, setActivePhase] = useState<ActionPhase>(null);

    useEffect(() => {
        if (!canvasRef.current) return;

        const canvas = canvasRef.current;
        const engine = new Engine(canvas, true);
        const scene = new Scene(engine);

        // Camera setup
        const camera = new ArcRotateCamera(
            'camera',
            MOBILE_CONFIG.cameraDefaults.alpha,
            MOBILE_CONFIG.cameraDefaults.beta,
            MOBILE_CONFIG.cameraDefaults.radius,
            Vector3.Zero(),
            scene
        );
        camera.attachControl(canvas, true);
        camera.lowerRadiusLimit = MOBILE_CONFIG.cameraDefaults.minRadius;
        camera.upperRadiusLimit = MOBILE_CONFIG.cameraDefaults.maxRadius;

        // Lighting
        const light1 = new HemisphericLight('light1', new Vector3(0, 1, 0), scene);
        light1.intensity = 0.9;

        const light2 = new HemisphericLight('light2', new Vector3(0, -1, 0), scene);
        light2.intensity = 0.4;

        // Ground
        const ground = MeshBuilder.CreateGround('ground', { width: 30, height: 30 }, scene);
        const groundMat = new StandardMaterial('groundMat', scene);
        groundMat.diffuseColor = new Color3(0.15, 0.15, 0.2);
        ground.material = groundMat;

        // Render vertices and pieces
        Object.values(gameState.vertices).forEach(vertex => {
            // Vertex sphere
            const sphere = MeshBuilder.CreateSphere(
                vertex.id,
                { diameter: 0.6, segments: 16 },
                scene
            );
            sphere.position = vertex.position;

            const mat = new StandardMaterial(`mat-${vertex.id}`, scene);
            mat.diffuseColor = new Color3(0.7, 0.7, 0.9);
            sphere.material = mat;

            // Pieces
            vertex.stack.forEach((piece, index) => {
                const pieceSphere = MeshBuilder.CreateSphere(
                    `piece-${piece.id}`,
                    { diameter: 0.45, segments: 12 },
                    scene
                );
                pieceSphere.position = vertex.position.add(new Vector3(0, (index + 1) * 0.5, 0));

                const pieceMat = new StandardMaterial(`piecemat-${piece.id}`, scene);
                const color = piece.player === 'Player1'
                    ? new Color3(COLORS.player1.r, COLORS.player1.g, COLORS.player1.b)
                    : new Color3(COLORS.player2.r, COLORS.player2.g, COLORS.player2.b);
                pieceMat.diffuseColor = color;
                pieceSphere.material = pieceMat;
            });

            // Force field
            if (vertex.stack.length > 0) {
                const force = getForce(vertex);
                if (force > 0) {
                    const radius = (force / 10) * (vertex.vertexSpacing / 2) * 0.9;
                    const forceSphere = MeshBuilder.CreateSphere(
                        `force-${vertex.id}`,
                        { diameter: radius * 2, segments: 16 },
                        scene
                    );
                    forceSphere.position = vertex.position;

                    const forceMat = new StandardMaterial(`forcemat-${vertex.id}`, scene);
                    const player = vertex.stack[0].player;
                    forceMat.diffuseColor = player === 'Player1'
                        ? new Color3(COLORS.player1.r, COLORS.player1.g, COLORS.player1.b)
                        : new Color3(COLORS.player2.r, COLORS.player2.g, COLORS.player2.b);
                    forceMat.alpha = 0.2;
                    forceSphere.material = forceMat;
                }
            }
        });

        // Click handling
        scene.onPointerDown = (evt, pickResult) => {
            if (pickResult.hit && pickResult.pickedMesh) {
                const meshName = pickResult.pickedMesh.name;
                if (meshName.startsWith('vertex-')) {
                    handleVertexClick(meshName);
                }
            }
        };

        // Render loop
        engine.runRenderLoop(() => {
            scene.render();
        });

        // Cleanup
        return () => {
            engine.dispose();
        };
    }, [gameState]);

    const handleVertexClick = useCallback((vertexId: string) => {
        const { selectedVertexId, validPlacementVertices, validInfusionVertices, 
                validAttackTargets, validPincerTargets, validMoveTargets, 
                currentPlayerId, vertices, turn } = gameState;
        const clickedVertex = vertices[vertexId];

        if (selectedVertexId) {
            if (validAttackTargets.includes(vertexId)) {
                handleAction({ type: 'attack', vertexId: selectedVertexId, targetId: vertexId });
            } else if (validMoveTargets.includes(vertexId)) {
                handleAction({ type: 'move', fromId: selectedVertexId, toId: vertexId });
            } else if (validPincerTargets && validPincerTargets[vertexId] && 
                       validPincerTargets[vertexId].includes(selectedVertexId)) {
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

    const handlePhaseSelect = useCallback((phase: ActionPhase) => {
        if (phase === 'placement' && gameState.turn.hasPlaced) return;
        if (phase === 'infusion' && gameState.turn.hasInfused) return;
        if (phase === 'movement' && gameState.turn.hasMoved) return;
        
        setActivePhase(activePhase === phase ? null : phase);
        handleAction({ type: 'select', vertexId: null });
    }, [activePhase, gameState.turn, handleAction]);

    const { turn, currentPlayerId, winner, players } = gameState;
    const allMandatoryDone = turn.hasPlaced && turn.hasInfused && turn.hasMoved;
    const playerColor = currentPlayerId === 'Player1' ? COLORS.player1.hex : COLORS.player2.hex;

    return (
        <div className="app-container">
            {/* Top Bar */}
            <div className="top-bar" style={{ borderBottom: `3px solid ${playerColor}` }}>
                <button className="menu-button">☰</button>
                <div className="player-indicator">
                    <div className="player-dot" style={{ background: playerColor, boxShadow: `0 0 12px ${playerColor}` }} />
                    <span className="player-name">{currentPlayerId === 'Player1' ? 'Player 1' : 'Player 2'}</span>
                </div>
                <div className="reinforcements-badge">
                    <span>⚡</span>
                    <span>{players[currentPlayerId].reinforcements}</span>
                </div>
            </div>

            {/* 3D Canvas */}
            <canvas ref={canvasRef} className="render-canvas" />

            {/* Winner Banner */}
            {winner && (
                <div className="winner-banner">
                    <h1>🏆 Victory! 🏆</h1>
                    <p>{winner === 'Player1' ? 'Player 1' : 'Player 2'} Wins!</p>
                    <button onClick={() => window.location.reload()}>New Game</button>
                </div>
            )}

            {/* Bottom Action Bar */}
            {!winner && (
                <div className="bottom-action-bar">
                    <div className="action-buttons">
                        <button
                            onClick={() => handlePhaseSelect('placement')}
                            disabled={turn.hasPlaced}
                            className={`action-btn ${activePhase === 'placement' ? 'active' : ''} ${turn.hasPlaced ? 'complete' : ''}`}
                        >
                            <span className="action-icon">📍</span>
                            <span>PLACE</span>
                            {turn.hasPlaced && <span className="check">✓</span>}
                        </button>

                        <button
                            onClick={() => handlePhaseSelect('infusion')}
                            disabled={turn.hasInfused}
                            className={`action-btn ${activePhase === 'infusion' ? 'active' : ''} ${turn.hasInfused ? 'complete' : ''}`}
                        >
                            <span className="action-icon">⚡</span>
                            <span>INFUSE</span>
                            {turn.hasInfused && <span className="check">✓</span>}
                        </button>

                        <button
                            onClick={() => handlePhaseSelect('movement')}
                            disabled={turn.hasMoved}
                            className={`action-btn ${activePhase === 'movement' ? 'active' : ''} ${turn.hasMoved ? 'complete' : ''}`}
                        >
                            <span className="action-icon">➡️</span>
                            <span>MOVE</span>
                            {turn.hasMoved && <span className="check">✓</span>}
                        </button>
                    </div>

                    <button
                        onClick={() => handleAction({ type: 'endTurn' })}
                        disabled={!allMandatoryDone}
                        className={`end-turn-btn ${allMandatoryDone ? 'enabled' : ''}`}
                    >
                        END TURN
                    </button>
                </div>
            )}
        </div>
    );
};

export default App;