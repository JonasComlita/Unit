import React, { useState, useEffect, useMemo } from 'react';
import { useScene } from 'react-babylonjs';
import { Vector3, Quaternion, Color3, Mesh, PointerInfo, PointerEventTypes } from '@babylonjs/core';
import { GameState, Piece, PlayerId, Vertex } from '../game/types';
import { getForce } from '../game/gameLogic';
import { ActionPhase } from '../hooks/useGame';

// Game Board Component
const GameBoard: React.FC<{ gameState: GameState; onVertexClick: (id: string) => void; activePhase: ActionPhase | null }> = ({ gameState, onVertexClick, activePhase }) => {
    const { vertices, validPlacementVertices, validInfusionVertices, validAttackTargets, validPincerTargets, validMoveOrigins, validMoveTargets, selectedVertexId } = gameState;
    const scene = useScene();
    const [pulseTime, setPulseTime] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setPulseTime((t: number) => (t + 0.05) % (Math.PI * 2));
        }, 50);
        return () => clearInterval(interval);
    }, []);

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
        const pulse = Math.sin(pulseTime) * 0.5 + 0.5;

        return Object.values(vertices).map((vertex: Vertex) => {
            const pos = vertex.position;
            const force = getForce(vertex);
            const maxRadius = (vertex.vertexSpacing / 2) * 0.9;
            const radius = (force / 10) * maxRadius;
            const controller = vertex.stack.length > 0 ? vertex.stack[0].player : null;
            const color = controller === 'Player1' ? new Color3(0.29, 0.56, 0.89) : new Color3(0.82, 0.13, 0.11);
            
            // Increased tap target size (minimum 44pt equivalent in 3D)
            const tapTargetDiameter = Math.max(1.2, vertex.vertexSpacing / 6);

            const isPlacementTarget = validPlacementVertices.includes(vertex.id);
            const isInfusionTarget = validInfusionVertices.includes(vertex.id);
            const isMoveOrigin = validMoveOrigins.includes(vertex.id);
            const isMoveTarget = validMoveTargets.includes(vertex.id);
            const isAttackTarget = validAttackTargets.includes(vertex.id);
            const isPincerTarget = validPincerTargets && validPincerTargets[vertex.id];
            const isSelected = selectedVertexId === vertex.id;

            let emissiveColor = Color3.Black();
            let shouldPulse = false;

            if (activePhase === 'placement' && isPlacementTarget) {
                emissiveColor = new Color3(0.31, 0.78, 0.47);
                shouldPulse = true;
            } else if (activePhase === 'infusion' && isInfusionTarget) {
                emissiveColor = new Color3(1, 0.75, 0);
                shouldPulse = true;
            } else if (activePhase === 'movement' && isMoveOrigin) {
                emissiveColor = new Color3(0, 0.81, 0.82);
                shouldPulse = true;
            } else if (isAttackTarget) {
                emissiveColor = new Color3(0.86, 0.08, 0.24);
                shouldPulse = true;
            } else if (isPincerTarget) {
                emissiveColor = new Color3(0.55, 0, 1);
                shouldPulse = true;
            } else if (isMoveTarget) {
                emissiveColor = new Color3(0, 0.81, 0.82);
                shouldPulse = true;
            } else if (isSelected) {
                emissiveColor = Color3.White();
            }

            const finalEmissive = shouldPulse 
                ? emissiveColor.scale(0.3 + pulse * 0.7)
                : emissiveColor;

            return (
                <React.Fragment key={vertex.id}>
                    {/* Adjacency Lines */}
                    {vertex.adjacencies.map((adjId: string) => {
                        const adjVertex = vertices[adjId];
                        const lineId = [vertex.id, adjId].sort().join('-');
                        if (drawnLines.has(lineId)) return null;
                        drawnLines.add(lineId);
                        return (
                            <lines key={lineId} name={`line-${lineId}`} points={[pos, adjVertex.position]} color={new Color3(0.4, 0.4, 0.6)} alpha={0.3} />
                        );
                    })}

                    {/* Clickable Tap Target (Invisible) */}
                    <sphere name={vertex.id} diameter={tapTargetDiameter} segments={8} position={pos}>
                        <standardMaterial name={`mat-tap-${vertex.id}`} 
                            diffuseColor={new Color3(0.7, 0.7, 0.9)} 
                            emissiveColor={finalEmissive}
                            alpha={0.4}
                            specularColor={Color3.Black()} />
                    </sphere>
                    
                    {/* Visual Indicator Ring for Valid Actions */}
                    {shouldPulse && (
                        <torus name={`indicator-${vertex.id}`} diameter={tapTargetDiameter * 1.2} thickness={0.08} tessellation={32} position={pos}>
                            <standardMaterial name={`mat-indicator-${vertex.id}`} 
                                emissiveColor={finalEmissive}
                                alpha={0.8} />
                        </torus>
                    )}

                    {/* Force Visualization */}
                    {force > 0 && controller && (
                        <sphere name={`radius-${vertex.id}`} diameter={radius * 2} segments={16} position={pos}>
                            <standardMaterial name={`mat-radius-${vertex.id}`} 
                                diffuseColor={color} 
                                alpha={0.15} />
                        </sphere>
                    )}

                    {/* Piece Stack */}
                    {vertex.stack.map((piece: Piece, stackIndex: number) => {
                        const stackOffset = new Vector3(0, (stackIndex + 1) * 0.5, 0);
                        const piecePosition = pos.add(stackOffset);
                        return (
                            <sphere key={piece.id} name={`piece-${piece.id}`} diameter={0.45} segments={12} position={piecePosition}>
                                <standardMaterial name={`mat-piece-${piece.id}`} diffuseColor={color} specularColor={Color3.Black()} />
                            </sphere>
                        );
                    })}

                    {/* Energy Indicator */}
                    {vertex.energy > 0 && (
                        <plane name={`energy-${vertex.id}`} size={0.6} position={pos.add(new Vector3(0, 0.2, 0))}>
                            <advancedDynamicTexture name={`text-${vertex.id}`} height={128} width={128} createForParentMesh>
                                <textBlock text={`⚡${vertex.energy}`} color="yellow" fontSize={60} fontWeight="bold" />
                            </advancedDynamicTexture>
                        </plane>
                    )}
                </React.Fragment>
            );
        });
    }, [vertices, validPlacementVertices, validInfusionVertices, validAttackTargets, validPincerTargets, validMoveOrigins, validMoveTargets, selectedVertexId, activePhase, pulseTime]);

    return (
        <transformNode name="board-transform" rotationQuaternion={Quaternion.RotationAxis(Vector3.Up(), Math.PI / 4)}>
            {boardElements}
        </transformNode>
    );
};

export default GameBoard;