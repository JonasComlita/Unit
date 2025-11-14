import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Engine, Scene, useScene } from 'react-babylonjs';
import { Vector3, Color3, Quaternion, PointerInfo, PointerEventTypes, Mesh, Animation, CubicEase, EasingFunction } from '@babylonjs/core';
import { useGame, ActionPhase } from '../hooks/useGame';
import GameBoard from './GameBoard';
import MobileCameraController from './MobileCameraController';
import MobileHUD from './MobileHUD';


// Main Game Component
const UnitGame: React.FC = () => {
  const { gameState, handleAction } = useGame();
  const [activePhase, setActivePhase] = useState<ActionPhase>(null);

  const handlePhaseSelect = useCallback((phase: ActionPhase) => {
    if (phase === 'placement' && gameState.turn.hasPlaced) return;
    if (phase === 'infusion' && gameState.turn.hasInfused) return;
    if (phase === 'movement' && gameState.turn.hasMoved) return;
    
    setActivePhase(activePhase === phase ? null : phase);
    handleAction({ type: 'select', vertexId: null });
  }, [activePhase, gameState.turn, handleAction]);

  const handleVertexClick = useCallback((vertexId: string) => {
    const { selectedVertexId, validPlacementVertices, validInfusionVertices, validAttackTargets, validPincerTargets, validMoveTargets, currentPlayerId, vertices, turn } = gameState;
    const clickedVertex = vertices[vertexId];

    // Combat actions (when a vertex is already selected)
    if (selectedVertexId) {
        if (validAttackTargets.includes(vertexId)) {
            handleAction({ type: 'attack', vertexId: selectedVertexId, targetId: vertexId });
            setActivePhase(null);
            return;
        } else if (validMoveTargets.includes(vertexId)) {
            handleAction({ type: 'move', fromId: selectedVertexId, toId: vertexId });
            setActivePhase(null);
            return;
        } else if (validPincerTargets && validPincerTargets[vertexId] && validPincerTargets[vertexId].includes(selectedVertexId)) {
            handleAction({ type: 'pincer', targetId: vertexId, originIds: validPincerTargets[vertexId] });
            setActivePhase(null);
            return;
        }
    }

    // Phase-based actions
    if (activePhase === 'placement' && !turn.hasPlaced && validPlacementVertices.includes(vertexId)) {
        handleAction({ type: 'place', vertexId });
        setActivePhase(null);
    } else if (activePhase === 'infusion' && !turn.hasInfused && validInfusionVertices.includes(vertexId)) {
        handleAction({ type: 'infuse', vertexId });
        setActivePhase(null);
    } else if (activePhase === 'movement' && clickedVertex.stack.length > 0 && clickedVertex.stack[0].player === currentPlayerId) {
        handleAction({ type: 'select', vertexId: selectedVertexId === vertexId ? null : vertexId });
    } else if (clickedVertex.stack.length > 0 && clickedVertex.stack[0].player === currentPlayerId) {
        // General selection for combat
        handleAction({ type: 'select', vertexId: selectedVertexId === vertexId ? null : vertexId });
    } else {
        // Deselect
        handleAction({ type: 'select', vertexId: null });
    }
  }, [gameState, handleAction, activePhase]);

  return (
    <div style={{ 
        width: '100vw', 
        height: '100vh', 
        margin: 0, 
        padding: 0, 
        overflow: 'hidden',
        background: 'linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%)',
        position: 'relative',
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <MobileHUD 
        gameState={gameState} 
        onEndTurn={() => {
            handleAction({ type: 'endTurn' });
            setActivePhase(null);
        }}
        activePhase={activePhase}
        onPhaseSelect={handlePhaseSelect}
      />
      <Engine antialias adaptToDeviceRatio canvasId="babylonJS">
        <Scene>
          <arcRotateCamera 
            name="camera1" 
            target={Vector3.Zero()} 
            alpha={-Math.PI / 2.5} 
            beta={Math.PI / 3} 
            radius={40} 
            minZ={0.001} 
            wheelPrecision={50} 
            lowerRadiusLimit={20} 
            upperRadiusLimit={80}
            panningSensibility={0}
          />
          <MobileCameraController />
          <hemisphericLight name="light1" intensity={0.9} direction={Vector3.Up()} />
          <hemisphericLight name="light2" intensity={0.4} direction={Vector3.Down()} />
          <GameBoard gameState={gameState} onVertexClick={handleVertexClick} activePhase={activePhase} />
        </Scene>
      </Engine>
    </div>
  );
};

export default UnitGame;