// src/components/UnitGame.tsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Engine, Scene, useScene } from 'react-babylonjs';
import { Vector3, Color3 } from '@babylonjs/core';
import { MeshBuilder, StandardMaterial } from '@babylonjs/core';
import { useGame, ActionPhase } from '../hooks/useGame';
import GameBoard from './GameBoard';
import MobileCameraController from './MobileCameraController';
import MobileHUD from './MobileHUD';
import { VisualEffectsManager } from '../game/visualEffects';
import { Vertex } from '../game/types';


const UnitGame: React.FC = () => {
  const scene = useScene();
  const vfxRef = useRef<VisualEffectsManager | null>(null);

  // Initialize visual effects when scene is ready
  useEffect(() => {
    if (!scene || vfxRef.current) return;

    // Create visual effects manager
    vfxRef.current = new VisualEffectsManager(scene, {
      enableParticles: true,
      enableGlow: true,
      enableAnimations: true,
      enableShadows: !isMobile(), // Desktop only
      enablePostProcessing: !isMobile(), // Desktop only
      quality: isMobile() ? 'medium' : 'high'
    });

    console.log('✨ Visual effects initialized');

    // Cleanup on unmount
    return () => {
      if (vfxRef.current) {
        vfxRef.current.dispose();
        vfxRef.current = null;
      }
    };
  }, [scene]);

  const { gameState, handleAction: baseHandleAction } = useGame();
  const [activePhase, setActivePhase] = useState<ActionPhase | null>(null);

  // Enhanced handleAction with VFX
  const vfx = vfxRef.current;

  // Pulse animation for selected pieces
  useEffect(() => {
    const selectedVertexId = gameState.selectedVertexId;
    if (selectedVertexId && vfx && scene) {
      const mesh = scene.getMeshByName(`piece-${selectedVertexId}`);
      if (mesh) {
        vfx.animatePulse(mesh, 2000);
      }
    }
  }, [gameState.selectedVertexId, vfx, scene]);
  const handleAction = useCallback((action: any) => {
    baseHandleAction(action);
    // Add visual effects based on action type
    if (vfx) {
      switch (action.type) {
        case 'place': {
          const placeVertex = gameState.vertices[action.vertexId];
          if (placeVertex) vfx.createPlacementParticles(placeVertex.position);
          break;
        }
        case 'infuse': {
          const infuseVertex = gameState.vertices[action.vertexId];
          if (infuseVertex) vfx.createInfusionParticles(infuseVertex.position);
          break;
        }
        case 'attack': {
          const targetVertex = gameState.vertices[action.targetId];
          if (targetVertex) {
            // Get attacker and defender colors
            const attackerVertex = gameState.vertices[action.vertexId];
            const attackerColor = attackerVertex && attackerVertex.stack.length > 0 && attackerVertex.stack[0].player === 'Player1'
              ? new Color3(0.29, 0.56, 0.89)
              : new Color3(0.82, 0.13, 0.11);
            const defenderColor = targetVertex && targetVertex.stack.length > 0 && targetVertex.stack[0].player === 'Player1'
              ? new Color3(0.29, 0.56, 0.89)
              : new Color3(0.82, 0.13, 0.11);
            vfx.createAttackParticles(targetVertex.position, attackerColor, defenderColor);
            vfx.animateCameraShake(0.1, 300);
          }
          break;
        }
        case 'move': {
          // Animation handled in next step
          break;
        }
        default:
          break;
      }
    }
  }, [vfx, gameState, baseHandleAction]);
  // Victory celebration effect
  useEffect(() => {
    if (gameState.winner && vfx) {
      // Get winner's pieces positions
      const winnerPieces = Object.values(gameState.vertices as Record<string, Vertex>)
        .filter((v: Vertex) => v.stack.length > 0 && v.stack[0].player === gameState.winner)
        .map((v: Vertex) => v.position);

      // Create victory particles at each piece
      winnerPieces.forEach(pos => {
        setTimeout(() => {
          // Ensure pos is a Vector3
          vfx.createVictoryParticles(
            pos && typeof pos.add === 'function' ? pos : new Vector3(pos.x, pos.y, pos.z)
          );
        }, Math.random() * 1000);
      });

      // Camera celebration shake
      setTimeout(() => {
        vfx.animateCameraShake(0.05, 500);
      }, 500);
    }
  }, [gameState.winner, gameState.vertices, vfx]);
  // Register piece meshes as shadow casters and ground as shadow receiver
  useEffect(() => {
      if (!scene) return;
      
      // Create ground ONCE
      const existingGround = scene.getMeshByName('ground');
      if (!existingGround) {
          const ground = MeshBuilder.CreateGround('ground', { width: 30, height: 30 }, scene);
          ground.position.y = -10;
          const groundMat = new StandardMaterial('groundMat', scene);
          groundMat.diffuseColor = new Color3(0.15, 0.15, 0.2);
          ground.material = groundMat;
      }
  }, [scene]);

  // Victory celebration effect
  useEffect(() => {
    if (gameState.winner && vfx) {
      // Get winner's pieces positions
      const winnerPieces = Object.values(gameState.vertices as Record<string, Vertex>)
        .filter((v: Vertex) => v.stack.length > 0 && v.stack[0].player === gameState.winner)
        .map((v: Vertex) => v.position);

      // Create victory particles at each piece
      winnerPieces.forEach(pos => {
        setTimeout(() => {
          vfx.createVictoryParticles(pos);
        }, Math.random() * 1000);
      });

      // Camera celebration shake
      setTimeout(() => {
        vfx.animateCameraShake(0.05, 500);
      }, 500);
    }
  }, [gameState.winner, gameState.vertices, vfx]);

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

  const [visualQuality, setVisualQuality] = useState<'low' | 'medium' | 'high'>(
    isMobile() ? 'medium' : 'high'
  );

  const handleQualityChange = (quality: 'low' | 'medium' | 'high') => {
    setVisualQuality(quality);
    if (vfx) {
      vfx.updateQuality(quality);
    }
  };

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
      {/* Settings menu for visual quality */}
      <div style={{ position: 'absolute', top: 16, right: 16, zIndex: 10, background: '#222b', borderRadius: 8, padding: 8 }}>
        <label htmlFor="visual-quality-select" style={{ color: '#fff', marginRight: 8 }}>Visual Quality:</label>
        <select
          id="visual-quality-select"
          value={visualQuality}
          onChange={e => handleQualityChange(e.target.value as 'low' | 'medium' | 'high')}
          style={{ fontSize: 14, padding: 4, borderRadius: 4 }}
        >
          <option value="low">Low (Best Performance)</option>
          <option value="medium">Medium (Balanced)</option>
          <option value="high">High (Best Quality)</option>
        </select>
      </div>
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

function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

export default UnitGame;