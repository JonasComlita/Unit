// src/components/UnitGame.tsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Engine, Scene, useScene } from 'react-babylonjs';
import { Vector3, Color3 } from '@babylonjs/core';
import { MeshBuilder, StandardMaterial } from '@babylonjs/core';
import { useGame, ActionPhase } from '../hooks/useGame';
import GameBoard from './GameBoard';
import MobileCameraController from './MobileCameraController';
import MobileHUD from './MobileHUD';
import DebugPanel from './DebugPanel';
import { VisualEffectsManager } from '../game/visualEffects';
import { Vertex } from '../game/types';
import { MatchInfo } from '../services/multiplayerService';


interface UnitGameProps {
  onGameStart?: () => boolean;
  isPremium?: boolean;
  onMultiplayerSelect?: () => void;
  matchInfo?: MatchInfo | null;
}

const UnitGame: React.FC<UnitGameProps> = ({ onGameStart, isPremium = false, onMultiplayerSelect, matchInfo }) => {
  const scene = useScene();
  const vfxRef = useRef<VisualEffectsManager | null>(null);
  const [visualQuality, setVisualQuality] = useState<'low' | 'medium' | 'high'>(
    isMobile() ? 'medium' : 'high'
  );
  const [gameStarted, setGameStarted] = useState(false);
  const [difficulty, setDifficulty] = useState<number>(5);

  // Initialize visual effects when scene is ready
  useEffect(() => {
    if (!scene) return;

    // If the manager isn't created yet, create it with the current visualQuality.
    // If it already exists, update its quality when visualQuality changes.
    if (!vfxRef.current) {
      vfxRef.current = new VisualEffectsManager(scene, {
        enableParticles: true,
        enableGlow: true,
        enableAnimations: true,
        enableShadows: !isMobile(), // Desktop only
        enablePostProcessing: !isMobile(), // Desktop only
        quality: visualQuality || (isMobile() ? 'medium' : 'high')
      });
      console.log('✨ Visual effects initialized (quality:', visualQuality, ')');

      // Cleanup on unmount
      return () => {
        if (vfxRef.current) {
          vfxRef.current.dispose();
          vfxRef.current = null;
        }
      };
    }

    // If the manager exists and visualQuality changed, ensure it's applied.
    if (vfxRef.current && visualQuality) {
      vfxRef.current.updateQuality(visualQuality);
    }
  }, [scene, visualQuality]);

  const { gameState, handleAction: baseHandleAction, undo, moveHistory, setDifficulty: setGameDifficulty } = useGame(matchInfo);
  const [activePhase, setActivePhase] = useState<ActionPhase | null>(null);

  // Enhanced handleAction with VFX
  const vfx = vfxRef.current;

  // Pulse animation for selected pieces
  useEffect(() => {
    const selectedVertexId = gameState.selectedVertexId;
    const manager = vfxRef.current;
    if (selectedVertexId && manager && scene) {
      const mesh = scene.getMeshByName(`piece-${selectedVertexId}`);
      if (mesh) {
        manager.animatePulse(mesh, 2000);
      }
    }
  }, [gameState.selectedVertexId, scene]);
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
      // Check pincer FIRST, because a pincer target is also a valid attack target
      if (validPincerTargets && validPincerTargets[vertexId] && validPincerTargets[vertexId].includes(selectedVertexId)) {
        handleAction({ type: 'pincer', targetId: vertexId, originIds: validPincerTargets[vertexId] });
        setActivePhase(null);
        return;
      } else if (validAttackTargets.includes(vertexId)) {
        handleAction({ type: 'attack', vertexId: selectedVertexId, targetId: vertexId });
        setActivePhase(null);
        return;
      } else if (validMoveTargets.includes(vertexId)) {
        handleAction({ type: 'move', fromId: selectedVertexId, toId: vertexId });
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
    } else if (activePhase === 'movement') {
      // During movement phase, only allow selecting friendly vertices as sources
      if (clickedVertex.stack.length > 0 && clickedVertex.stack[0].player === currentPlayerId) {
        handleAction({ type: 'select', vertexId: selectedVertexId === vertexId ? null : vertexId });
      } else {
        // Deselect if clicking on non-friendly vertex during movement phase
        handleAction({ type: 'select', vertexId: null });
      }
    } else if (clickedVertex.stack.length > 0 && clickedVertex.stack[0].player === currentPlayerId) {
      // General selection for friendly vertices (for combat)
      handleAction({ type: 'select', vertexId: selectedVertexId === vertexId ? null : vertexId });
    } else {
      // Allow viewing enemy/empty vertices (but not selecting them as action sources)
      handleAction({ type: 'select', vertexId: selectedVertexId === vertexId ? null : vertexId });
    }
  }, [gameState, handleAction, activePhase]);

  const handleQualityChange = (quality: 'low' | 'medium' | 'high') => {
    setVisualQuality(quality);
    if (vfxRef.current) {
      vfxRef.current.updateQuality(quality);
    }
  };

  const startGame = () => {
    if (onGameStart) {
      const canStart = onGameStart();
      if (canStart) {
        setGameStarted(true);
        if (setGameDifficulty) {
          setGameDifficulty(difficulty);
        }
      }
    } else {
      setGameStarted(true);
      if (setGameDifficulty) {
        setGameDifficulty(difficulty);
      }
    }
  };

  // Auto-start if match found
  useEffect(() => {
    if (matchInfo) {
      setGameStarted(true);
    }
  }, [matchInfo]);

  const getDifficultyLabel = (level: number): string => {
    const labels = [
      '', // 0 (unused)
      'Random',
      'Greedy',
      'Aggressor',
      'Banker',
      'Spreader',
      'Dynamic',
      'Alpha-Beta (2)',
      'Alpha-Beta (3)'
    ];
    return labels[level] || 'Unknown';
  };

  const getDifficultyDescription = (level: number): string => {
    const descriptions = [
      '', // 0 (unused)
      'Perfect for beginners. Opponent makes random moves.',
      'Basic strategy. Evaluates material advantage.',
      'Offensive focused. Prioritizes attacking enemy corners.',
      'Defensive focused. Protects home corners and builds material.',
      'Territory focused. Spreads across the board and sets up pincers.',
      'Adaptive strategy. Switches tactics based on game phase.',
      'Looks 2 moves ahead. Strong tactical play.',
      'Looks 3 moves ahead. Very strong but slower.'
    ];
    return descriptions[level] || '';
  };

  if (!gameStarted) {
    return (
      <div className="start-screen">
        <div className="start-content">
          <h1>Unit</h1>
          <p className="subtitle">Strategy Across Dimensions</p>

          <div className="difficulty-selector">
            <h3>Select AI Level (1-8)</h3>
            <div className="difficulty-slider-container">
              <input
                type="range"
                min="1"
                max="8"
                value={difficulty}
                onChange={(e) => setDifficulty(parseInt(e.target.value))}
                className="difficulty-slider"
              />
              <div className="difficulty-level-display">
                <span className="level-number">Level {difficulty}</span>
                <span className="level-name">{getDifficultyLabel(difficulty)}</span>
              </div>
            </div>
            <div className="difficulty-desc">
              {getDifficultyDescription(difficulty)}
            </div>
          </div>

          <button className="start-button" onClick={startGame}>
            ENTER THE ARENA
          </button>

          {onMultiplayerSelect && (
            <button
              className="start-button multiplayer-btn"
              onClick={onMultiplayerSelect}
              style={{
                marginTop: '1rem',
                background: isPremium
                  ? 'linear-gradient(45deg, #4a90e2, #003973)'
                  : 'linear-gradient(45deg, #4a4a4a, #2a2a2a)'
              }}
            >
              MULTIPLAYER {isPremium ? '' : '🔒'}
            </button>
          )}
        </div>
      </div>
    );
  }

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
      {/* Settings are now accessible via the top-left menu -> Settings in MobileHUD */}
      <MobileHUD
        gameState={gameState}
        onEndTurn={() => {
          handleAction({ type: 'endTurn' });
          setActivePhase(null);
        }}
        activePhase={activePhase}
        onPhaseSelect={handlePhaseSelect}
        onUndo={undo}
        undoCount={moveHistory.length}
        visualQuality={visualQuality}
        onQualityChange={handleQualityChange}
        onBack={() => {
          // Clear the current game and return to start screen
          localStorage.removeItem('currentGame');
          setGameStarted(false);
          handleAction({ type: 'select', vertexId: null });
          setActivePhase(null);
        }}
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
            onCreated={camera => camera.attachControl(true)}
          />
          <MobileCameraController />
          <hemisphericLight name="light1" intensity={0.9} direction={Vector3.Up()} />
          <hemisphericLight name="light2" intensity={0.4} direction={Vector3.Down()} />
          <GameBoard gameState={gameState} onVertexClick={handleVertexClick} activePhase={activePhase} />
        </Scene>
      </Engine>
      {/* On-screen debug panel (toggleable) */}
      <DebugPanel gameState={gameState} />
    </div>
  );
};

function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

export default UnitGame;