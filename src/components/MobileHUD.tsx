import React, { useState, useEffect } from 'react';

// Mock types for the example
type PlayerId = 'Player1' | 'Player2';
type ActionPhase = 'placement' | 'infusion' | 'movement';

interface GameState {
  turn: { hasPlaced: boolean; hasInfused: boolean; hasMoved: boolean; turnNumber: number };
  currentPlayerId: PlayerId;
  winner: PlayerId | null;
  players: Record<PlayerId, { id: PlayerId; reinforcements: number }>;
  selectedVertexId: string | null;
  validAttackTargets: string[];
  validPincerTargets: Record<string, string[]> | null;
  validMoveTargets: string[];
  vertices: Record<string, {
    id: string;
    stack: Array<{ player: PlayerId; id: string }>;
    energy: number;
    layer: number;
  }>;
}

type PanelState = 'default' | 'unit-selected' | 'enemy-selected' | 'action-active' | 'message';

interface MobileHUDProps {
  gameState: GameState;
  onEndTurn: () => void;
  activePhase: ActionPhase | null;
  onPhaseSelect: (phase: ActionPhase) => void;
  onUndo?: () => void;
  undoCount?: number;
  visualQuality: 'low' | 'medium' | 'high';
  onQualityChange: (quality: 'low' | 'medium' | 'high') => void;
}

const MobileHUD: React.FC<MobileHUDProps> = ({ gameState, onEndTurn, activePhase, onPhaseSelect, onUndo, undoCount, visualQuality, onQualityChange }) => {
  const { turn, currentPlayerId, winner, players, selectedVertexId, validAttackTargets, validMoveTargets, vertices } = gameState;
  const [showMenu, setShowMenu] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [panelState, setPanelState] = useState<PanelState>('default');
  const [message, setMessage] = useState<string | null>(null);
  const [activeAction, setActiveAction] = useState<'move' | 'attack' | 'infuse' | null>(null);

  const allMandatoryDone = turn.hasPlaced && turn.hasInfused && turn.hasMoved;
  const playerColor = currentPlayerId === 'Player1' ? '#4A90E2' : '#D0021B';
  const selectedVertex = selectedVertexId ? vertices[selectedVertexId] : null;
  const isOwnUnit = selectedVertex && selectedVertex.stack.length > 0 && selectedVertex.stack[0].player === currentPlayerId;
  const isEnemyUnit = selectedVertex && selectedVertex.stack.length > 0 && selectedVertex.stack[0].player !== currentPlayerId;

  // Determine panel state based on selection and active phase
  useEffect(() => {
    if (message) {
      setPanelState('message');
    } else if (activeAction) {
      setPanelState('action-active');
    } else if (isOwnUnit) {
      setPanelState('unit-selected');
    } else if (isEnemyUnit) {
      setPanelState('enemy-selected');
    } else {
      setPanelState('default');
    }
  }, [selectedVertexId, activeAction, message, isOwnUnit, isEnemyUnit]);

  // Auto-clear messages after 3 seconds
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => setMessage(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  const handleActionClick = (action: 'move' | 'attack' | 'infuse') => {
    if (action === 'move' && turn.hasMoved) {
      setMessage('You have already moved this turn!');
      return;
    }
    if (action === 'infuse' && turn.hasInfused) {
      setMessage('You have already infused this turn!');
      return;
    }
    setActiveAction(action);
  };

  const handleCancelAction = () => {
    setActiveAction(null);
    onPhaseSelect(null as any);
  };

  const canAttack = selectedVertexId && validAttackTargets.length > 0;
  const canMove = selectedVertexId && validMoveTargets.length > 0 && !turn.hasMoved;
  const canInfuse = !turn.hasInfused;

  return (
    <>
      {/* Enhanced Status Bar - Always Visible */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '70px',
        background: 'linear-gradient(180deg, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.85) 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 15px',
        zIndex: 100,
        borderBottom: `4px solid ${playerColor}`,
        boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
      }}>
        <button 
          onClick={() => setShowMenu(!showMenu)}
          style={{
            width: '50px',
            height: '50px',
            background: 'rgba(255,255,255,0.1)',
            border: '2px solid rgba(255,255,255,0.3)',
            borderRadius: '10px',
            color: 'white',
            fontSize: '24px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
        >
          ☰
        </button>

        {/* Turn Info */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '15px',
          flex: 1,
          justifyContent: 'center',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}>
            <div style={{
              width: '16px',
              height: '16px',
              borderRadius: '50%',
              background: playerColor,
              boxShadow: `0 0 16px ${playerColor}`,
              animation: 'pulse 2s ease-in-out infinite',
            }} />
            <span style={{ 
              color: 'white', 
              fontSize: '20px', 
              fontWeight: 'bold',
              fontFamily: 'system-ui, -apple-system, sans-serif',
            }}>
              {currentPlayerId === 'Player1' ? 'Blue' : 'Red'} Player
            </span>
          </div>
          
          {/* Turn Counter */}
          <div style={{
            background: 'rgba(255,255,255,0.1)',
            padding: '6px 12px',
            borderRadius: '20px',
            border: '2px solid rgba(255,255,255,0.2)',
          }}>
            <span style={{ 
              color: 'white', 
              fontSize: '16px', 
              fontWeight: 'bold',
            }}>
              Turn {turn.turnNumber || 1}
            </span>
          </div>
        </div>

        {/* Reinforcements */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'rgba(255,255,255,0.1)',
          padding: '10px 14px',
          borderRadius: '25px',
          border: '2px solid rgba(255,255,255,0.2)',
        }}>
          <span style={{ fontSize: '22px' }}>⚡</span>
          <span style={{ 
            color: 'white', 
            fontSize: '20px', 
            fontWeight: 'bold',
            minWidth: '24px',
            textAlign: 'center',
          }}>
            {players[currentPlayerId].reinforcements}
          </span>
        </div>
        {/* Undo Button */}
        <div style={{ marginLeft: 12 }}>
          <button
            onClick={() => onUndo && onUndo()}
            disabled={!undoCount || undoCount <= 0}
            style={{
              padding: '10px 14px',
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '10px',
              color: undoCount && undoCount > 0 ? 'white' : 'rgba(255,255,255,0.4)',
              cursor: undoCount && undoCount > 0 ? 'pointer' : 'not-allowed',
            }}
            title={undoCount && undoCount > 0 ? `Undo (${undoCount})` : 'Nothing to undo'}
          >
            ↺ Undo
          </button>
        </div>
      </div>

      {/* Menu Overlay */}
      {showMenu && (
        <div style={{
          position: 'absolute',
          top: '70px',
          left: '15px',
          background: 'rgba(0,0,0,0.95)',
          border: '2px solid rgba(255,255,255,0.3)',
          borderRadius: '12px',
          padding: '15px',
          zIndex: 101,
          minWidth: '220px',
          animation: 'slideIn 0.2s ease-out',
        }}>
          <h3 style={{ margin: '0 0 12px 0', color: 'white', fontSize: '18px' }}>Menu</h3>
          <button style={{
            width: '100%',
            padding: '12px',
            marginBottom: '8px',
            background: 'rgba(255,255,255,0.1)',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            color: 'white',
            fontSize: '16px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onClick={() => {
            // Open tutorial (placeholder)
            setShowMenu(false);
          }}
          >
            📖 Tutorial
          </button>
          <button style={{
            width: '100%',
            padding: '12px',
            background: 'rgba(255,255,255,0.1)',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            color: 'white',
            fontSize: '16px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onClick={() => {
            // Open settings overlay
            setShowSettings(true);
            setShowMenu(false);
          }}
          >
            ⚙️ Settings
          </button>
        </div>
      )}

      {/* Settings Panel - opened from Menu -> Settings */}
      {showSettings && (
        <div style={{
          position: 'absolute',
          top: '90px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(0,0,0,0.95)',
          border: '2px solid rgba(255,255,255,0.3)',
          borderRadius: '12px',
          padding: '16px',
          zIndex: 110,
          minWidth: '260px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
        }}>
          <h3 style={{ margin: '0 0 10px 0', color: 'white', fontSize: '18px' }}>Settings</h3>
          <div style={{ color: 'white', marginBottom: 12 }}>
            <label htmlFor="visual-quality-select" style={{ marginRight: 8 }}>Visual Quality</label>
            <select
              id="visual-quality-select"
              value={visualQuality}
              onChange={e => onQualityChange(e.target.value as 'low' | 'medium' | 'high')}
              style={{ fontSize: 14, padding: 6, borderRadius: 6 }}
            >
              <option value="low">Low (Best Performance)</option>
              <option value="medium">Medium (Balanced)</option>
              <option value="high">High (Best Quality)</option>
            </select>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button
              onClick={() => setShowSettings(false)}
              style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(255,255,255,0.08)', color: 'white', border: '1px solid rgba(255,255,255,0.2)' }}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Winner Banner */}
      {winner && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'linear-gradient(135deg, rgba(255,215,0,0.98) 0%, rgba(255,165,0,0.98) 100%)',
          border: '4px solid gold',
          borderRadius: '20px',
          padding: '40px 60px',
          zIndex: 103,
          textAlign: 'center',
          boxShadow: '0 0 60px rgba(255,215,0,0.7)',
          animation: 'victoryPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        }}>
          <h1 style={{ 
            margin: '0 0 20px 0', 
            fontSize: '42px',
            color: '#000',
            textShadow: '3px 3px 6px rgba(255,255,255,0.5)',
          }}>
            🏆 Victory! 🏆
          </h1>
          <p style={{ 
            margin: '0 0 30px 0',
            fontSize: '28px',
            color: '#000',
            fontWeight: 'bold',
          }}>
            {winner === 'Player1' ? 'Blue Player' : 'Red Player'} Wins!
          </p>
          <button style={{
            padding: '16px 45px',
            background: 'rgba(0,0,0,0.85)',
            border: '3px solid rgba(255,255,255,0.5)',
            borderRadius: '14px',
            color: 'white',
            fontSize: '20px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
          onClick={() => window.location.reload()}>
            New Game
          </button>
        </div>
      )}

      {/* Dynamic Action Panel */}
      {!winner && (
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          background: 'linear-gradient(0deg, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.85) 100%)',
          padding: '20px 15px',
          zIndex: 100,
          borderTop: '3px solid rgba(255,255,255,0.2)',
          boxShadow: '0 -4px 12px rgba(0,0,0,0.5)',
          minHeight: '140px',
          transition: 'all 0.3s ease',
        }}>
          {/* State A: Default State */}
          {panelState === 'default' && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '12px',
                marginBottom: '12px',
              }}>
                {/* Place Button */}
                <button
                  onClick={() => onPhaseSelect('placement')}
                  disabled={turn.hasPlaced || players[currentPlayerId].reinforcements === 0}
                  style={{
                    minHeight: '75px',
                    background: turn.hasPlaced 
                      ? 'linear-gradient(135deg, rgba(50,205,50,0.3) 0%, rgba(34,139,34,0.3) 100%)'
                      : players[currentPlayerId].reinforcements === 0
                      ? 'linear-gradient(135deg, rgba(100,100,100,0.3) 0%, rgba(80,80,80,0.3) 100%)'
                      : 'linear-gradient(135deg, rgba(50,205,50,0.9) 0%, rgba(34,139,34,0.9) 100%)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '12px',
                    color: 'white',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    cursor: turn.hasPlaced || players[currentPlayerId].reinforcements === 0 ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    opacity: turn.hasPlaced || players[currentPlayerId].reinforcements === 0 ? 0.5 : 1,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    transition: 'all 0.2s ease',
                    boxShadow: turn.hasPlaced ? 'inset 0 4px 8px rgba(0,0,0,0.3)' : '0 4px 8px rgba(0,0,0,0.2)',
                  }}
                >
                  <span style={{ fontSize: '26px' }}>📍</span>
                  <span>PLACE</span>
                  {turn.hasPlaced && <span style={{ fontSize: '18px' }}>✓</span>}
                </button>

                {/* Infuse Button */}
                <button
                  onClick={() => onPhaseSelect('infusion')}
                  disabled={turn.hasInfused}
                  style={{
                    minHeight: '75px',
                    background: turn.hasInfused 
                      ? 'linear-gradient(135deg, rgba(50,205,50,0.3) 0%, rgba(34,139,34,0.3) 100%)'
                      : 'linear-gradient(135deg, rgba(255,191,0,0.9) 0%, rgba(255,140,0,0.9) 100%)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '12px',
                    color: 'white',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    cursor: turn.hasInfused ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    opacity: turn.hasInfused ? 0.5 : 1,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    transition: 'all 0.2s ease',
                    boxShadow: turn.hasInfused ? 'inset 0 4px 8px rgba(0,0,0,0.3)' : '0 4px 8px rgba(0,0,0,0.2)',
                  }}
                >
                  <span style={{ fontSize: '26px' }}>⚡</span>
                  <span>INFUSE</span>
                  {turn.hasInfused && <span style={{ fontSize: '18px' }}>✓</span>}
                </button>

                {/* Move Button */}
                <button
                  onClick={() => onPhaseSelect('movement')}
                  disabled={turn.hasMoved}
                  style={{
                    minHeight: '75px',
                    background: turn.hasMoved 
                      ? 'linear-gradient(135deg, rgba(50,205,50,0.3) 0%, rgba(34,139,34,0.3) 100%)'
                      : 'linear-gradient(135deg, rgba(0,206,209,0.9) 0%, rgba(0,139,139,0.9) 100%)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '12px',
                    color: 'white',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    cursor: turn.hasMoved ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    opacity: turn.hasMoved ? 0.5 : 1,
                    fontFamily: 'system-ui, -apple-system, sans-serif',
                    transition: 'all 0.2s ease',
                    boxShadow: turn.hasMoved ? 'inset 0 4px 8px rgba(0,0,0,0.3)' : '0 4px 8px rgba(0,0,0,0.2)',
                  }}
                >
                  <span style={{ fontSize: '26px' }}>➡️</span>
                  <span>MOVE</span>
                  {turn.hasMoved && <span style={{ fontSize: '18px' }}>✓</span>}
                </button>
              </div>

              {/* End Turn Button */}
              <button
                onClick={onEndTurn}
                disabled={!allMandatoryDone}
                style={{
                  width: '100%',
                  minHeight: '60px',
                  background: allMandatoryDone 
                    ? 'linear-gradient(135deg, rgba(255,215,0,0.95) 0%, rgba(255,165,0,0.95) 100%)'
                    : 'linear-gradient(135deg, rgba(100,100,100,0.4) 0%, rgba(80,80,80,0.4) 100%)',
                  border: allMandatoryDone ? '3px solid gold' : '2px solid rgba(255,255,255,0.2)',
                  borderRadius: '12px',
                  color: allMandatoryDone ? '#000' : 'rgba(255,255,255,0.4)',
                  fontSize: '20px',
                  fontWeight: 'bold',
                  cursor: allMandatoryDone ? 'pointer' : 'not-allowed',
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                  boxShadow: allMandatoryDone ? '0 0 24px rgba(255,215,0,0.5)' : 'inset 0 2px 4px rgba(0,0,0,0.3)',
                  transition: 'all 0.2s ease',
                }}
              >
                END TURN {!allMandatoryDone && '(Complete all actions)'}
              </button>
            </div>
          )}

          {/* State B: Unit Selected (Own Unit) */}
          {panelState === 'unit-selected' && selectedVertex && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              {/* Unit Info Header */}
              <div style={{
                background: 'linear-gradient(135deg, rgba(74,144,226,0.2) 0%, rgba(208,2,27,0.2) 100%)',
                border: `2px solid ${playerColor}`,
                borderRadius: '12px',
                padding: '12px 15px',
                marginBottom: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: playerColor,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '20px',
                    boxShadow: `0 0 16px ${playerColor}`,
                  }}>
                    👤
                  </div>
                  <div>
                    <div style={{ color: 'white', fontSize: '16px', fontWeight: 'bold' }}>
                      {currentPlayerId === 'Player1' ? 'Blue' : 'Red'} Unit
                    </div>
                    <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '13px' }}>
                      Stack: {selectedVertex.stack.length} | Energy: ⚡{selectedVertex.energy}
                    </div>
                  </div>
                </div>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)' }}>
                  Layer {selectedVertex.layer}
                </div>
              </div>

              {/* Contextual Actions */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '10px',
              }}>
                <button
                  onClick={() => handleActionClick('move')}
                  disabled={!canMove}
                  style={{
                    minHeight: '65px',
                    background: canMove
                      ? 'linear-gradient(135deg, rgba(0,206,209,0.9) 0%, rgba(0,139,139,0.9) 100%)'
                      : 'linear-gradient(135deg, rgba(100,100,100,0.3) 0%, rgba(80,80,80,0.3) 100%)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '10px',
                    color: 'white',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: canMove ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    opacity: canMove ? 1 : 0.4,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <span style={{ fontSize: '24px' }}>➡️</span>
                  <span>MOVE</span>
                </button>

                <button
                  onClick={() => handleActionClick('attack')}
                  disabled={!canAttack}
                  style={{
                    minHeight: '65px',
                    background: canAttack
                      ? 'linear-gradient(135deg, rgba(220,20,60,0.9) 0%, rgba(139,0,0,0.9) 100%)'
                      : 'linear-gradient(135deg, rgba(100,100,100,0.3) 0%, rgba(80,80,80,0.3) 100%)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '10px',
                    color: 'white',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: canAttack ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    opacity: canAttack ? 1 : 0.4,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <span style={{ fontSize: '24px' }}>⚔️</span>
                  <span>ATTACK</span>
                </button>

                <button
                  onClick={() => handleActionClick('infuse')}
                  disabled={!canInfuse}
                  style={{
                    minHeight: '65px',
                    background: canInfuse
                      ? 'linear-gradient(135deg, rgba(255,191,0,0.9) 0%, rgba(255,140,0,0.9) 100%)'
                      : 'linear-gradient(135deg, rgba(100,100,100,0.3) 0%, rgba(80,80,80,0.3) 100%)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '10px',
                    color: 'white',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: canInfuse ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    opacity: canInfuse ? 1 : 0.4,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <span style={{ fontSize: '24px' }}>⚡</span>
                  <span>INFUSE</span>
                </button>
              </div>
            </div>
          )}

          {/* State C: Enemy Unit Selected */}
          {panelState === 'enemy-selected' && selectedVertex && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                background: 'linear-gradient(135deg, rgba(139,0,0,0.3) 0%, rgba(139,0,0,0.1) 100%)',
                border: '2px solid rgba(220,20,60,0.5)',
                borderRadius: '12px',
                padding: '20px',
                textAlign: 'center',
              }}>
                <div style={{
                  fontSize: '48px',
                  marginBottom: '12px',
                }}>
                  🛡️
                </div>
                <div style={{ color: 'white', fontSize: '18px', fontWeight: 'bold', marginBottom: '8px' }}>
                  Enemy Unit
                </div>
                <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '14px', marginBottom: '12px' }}>
                  Stack: {selectedVertex.stack.length} | Energy: ⚡{selectedVertex.energy} | Layer {selectedVertex.layer}
                </div>
                <div style={{ 
                  color: 'rgba(255,255,255,0.5)', 
                  fontSize: '13px',
                  fontStyle: 'italic',
                }}>
                  Select your own unit to take actions
                </div>
              </div>
            </div>
          )}

          {/* State D: Action Active (e.g., Move Mode) */}
          {panelState === 'action-active' && activeAction && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                background: activeAction === 'move' 
                  ? 'linear-gradient(135deg, rgba(0,206,209,0.3) 0%, rgba(0,139,139,0.3) 100%)'
                  : activeAction === 'attack'
                  ? 'linear-gradient(135deg, rgba(220,20,60,0.3) 0%, rgba(139,0,0,0.3) 100%)'
                  : 'linear-gradient(135deg, rgba(255,191,0,0.3) 0%, rgba(255,140,0,0.3) 100%)',
                border: `3px solid ${
                  activeAction === 'move' ? '#00CED1' : 
                  activeAction === 'attack' ? '#DC143C' : '#FFD700'
                }`,
                borderRadius: '12px',
                padding: '20px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '48px', marginBottom: '12px' }}>
                  {activeAction === 'move' ? '➡️' : activeAction === 'attack' ? '⚔️' : '⚡'}
                </div>
                <div style={{ color: 'white', fontSize: '20px', fontWeight: 'bold', marginBottom: '8px' }}>
                  {activeAction === 'move' ? 'Select Move Destination' : 
                   activeAction === 'attack' ? 'Select Attack Target' : 
                   'Select Unit to Infuse'}
                </div>
                <div style={{ 
                  color: 'rgba(255,255,255,0.8)', 
                  fontSize: '14px',
                  marginBottom: '16px',
                }}>
                  {activeAction === 'move' && 'Tap a highlighted tile to move'}
                  {activeAction === 'attack' && 'Tap an enemy unit to attack'}
                  {activeAction === 'infuse' && 'Tap a friendly unit to add energy'}
                </div>
                <button
                  onClick={handleCancelAction}
                  style={{
                    padding: '12px 30px',
                    background: 'rgba(255,255,255,0.2)',
                    border: '2px solid rgba(255,255,255,0.4)',
                    borderRadius: '10px',
                    color: 'white',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.3)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
                >
                  ✕ Cancel
                </button>
              </div>
            </div>
          )}

          {/* Message State */}
          {panelState === 'message' && message && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                background: 'linear-gradient(135deg, rgba(220,20,60,0.3) 0%, rgba(139,0,0,0.3) 100%)',
                border: '3px solid rgba(220,20,60,0.6)',
                borderRadius: '12px',
                padding: '20px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '48px', marginBottom: '12px' }}>
                  ⚠️
                </div>
                <div style={{ color: 'white', fontSize: '18px', fontWeight: 'bold', marginBottom: '16px' }}>
                  {message}
                </div>
                <button
                  onClick={() => setMessage(null)}
                  style={{
                    padding: '12px 30px',
                    background: 'rgba(255,255,255,0.2)',
                    border: '2px solid rgba(255,255,255,0.4)',
                    borderRadius: '10px',
                    color: 'white',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.3)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
                >
                  OK
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* CSS Animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
            transform: scale(1);
          }
          50% {
            opacity: 0.7;
            transform: scale(1.1);
          }
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes victoryPop {
          0% {
            transform: translate(-50%, -50%) scale(0.5);
            opacity: 0;
          }
          70% {
            transform: translate(-50%, -50%) scale(1.05);
          }
          100% {
            transform: translate(-50%, -50%) scale(1);
            opacity: 1;
          }
        }

        button:hover:not(:disabled) {
          filter: brightness(1.15);
          transform: translateY(-2px);
          box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
        }

        button:active:not(:disabled) {
          transform: translateY(0);
          box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
      `}</style>
    </>
  );
};

// Demo wrapper with mock data

export default MobileHUD;