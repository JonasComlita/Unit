import React, { useState, useEffect } from 'react';
import { getForce } from '../game/gameLogic';
import { BOARD_CONFIG, GAME_RULES, getOccupationRequirement } from '../game/constants';

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
  onBack?: () => void;
}

const MobileHUD: React.FC<MobileHUDProps> = ({ gameState, onEndTurn, activePhase, onPhaseSelect, onUndo, undoCount, visualQuality, onQualityChange, onBack }) => {
  const { turn, currentPlayerId, winner, players, selectedVertexId, validAttackTargets, validMoveTargets, vertices } = gameState;
  const [showMenu, setShowMenu] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showTutorial, setShowTutorial] = useState(false);
  const [showQuickHelp, setShowQuickHelp] = useState(false);
  const [quickHelpRightPx, setQuickHelpRightPx] = useState<number | null>(384);
  const [panelState, setPanelState] = useState<PanelState>('default');
  const [message, setMessage] = useState<string | null>(null);
  const [activeAction, setActiveAction] = useState<'move' | 'attack' | 'infuse' | 'pincer' | null>(null);

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

  // Position the Quick Help button adjacent to the DebugPanel by measuring it.
  useEffect(() => {
    let observer: ResizeObserver | null = null;
    const el = document.getElementById('debug-panel-inner');
    const gap = 8; // px gap between quick-help and debug panel
    const baseRight = 12; // DebugPanel right offset
    const update = () => {
      if (!el) {
        setQuickHelpRightPx(384);
        return;
      }
      const rect = el.getBoundingClientRect();
      const panelWidth = Math.round(rect.width || 0);
      setQuickHelpRightPx(baseRight + panelWidth + gap);
    };

    if (el && (window as any).ResizeObserver) {
      // ResizeObserver type from window may differ in this TS environment; use any to construct.
      // @ts-ignore
      observer = new (window as any).ResizeObserver(update);
      observer!.observe(el as Element);
    }
    // initial update and on window resize
    update();
    window.addEventListener('resize', update);
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', update);
    };
  }, []);

  const handleActionClick = (action: 'move' | 'attack' | 'infuse' | 'pincer') => {
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
  // Check if selected vertex can participate in any pincer attacks
  const canPincer = selectedVertexId && gameState.validPincerTargets &&
    Object.entries(gameState.validPincerTargets).some(([targetId, originIds]) =>
      originIds.includes(selectedVertexId)
    );

  return (
    <>
      {/* Professional Status Bar */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '60px',
        background: '#1a1a1a',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px',
        zIndex: 100,
        borderBottom: '1px solid #444',
        fontFamily: '"Times New Roman", Times, serif',
      }}>
        <button
          onClick={() => setShowMenu(!showMenu)}
          style={{
            padding: '8px 16px',
            background: '#333',
            border: '1px solid #555',
            borderRadius: '2px',
            color: '#ddd',
            fontSize: '14px',
            fontFamily: 'inherit',
            textTransform: 'uppercase',
            cursor: 'pointer',
            letterSpacing: '1px',
          }}
        >
          Menu
        </button>

        {/* Turn Info */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '20px',
          flex: 1,
          justifyContent: 'center',
          color: '#eee',
        }}>
          <div style={{
            fontSize: '16px',
            fontWeight: 'normal',
            letterSpacing: '0.5px',
          }}>
            {currentPlayerId === 'Player1' ? 'PLAYER 1 (BLUE)' : 'PLAYER 2 (RED)'}
          </div>
          <div style={{ width: '1px', height: '20px', background: '#444' }} />
          <div style={{
            fontSize: '16px',
          }}>
            TURN {turn.turnNumber || 1}
          </div>
        </div>

        {/* Player summary */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '15px',
          color: '#ccc',
          fontSize: '14px',
        }}>
          {(() => {
            const totals = Object.values(vertices).reduce((acc, v) => {
              const ownPieces = v.stack.filter(p => p.player === currentPlayerId).length;
              acc.pieces += ownPieces;
              if (v.stack.length > 0 && v.stack[0].player === currentPlayerId) acc.energy += v.energy;
              return acc;
            }, { pieces: 0, energy: 0 });
            return (
              <>
                <div>UNITS: {totals.pieces}</div>
                <div>ENERGY: {totals.energy}</div>
              </>
            );
          })()}
        </div>
        {/* Undo Button */}
        <div style={{ marginLeft: 15 }}>
          <button
            onClick={() => onUndo && onUndo()}
            disabled={!undoCount || undoCount <= 0}
            style={{
              padding: '8px 16px',
              background: '#333',
              border: '1px solid #555',
              borderRadius: '2px',
              color: undoCount && undoCount > 0 ? '#ddd' : '#555',
              cursor: undoCount && undoCount > 0 ? 'pointer' : 'not-allowed',
              fontFamily: 'inherit',
              textTransform: 'uppercase',
              fontSize: '13px',
            }}
          >
            Undo
          </button>
        </div>
      </div>

      {/* Floating Quick Help Button (adjacent to DebugPanel) */}
      <div style={{ position: 'absolute', top: 70, right: quickHelpRightPx ? `${quickHelpRightPx}px` : '384px', zIndex: 199 }}>
        <button
          onClick={() => setShowQuickHelp(s => !s)}
          title="Quick Help"
          style={{
            width: 40,
            height: 40,
            borderRadius: 2,
            background: '#333',
            border: '1px solid #555',
            color: '#ddd',
            fontSize: '18px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: '"Times New Roman", Times, serif',
            fontWeight: 'bold',
          }}
        >
          ?
        </button>
      </div>

      {/* Menu Overlay */}
      {showMenu && (
        <div style={{
          position: 'absolute',
          top: '60px',
          left: '20px',
          background: '#222',
          border: '1px solid #444',
          borderRadius: '2px',
          padding: '15px',
          zIndex: 101,
          minWidth: '200px',
          animation: 'slideIn 0.2s ease-out',
          fontFamily: '"Times New Roman", Times, serif',
        }}>
          <h3 style={{ margin: '0 0 15px 0', color: '#eee', fontSize: '16px', textTransform: 'uppercase', letterSpacing: '1px', borderBottom: '1px solid #444', paddingBottom: '10px' }}>Menu</h3>
          <button style={{
            width: '100%',
            padding: '10px',
            marginBottom: '8px',
            background: '#333',
            border: '1px solid #555',
            borderRadius: '2px',
            color: '#ddd',
            fontSize: '14px',
            cursor: 'pointer',
            textAlign: 'left',
            fontFamily: 'inherit',
          }}
            onClick={() => {
              // Open tutorial modal
              setShowMenu(false);
              setShowTutorial(true);
            }}
          >
            Tutorial
          </button>
          {/* Quick Help moved to floating button next to DebugPanel (removed from menu) */}
          <button style={{
            width: '100%',
            padding: '10px',
            marginBottom: '8px',
            background: '#333',
            border: '1px solid #555',
            borderRadius: '2px',
            color: '#ddd',
            fontSize: '14px',
            cursor: 'pointer',
            textAlign: 'left',
            fontFamily: 'inherit',
          }}
            onClick={() => {
              // Open settings overlay
              setShowSettings(true);
              setShowMenu(false);
            }}
          >
            Settings
          </button>
          <button style={{
            width: '100%',
            padding: '10px',
            background: '#D0021B',
            border: '1px solid #ff4444',
            borderRadius: '2px',
            color: '#fff',
            fontSize: '14px',
            cursor: 'pointer',
            textAlign: 'left',
            fontFamily: 'inherit',
            fontWeight: 'bold',
          }}
            onClick={() => {
              if (window.confirm('Are you sure you want to quit? Your current game will be lost.')) {
                setShowMenu(false);
                onBack && onBack();
              }
            }}
          >
            Quit Game
          </button>
        </div>
      )}

      {/* Settings Panel - opened from Menu -> Settings */}
      {showSettings && (
        <div style={{
          position: 'absolute',
          top: '80px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: '#222',
          border: '1px solid #444',
          borderRadius: '2px',
          padding: '20px',
          zIndex: 110,
          minWidth: '300px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          fontFamily: '"Times New Roman", Times, serif',
        }}>
          <h3 style={{ margin: '0 0 15px 0', color: '#eee', fontSize: '16px', textTransform: 'uppercase', letterSpacing: '1px', borderBottom: '1px solid #444', paddingBottom: '10px' }}>Settings</h3>
          <div style={{ color: '#ddd', marginBottom: 20, fontSize: '14px' }}>
            <label htmlFor="visual-quality-select" style={{ marginRight: 10 }}>Visual Quality:</label>
            <select
              id="visual-quality-select"
              value={visualQuality}
              onChange={e => onQualityChange(e.target.value as 'low' | 'medium' | 'high')}
              style={{ fontSize: 14, padding: '5px', borderRadius: '2px', background: '#333', color: '#eee', border: '1px solid #555', fontFamily: 'inherit' }}
            >
              <option value="low">Low (Performance)</option>
              <option value="medium">Medium (Balanced)</option>
              <option value="high">High (Quality)</option>
            </select>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button
              onClick={() => setShowSettings(false)}
              style={{ padding: '8px 16px', borderRadius: '2px', background: '#333', color: '#ddd', border: '1px solid #555', cursor: 'pointer', fontFamily: 'inherit', textTransform: 'uppercase', fontSize: '12px' }}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Tutorial Modal */}
      {showTutorial && (
        <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.9)', zIndex: 250, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: '"Times New Roman", Times, serif' }}>
          <div style={{ width: 'min(920px, 95%)', maxHeight: '85%', background: '#1a1a1a', border: '1px solid #444', borderRadius: '2px', padding: 30, color: '#ddd', overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, borderBottom: '1px solid #444', paddingBottom: 10 }}>
              <h2 style={{ margin: 0, fontSize: '20px', textTransform: 'uppercase', letterSpacing: '1px' }}>How to Play</h2>
              <div style={{ display: 'flex', gap: 10 }}>
                <button onClick={() => setShowQuickHelp(s => !s)} style={{ padding: '8px 12px', borderRadius: '2px', background: '#333', color: '#ddd', border: '1px solid #555', cursor: 'pointer', fontFamily: 'inherit', fontSize: '12px', textTransform: 'uppercase' }}>Toggle Numeric Help</button>
                <button onClick={() => setShowTutorial(false)} style={{ padding: '8px 12px', borderRadius: '2px', background: '#333', color: '#ddd', border: '1px solid #555', cursor: 'pointer', fontFamily: 'inherit', fontSize: '12px', textTransform: 'uppercase' }}>Close</button>
              </div>
            </div>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Goal</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>Each player controls two home corners. Win by occupying all of your opponent’s home corners with your pieces (the piece on top of the stack controls a corner).</p>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Turn Rewards</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>At the start of your turn you gain <strong>+1 piece</strong> and <strong>+1 energy</strong>. New pieces must be placed onto one of your home corners. New energy can be infused into any vertex that already contains your pieces.</p>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Three Actions per Turn</h3>
              <ol style={{ fontSize: '14px', lineHeight: '1.5', paddingLeft: '20px' }}>
                <li><strong>Place</strong> — place your reinforcement on a home corner.</li>
                <li><strong>Infuse</strong> — add one energy to a friendly vertex.</li>
                <li><strong>Move</strong> — move a stack or single piece to an adjacent vertex (can also initiate attacks or pincers).</li>
              </ol>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Board Layers & Gravity</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>The board has 5 layers (outer → center → outer): <code>[3×3, 5×5, 7×7, 5×5, 3×3]</code>. Each layer has a gravity value that reduces effective force. Gravity values: <strong>{BOARD_CONFIG.layerGravity.join(', ')}</strong>.</p>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Force</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>Force for a vertex = (stack size × vertex energy) / layer gravity. Force is clamped to a maximum of <strong>{GAME_RULES.forceCapMax}</strong>. Force determines attack outcomes and some movement rules.</p>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Movement Requirements</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>Moving a single piece into an empty vertex requires the source to meet the destination layer’s occupation thresholds (pieces, energy, and force). Multi-piece stacks can move more freely.</p>
              <ul style={{ fontSize: '14px', lineHeight: '1.5', paddingLeft: '20px' }}>
                {BOARD_CONFIG.layout.map((size, idx) => {
                  const req = getOccupationRequirement(idx);
                  return (
                    <li key={idx}>Layer {idx} ({size}×{size}): minPieces={req.minPieces}, minEnergy={req.minEnergy}, minForce={req.minForce}</li>
                  );
                })}
              </ul>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Attacking</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>Select a friendly vertex and then an adjacent enemy-occupied vertex to attack. Compare attacker vs defender force. The result produces <em>newPieces</em> = |attackerPieces − defenderPieces| and <em>newEnergy</em> = |attackerEnergy − defenderEnergy|. If attackerForce &gt; defenderForce the attacker conquers the vertex (replaced with newPieces of attacker); otherwise the defender holds and attacker is removed.</p>
            </section>

            <section style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: '16px', color: '#fff', marginBottom: '5px' }}>Pincer</h3>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>A pincer attack uses two or more neighboring friendly origins against one target. Attacker force is the product of origin forces (then clamped). Pieces and energy sum across origins. If attackerForce &gt; defenderForce the target is conquered; otherwise origins are cleared. The maximum pincer participants is <strong>{GAME_RULES.maxPincerParticipants}</strong>.</p>
            </section>

            <section style={{ marginTop: 15, paddingTop: 15, borderTop: '1px dashed #444' }}>
              <h4 style={{ fontSize: '14px', color: '#fff', marginBottom: '5px' }}>Examples</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>See the in-game examples in the manual or try test attacks in the sandbox.</p>
            </section>
          </div>
        </div>
      )}

      {/* Quick Help Overlay (numeric rules) */}
      {showQuickHelp && (
        <div style={{ position: 'absolute', top: 84, right: 12, zIndex: 240 }}>
          <div style={{ width: 320, background: 'rgba(8,10,18,0.95)', color: 'white', borderRadius: 10, padding: 12, border: '1px solid rgba(255,255,255,0.06)', fontSize: 13, fontFamily: 'monospace' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <strong>Quick Numeric Help</strong>
              <button onClick={() => setShowQuickHelp(false)} style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}>✕</button>
            </div>
            <div><strong>Gravity:</strong> [{BOARD_CONFIG.layerGravity.join(', ')}]</div>
            <div style={{ marginTop: 8 }}><strong>Occupation thresholds:</strong></div>
            <div style={{ marginTop: 6 }}>
              {BOARD_CONFIG.layout.map((size, idx) => {
                const req = getOccupationRequirement(idx);
                return (<div key={idx}>L{idx} ({size}×{size}): pieces&gt;={req.minPieces}, energy&gt;={req.minEnergy}, force&gt;={req.minForce}</div>);
              })}
            </div>
            <div style={{ marginTop: 8 }}><strong>Force formula:</strong> (stackSize × energy) / gravity, capped at {GAME_RULES.forceCapMax}</div>
            <div style={{ marginTop: 8 }}><strong>Reinforcements:</strong> +{GAME_RULES.reinforcementsPerTurn} per turn</div>
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
          background: '#111',
          border: '2px solid #fff',
          borderRadius: '2px',
          padding: '40px 60px',
          zIndex: 103,
          textAlign: 'center',
          boxShadow: '0 0 60px rgba(0,0,0,0.8)',
          fontFamily: '"Times New Roman", Times, serif',
        }}>
          <h1 style={{
            margin: '0 0 20px 0',
            fontSize: '36px',
            color: '#fff',
            textTransform: 'uppercase',
            letterSpacing: '4px',
          }}>
            Victory
          </h1>
          <p style={{
            margin: '0 0 30px 0',
            fontSize: '20px',
            color: '#aaa',
            textTransform: 'uppercase',
            letterSpacing: '2px',
          }}>
            {winner === 'Player1' ? 'Blue Player' : 'Red Player'} Wins
          </p>
          <button style={{
            padding: '15px 40px',
            background: '#333',
            border: '1px solid #666',
            borderRadius: '2px',
            color: '#fff',
            fontSize: '16px',
            cursor: 'pointer',
            fontFamily: 'inherit',
            textTransform: 'uppercase',
            letterSpacing: '2px',
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
          background: '#1a1a1a',
          padding: '20px 20px',
          zIndex: 100,
          borderTop: '1px solid #444',
          minHeight: '140px',
          fontFamily: '"Times New Roman", Times, serif',
        }}>
          {/* State A: Default State */}
          {panelState === 'default' && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '15px',
                marginBottom: '15px',
              }}>
                {/* Place Button */}
                <button
                  onClick={() => onPhaseSelect('placement')}
                  disabled={turn.hasPlaced || players[currentPlayerId].reinforcements === 0}
                  style={{
                    minHeight: '60px',
                    background: turn.hasPlaced ? '#222' : '#333',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: turn.hasPlaced ? '#666' : '#eee',
                    fontSize: '14px',
                    fontWeight: 'normal',
                    cursor: turn.hasPlaced || players[currentPlayerId].reinforcements === 0 ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: turn.hasPlaced || players[currentPlayerId].reinforcements === 0 ? 0.6 : 1,
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  <span>PLACE</span>
                  {turn.hasPlaced && <span style={{ fontSize: '12px' }}>[DONE]</span>}
                </button>

                {/* Infuse Button */}
                <button
                  onClick={() => onPhaseSelect('infusion')}
                  disabled={turn.hasInfused}
                  style={{
                    minHeight: '60px',
                    background: turn.hasInfused ? '#222' : '#333',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: turn.hasInfused ? '#666' : '#eee',
                    fontSize: '14px',
                    fontWeight: 'normal',
                    cursor: turn.hasInfused ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: turn.hasInfused ? 0.6 : 1,
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  <span>INFUSE</span>
                  {turn.hasInfused && <span style={{ fontSize: '12px' }}>[DONE]</span>}
                </button>

                {/* Move Button */}
                <button
                  onClick={() => onPhaseSelect('movement')}
                  disabled={turn.hasMoved}
                  style={{
                    minHeight: '60px',
                    background: turn.hasMoved ? '#222' : '#333',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: turn.hasMoved ? '#666' : '#eee',
                    fontSize: '14px',
                    fontWeight: 'normal',
                    cursor: turn.hasMoved ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: turn.hasMoved ? 0.6 : 1,
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  <span>MOVE</span>
                  {turn.hasMoved && <span style={{ fontSize: '12px' }}>[DONE]</span>}
                </button>
              </div>

              {/* End Turn Button */}
              <button
                onClick={onEndTurn}
                disabled={!allMandatoryDone}
                style={{
                  width: '100%',
                  minHeight: '50px',
                  background: allMandatoryDone ? '#ddd' : '#222',
                  border: allMandatoryDone ? '1px solid #fff' : '1px solid #444',
                  borderRadius: '2px',
                  color: allMandatoryDone ? '#000' : '#555',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  cursor: allMandatoryDone ? 'pointer' : 'not-allowed',
                  fontFamily: 'inherit',
                  textTransform: 'uppercase',
                  letterSpacing: '2px',
                }}
              >
                END TURN {!allMandatoryDone && '(ACTIONS REMAINING)'}
              </button>
            </div>
          )}

          {/* State B: Unit Selected (Own Unit) */}
          {panelState === 'unit-selected' && selectedVertex && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              {/* Unit Info Header */}
              <div style={{
                background: '#222',
                border: '1px solid #444',
                borderRadius: '2px',
                padding: '10px 15px',
                marginBottom: '15px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <div style={{
                    fontSize: '14px',
                    fontWeight: 'bold',
                    color: '#eee',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}>
                    {currentPlayerId === 'Player1' ? 'Blue Unit' : 'Red Unit'}
                  </div>
                  <div style={{ color: '#888', fontSize: '13px' }}>
                    {(() => {
                      const gravity = BOARD_CONFIG.layerGravity[selectedVertex.layer] ?? 1;
                      const force = getForce(selectedVertex as any);
                      return (
                        <div style={{ display: 'flex', gap: 15, alignItems: 'center' }}>
                          <div>STACK: {selectedVertex.stack.length}</div>
                          <div>ENERGY: {selectedVertex.energy}</div>
                          <div>GRAVITY: {gravity}</div>
                          <div>FORCE: {force.toFixed(2)}</div>
                        </div>
                      );
                    })()}
                  </div>
                </div>
                <div style={{ fontSize: '12px', color: '#666', textTransform: 'uppercase' }}>
                  LAYER {selectedVertex.layer}
                </div>
              </div>

              {/* Contextual Actions */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: canPincer ? 'repeat(4, 1fr)' : 'repeat(3, 1fr)',
                gap: '10px',
              }}>
                <button
                  onClick={() => handleActionClick('move')}
                  disabled={!canMove}
                  style={{
                    minHeight: '50px',
                    background: canMove ? '#333' : '#222',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: canMove ? '#eee' : '#555',
                    fontSize: '13px',
                    fontWeight: 'normal',
                    cursor: canMove ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: canMove ? 1 : 0.6,
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  <span>MOVE</span>
                </button>

                <button
                  onClick={() => handleActionClick('attack')}
                  disabled={!canAttack}
                  style={{
                    minHeight: '50px',
                    background: canAttack ? '#333' : '#222',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: canAttack ? '#eee' : '#555',
                    fontSize: '13px',
                    fontWeight: 'normal',
                    cursor: canAttack ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: canAttack ? 1 : 0.6,
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  <span>ATTACK</span>
                </button>

                {canPincer && (
                  <button
                    onClick={() => handleActionClick('pincer')}
                    style={{
                      minHeight: '50px',
                      background: '#333',
                      border: '1px solid #555',
                      borderRadius: '2px',
                      color: '#eee',
                      fontSize: '13px',
                      fontWeight: 'normal',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '4px',
                      fontFamily: 'inherit',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                    }}
                  >
                    <span>PINCER</span>
                  </button>
                )}

                <button
                  onClick={() => handleActionClick('infuse')}
                  disabled={!canInfuse}
                  style={{
                    minHeight: '50px',
                    background: canInfuse ? '#333' : '#222',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: canInfuse ? '#eee' : '#555',
                    fontSize: '13px',
                    fontWeight: 'normal',
                    cursor: canInfuse ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: canInfuse ? 1 : 0.6,
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  <span>INFUSE</span>
                </button>
              </div>

              {/* Back Button */}
              <button
                onClick={() => onBack && onBack()}
                style={{
                  width: '100%',
                  minHeight: '40px',
                  marginTop: '10px',
                  background: '#222',
                  border: '1px solid #444',
                  borderRadius: '2px',
                  color: '#aaa',
                  fontSize: '14px',
                  fontWeight: 'normal',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                }}
              >
                Back
              </button>
            </div>
          )}

          {/* State C: Enemy Unit Selected */}
          {panelState === 'enemy-selected' && selectedVertex && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                background: '#222',
                border: '1px solid #444',
                borderRadius: '2px',
                padding: '15px',
                textAlign: 'center',
              }}>
                <div style={{ color: '#eee', fontSize: '16px', fontWeight: 'bold', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  Enemy Unit
                </div>
                <div style={{ color: '#888', fontSize: '14px', marginBottom: '12px' }}>
                  STACK: {selectedVertex.stack.length} | ENERGY: {selectedVertex.energy} | GRAVITY: {BOARD_CONFIG.layerGravity[selectedVertex.layer]} | FORCE: {getForce(selectedVertex as any)}
                </div>
                <div style={{
                  color: '#666',
                  fontSize: '13px',
                  fontStyle: 'italic',
                }}>
                  Select your own unit to take actions
                </div>
              </div>

              {/* Back Button */}
              <button
                onClick={() => onBack && onBack()}
                style={{
                  width: '100%',
                  minHeight: '40px',
                  marginTop: '12px',
                  background: '#222',
                  border: '1px solid #444',
                  borderRadius: '2px',
                  color: '#aaa',
                  fontSize: '14px',
                  fontWeight: 'normal',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                }}
              >
                Back
              </button>
            </div>
          )}

          {/* State D: Action Active (e.g., Move Mode) */}
          {panelState === 'action-active' && activeAction && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                background: '#222',
                border: '1px solid #444',
                borderRadius: '2px',
                padding: '20px',
                textAlign: 'center',
              }}>
                <div style={{ color: '#eee', fontSize: '18px', fontWeight: 'bold', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  {activeAction === 'move' ? 'Select Destination' :
                    activeAction === 'attack' ? 'Select Target' :
                      activeAction === 'pincer' ? 'Select Pincer Target' :
                        'Select Unit to Infuse'}
                </div>
                <div style={{
                  color: '#888',
                  fontSize: '14px',
                  marginBottom: '16px',
                }}>
                  {activeAction === 'move' && 'Tap a highlighted tile to move'}
                  {activeAction === 'attack' && 'Tap an enemy unit to attack'}
                  {activeAction === 'pincer' && 'Tap a highlighted enemy to execute pincer attack'}
                  {activeAction === 'infuse' && 'Tap a friendly unit to add energy'}
                </div>
                <button
                  onClick={handleCancelAction}
                  style={{
                    padding: '10px 25px',
                    background: '#333',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: '#ddd',
                    fontSize: '14px',
                    fontWeight: 'normal',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Message State */}
          {panelState === 'message' && message && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              <div style={{
                background: '#222',
                border: '1px solid #444',
                borderRadius: '2px',
                padding: '20px',
                textAlign: 'center',
              }}>
                <div style={{ color: '#eee', fontSize: '16px', fontWeight: 'bold', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  {message}
                </div>
                <button
                  onClick={() => setMessage(null)}
                  style={{
                    padding: '10px 25px',
                    background: '#333',
                    border: '1px solid #555',
                    borderRadius: '2px',
                    color: '#ddd',
                    fontSize: '14px',
                    fontWeight: 'normal',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
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
          background: #444 !important;
        }

        button:active:not(:disabled) {
          background: #222 !important;
        }
      `}</style>
    </>
  );
};

// Demo wrapper with mock data

export default MobileHUD;