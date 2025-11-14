import React, { useState } from 'react';
import { GameState } from '../game/types';
import { ActionPhase } from '../hooks/useGame';

// Mobile HUD Component
const MobileHUD: React.FC<{ 
    gameState: GameState; 
    onEndTurn: () => void;
    activePhase: ActionPhase | null;
    onPhaseSelect: (phase: ActionPhase) => void;
}> = ({ gameState, onEndTurn, activePhase, onPhaseSelect }) => {
    const { turn, currentPlayerId, winner, players, selectedVertexId, validAttackTargets, validPincerTargets } = gameState;
    const [showMenu, setShowMenu] = useState(false);
    const [showCombatMenu, setShowCombatMenu] = useState(false);
    const allMandatoryDone = turn.hasPlaced && turn.hasInfused && turn.hasMoved;

    const playerColor = currentPlayerId === 'Player1' ? '#4A90E2' : '#D0021B';
    const canAttack = selectedVertexId && validAttackTargets.length > 0;
    const canPincer = selectedVertexId && validPincerTargets && Object.keys(validPincerTargets).length > 0;

    return (
        <>
            {/* Top Bar */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '60px',
                background: 'linear-gradient(180deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.7) 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 15px',
                zIndex: 100,
                borderBottom: `3px solid ${playerColor}`,
            }}>
                <button 
                    onClick={() => setShowMenu(!showMenu)}
                    style={{
                        width: '44px',
                        height: '44px',
                        background: 'rgba(255,255,255,0.1)',
                        border: '2px solid rgba(255,255,255,0.3)',
                        borderRadius: '8px',
                        color: 'white',
                        fontSize: '24px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                    }}
                >
                    ☰
                </button>

                <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '12px',
                    flex: 1,
                    justifyContent: 'center',
                }}>
                    <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        background: playerColor,
                        boxShadow: `0 0 12px ${playerColor}`,
                    }} />
                    <span style={{ 
                        color: 'white', 
                        fontSize: '20px', 
                        fontWeight: 'bold',
                        fontFamily: 'system-ui, -apple-system, sans-serif',
                    }}>
                        {currentPlayerId === 'Player1' ? 'Player 1' : 'Player 2'}
                    </span>
                </div>

                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    background: 'rgba(255,255,255,0.1)',
                    padding: '8px 12px',
                    borderRadius: '20px',
                    border: '2px solid rgba(255,255,255,0.2)',
                }}>
                    <span style={{ fontSize: '20px' }}>⚡</span>
                    <span style={{ 
                        color: 'white', 
                        fontSize: '18px', 
                        fontWeight: 'bold',
                        minWidth: '20px',
                        textAlign: 'center',
                    }}>
                        {players[currentPlayerId].reinforcements}
                    </span>
                </div>
            </div>

            {/* Menu Overlay */}
            {showMenu && (
                <div style={{
                    position: 'absolute',
                    top: '60px',
                    left: '15px',
                    background: 'rgba(0,0,0,0.95)',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderRadius: '12px',
                    padding: '15px',
                    zIndex: 101,
                    minWidth: '200px',
                }}>
                    <h3 style={{ margin: '0 0 10px 0', color: 'white', fontSize: '18px' }}>Menu</h3>
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
                    }}>
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
                    }}>
                        ⚙️ Settings
                    </button>
                </div>
            )}

            {/* Combat Menu */}
            {showCombatMenu && selectedVertexId && (canAttack || canPincer) && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    background: 'rgba(0,0,0,0.95)',
                    border: '3px solid rgba(255,255,255,0.4)',
                    borderRadius: '16px',
                    padding: '20px',
                    zIndex: 102,
                    minWidth: '280px',
                }}>
                    <h3 style={{ 
                        margin: '0 0 15px 0', 
                        color: 'white', 
                        fontSize: '20px',
                        textAlign: 'center',
                    }}>
                        Combat Actions
                    </h3>
                    <p style={{ 
                        color: 'rgba(255,255,255,0.7)', 
                        fontSize: '14px',
                        textAlign: 'center',
                        margin: '0 0 15px 0',
                    }}>
                        Select a target on the board
                    </p>
                    <button 
                        onClick={() => setShowCombatMenu(false)}
                        style={{
                            width: '100%',
                            padding: '14px',
                            background: 'rgba(255,255,255,0.1)',
                            border: '2px solid rgba(255,255,255,0.3)',
                            borderRadius: '10px',
                            color: 'white',
                            fontSize: '16px',
                            cursor: 'pointer',
                        }}
                    >
                        Cancel
                    </button>
                </div>
            )}

            {/* Winner Banner */}
            {winner && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    background: 'linear-gradient(135deg, rgba(255,215,0,0.95) 0%, rgba(255,165,0,0.95) 100%)',
                    border: '4px solid gold',
                    borderRadius: '20px',
                    padding: '40px 60px',
                    zIndex: 103,
                    textAlign: 'center',
                    boxShadow: '0 0 40px rgba(255,215,0,0.6)',
                }}>
                    <h1 style={{ 
                        margin: '0 0 20px 0', 
                        fontSize: '36px',
                        color: '#000',
                        textShadow: '2px 2px 4px rgba(255,255,255,0.5)',
                    }}>
                        🏆 Victory! 🏆
                    </h1>
                    <p style={{ 
                        margin: '0 0 30px 0',
                        fontSize: '24px',
                        color: '#000',
                        fontWeight: 'bold',
                    }}>
                        {winner === 'Player1' ? 'Player 1' : 'Player 2'} Wins!
                    </p>
                    <button style={{
                        padding: '15px 40px',
                        background: 'rgba(0,0,0,0.8)',
                        border: '3px solid rgba(255,255,255,0.5)',
                        borderRadius: '12px',
                        color: 'white',
                        fontSize: '18px',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                    }}
                    onClick={() => window.location.reload()}>
                        New Game
                    </button>
                </div>
            )}

            {/* Bottom Action Bar */}
            {!winner && (
                <div style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    background: 'linear-gradient(0deg, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.8) 100%)',
                    padding: '15px',
                    zIndex: 100,
                    borderTop: '2px solid rgba(255,255,255,0.2)',
                }}>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(3, 1fr)',
                        gap: '10px',
                        marginBottom: '12px',
                    }}>
                        {/* Place Button */}
                        <button
                            onClick={() => onPhaseSelect('placement')}
                            disabled={turn.hasPlaced}
                            style={{
                                minHeight: '80px',
                                background: turn.hasPlaced 
                                    ? 'linear-gradient(135deg, rgba(50,205,50,0.3) 0%, rgba(34,139,34,0.3) 100%)'
                                    : activePhase === 'placement'
                                    ? 'linear-gradient(135deg, rgba(50,205,50,0.9) 0%, rgba(34,139,34,0.9) 100%)'
                                    : 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%)',
                                border: activePhase === 'placement' ? '3px solid #50C878' : '2px solid rgba(255,255,255,0.3)',
                                borderRadius: '12px',
                                color: 'white',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                cursor: turn.hasPlaced ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                opacity: turn.hasPlaced ? 0.6 : 1,
                                fontFamily: 'system-ui, -apple-system, sans-serif',
                            }}
                        >
                            <span style={{ fontSize: '28px' }}>📍</span>
                            <span>PLACE</span>
                            {turn.hasPlaced && <span style={{ fontSize: '20px' }}>✓</span>}
                        </button>

                        {/* Infuse Button */}
                        <button
                            onClick={() => onPhaseSelect('infusion')}
                            disabled={turn.hasInfused}
                            style={{
                                minHeight: '80px',
                                background: turn.hasInfused 
                                    ? 'linear-gradient(135deg, rgba(50,205,50,0.3) 0%, rgba(34,139,34,0.3) 100%)'
                                    : activePhase === 'infusion'
                                    ? 'linear-gradient(135deg, rgba(255,191,0,0.9) 0%, rgba(255,140,0,0.9) 100%)'
                                    : 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%)',
                                border: activePhase === 'infusion' ? '3px solid #FFBF00' : '2px solid rgba(255,255,255,0.3)',
                                borderRadius: '12px',
                                color: 'white',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                cursor: turn.hasInfused ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                opacity: turn.hasInfused ? 0.6 : 1,
                                fontFamily: 'system-ui, -apple-system, sans-serif',
                            }}
                        >
                            <span style={{ fontSize: '28px' }}>⚡</span>
                            <span>INFUSE</span>
                            {turn.hasInfused && <span style={{ fontSize: '20px' }}>✓</span>}
                        </button>

                        {/* Move Button */}
                        <button
                            onClick={() => onPhaseSelect('movement')}
                            disabled={turn.hasMoved}
                            style={{
                                minHeight: '80px',
                                background: turn.hasMoved 
                                    ? 'linear-gradient(135deg, rgba(50,205,50,0.3) 0%, rgba(34,139,34,0.3) 100%)'
                                    : activePhase === 'movement'
                                    ? 'linear-gradient(135deg, rgba(0,206,209,0.9) 0%, rgba(0,139,139,0.9) 100%)'
                                    : 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%)',
                                border: activePhase === 'movement' ? '3px solid #00CED1' : '2px solid rgba(255,255,255,0.3)',
                                borderRadius: '12px',
                                color: 'white',
                                fontSize: '14px',
                                fontWeight: 'bold',
                                cursor: turn.hasMoved ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                opacity: turn.hasMoved ? 0.6 : 1,
                                fontFamily: 'system-ui, -apple-system, sans-serif',
                            }}
                        >
                            <span style={{ fontSize: '28px' }}>➡️</span>
                            <span>MOVE</span>
                            {turn.hasMoved && <span style={{ fontSize: '20px' }}>✓</span>}
                        </button>
                    </div>

                    {/* Combat & End Turn Row */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: selectedVertexId ? '1fr 1fr' : '1fr',
                        gap: '10px',
                    }}>
                        {/* Combat Button (only show when vertex selected) */}
                        {selectedVertexId && (
                            <button
                                onClick={() => setShowCombatMenu(true)}
                                disabled={!canAttack && !canPincer}
                                style={{
                                    minHeight: '60px',
                                    background: (canAttack || canPincer)
                                        ? 'linear-gradient(135deg, rgba(220,20,60,0.9) 0%, rgba(139,0,0,0.9) 100%)'
                                        : 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
                                    border: '2px solid rgba(255,255,255,0.3)',
                                    borderRadius: '12px',
                                    color: 'white',
                                    fontSize: '16px',
                                    fontWeight: 'bold',
                                    cursor: (canAttack || canPincer) ? 'pointer' : 'not-allowed',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '10px',
                                    opacity: (canAttack || canPincer) ? 1 : 0.4,
                                    fontFamily: 'system-ui, -apple-system, sans-serif',
                                }}
                            >
                                <span style={{ fontSize: '24px' }}>⚔️</span>
                                <span>COMBAT</span>
                            </button>
                        )}

                        {/* End Turn Button */}
                        <button
                            onClick={onEndTurn}
                            disabled={!allMandatoryDone}
                            style={{
                                minHeight: '60px',
                                background: allMandatoryDone 
                                    ? 'linear-gradient(135deg, rgba(255,215,0,0.9) 0%, rgba(255,165,0,0.9) 100%)'
                                    : 'linear-gradient(135deg, rgba(100,100,100,0.3) 0%, rgba(80,80,80,0.3) 100%)',
                                border: allMandatoryDone ? '3px solid gold' : '2px solid rgba(255,255,255,0.2)',
                                borderRadius: '12px',
                                color: allMandatoryDone ? '#000' : 'rgba(255,255,255,0.5)',
                                fontSize: '18px',
                                fontWeight: 'bold',
                                cursor: allMandatoryDone ? 'pointer' : 'not-allowed',
                                fontFamily: 'system-ui, -apple-system, sans-serif',
                                boxShadow: allMandatoryDone ? '0 0 20px rgba(255,215,0,0.4)' : 'none',
                            }}
                        >
                            END TURN
                        </button>
                    </div>
                </div>
            )}
        </>
    );
};

export default MobileHUD;