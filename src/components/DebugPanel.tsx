import React, { useState } from 'react';
import { GameState } from '../game/types';

const DebugPanel: React.FC<{ gameState: GameState }> = ({ gameState }) => {
  const [open, setOpen] = useState(true);
  const { currentPlayerId, turn, selectedVertexId, validPlacementVertices, validInfusionVertices, validMoveOrigins, validMoveTargets, validAttackTargets, validPincerTargets } = gameState;

  return (
    <div style={{ position: 'absolute', top: 70, right: 12, zIndex: 200 }}>
      <div id="debug-panel-inner" style={{
        width: open ? 360 : 50,
        background: open ? '#222' : '#333',
        color: '#ddd',
        borderRadius: '2px',
        padding: open ? '15px' : '0',
        border: '1px solid #555',
        fontFamily: '"Times New Roman", Times, serif',
        fontSize: '14px',
        height: open ? 'auto' : '40px',
        display: open ? 'block' : 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: open ? 'default' : 'pointer',
      }}
        onClick={(e) => !open && setOpen(true)}
      >
        {!open ? (
          <div style={{ fontWeight: 'bold', fontSize: '12px' }}>DBG</div>
        ) : (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15, borderBottom: '1px solid #444', paddingBottom: 10 }}>
              <strong style={{ fontSize: '16px', textTransform: 'uppercase', letterSpacing: '1px' }}>Debug Panel</strong>
              <button
                onClick={(e) => { e.stopPropagation(); setOpen(false); }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#aaa',
                  cursor: 'pointer',
                  fontSize: '16px',
                  padding: 0,
                }}
              >
                ✕
              </button>
            </div>
            <div style={{ maxHeight: 380, overflow: 'auto', fontFamily: 'monospace', fontSize: '12px', color: '#ccc' }}>
              <div style={{ marginBottom: 4 }}><strong>Player:</strong> {currentPlayerId}</div>
              <div style={{ marginBottom: 4 }}><strong>Turn:</strong> {JSON.stringify(turn)}</div>
              <div style={{ marginBottom: 4 }}><strong>Selected:</strong> {selectedVertexId}</div>
              <hr style={{ borderColor: '#444', margin: '10px 0' }} />
              <div style={{ marginBottom: 4 }}><strong>Placement:</strong> [{validPlacementVertices.join(', ')}]</div>
              <div style={{ marginBottom: 4 }}><strong>Infusion:</strong> [{validInfusionVertices.join(', ')}]</div>
              <div style={{ marginBottom: 4 }}><strong>Move Origins:</strong> [{validMoveOrigins.join(', ')}]</div>
              <div style={{ marginBottom: 4 }}><strong>Move Targets:</strong> [{validMoveTargets.join(', ')}]</div>
              <div style={{ marginBottom: 4 }}><strong>Attack Targets:</strong> [{validAttackTargets.join(', ')}]</div>
              <div style={{ whiteSpace: 'pre-wrap' }}><strong>Pincer:</strong> {JSON.stringify(validPincerTargets, null, 2)}</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default DebugPanel;
