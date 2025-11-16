import React, { useState } from 'react';
import { GameState } from '../game/types';

const DebugPanel: React.FC<{ gameState: GameState }> = ({ gameState }) => {
  const [open, setOpen] = useState(true);
  const { currentPlayerId, turn, selectedVertexId, validPlacementVertices, validInfusionVertices, validMoveOrigins, validMoveTargets, validAttackTargets, validPincerTargets } = gameState;

  return (
    <div style={{ position: 'absolute', top: 80, right: 12, zIndex: 200 }}>
      <div id="debug-panel-inner" style={{ width: open ? 360 : 48, background: 'rgba(0,0,0,0.7)', color: 'white', borderRadius: 8, padding: open ? 12 : 6, boxShadow: '0 6px 18px rgba(0,0,0,0.6)', fontFamily: 'monospace', fontSize: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <strong style={{ fontSize: 13 }}>{open ? 'Debug Panel' : 'DBG'}</strong>
          <button onClick={() => setOpen(!open)} style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}>{open ? '▢' : '▤'}</button>
        </div>
        {open && (
          <div style={{ maxHeight: 380, overflow: 'auto' }}>
            <div><strong>Player:</strong> {currentPlayerId}</div>
            <div><strong>Turn:</strong> {JSON.stringify(turn)}</div>
            <div><strong>Selected:</strong> {selectedVertexId}</div>
            <hr style={{ borderColor: 'rgba(255,255,255,0.08)' }} />
            <div><strong>Placement:</strong> [{validPlacementVertices.join(', ')}]</div>
            <div><strong>Infusion:</strong> [{validInfusionVertices.join(', ')}]</div>
            <div><strong>Move Origins:</strong> [{validMoveOrigins.join(', ')}]</div>
            <div><strong>Move Targets:</strong> [{validMoveTargets.join(', ')}]</div>
            <div><strong>Attack Targets:</strong> [{validAttackTargets.join(', ')}]</div>
            <div style={{ whiteSpace: 'pre-wrap' }}><strong>Pincer:</strong> {JSON.stringify(validPincerTargets, null, 2)}</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DebugPanel;
