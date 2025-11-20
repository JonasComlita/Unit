// src/components/MultiplayerLobby.tsx
import React, { useState, useEffect } from 'react';
import { multiplayerService, MatchInfo } from '../services/multiplayerService';
import './MultiplayerLobby.css';

interface MultiplayerLobbyProps {
    onMatchFound: (matchInfo: MatchInfo) => void;
    onCancel: () => void;
}

const MultiplayerLobby: React.FC<MultiplayerLobbyProps> = ({ onMatchFound, onCancel }) => {
    const [status, setStatus] = useState<'idle' | 'searching' | 'found'>('idle');
    const [error, setError] = useState<string | null>(null);
    const [matchInfo, setMatchInfo] = useState<MatchInfo | null>(null);

    const handleFindMatch = async () => {
        setStatus('searching');
        setError(null);

        try {
            const info = await multiplayerService.joinMatchmaking();
            setMatchInfo(info);
            setStatus('found');
            onMatchFound(info);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to find match');
            setStatus('idle');
        }
    };

    const handleCancel = () => {
        if (status === 'searching') {
            multiplayerService.cancelMatchmaking();
        }
        setStatus('idle');
        setError(null);
        onCancel();
    };

    useEffect(() => {
        // Listen for matchmaking events
        const handleWaiting = () => {
            console.log('Waiting for opponent...');
        };

        const handleCancelled = () => {
            setStatus('idle');
        };

        multiplayerService.on('matchmaking_waiting', handleWaiting);
        multiplayerService.on('matchmaking_cancelled', handleCancelled);

        return () => {
            multiplayerService.off('matchmaking_waiting', handleWaiting);
            multiplayerService.off('matchmaking_cancelled', handleCancelled);
        };
    }, []);

    return (
        <div className="multiplayer-lobby">
            <div className="lobby-card">
                <h2>🎮 Multiplayer</h2>

                {status === 'idle' && (
                    <>
                        <p>Challenge another player to a match!</p>
                        <button
                            className="find-match-btn"
                            onClick={handleFindMatch}
                        >
                            Find Match
                        </button>
                        <button
                            className="back-btn"
                            onClick={handleCancel}
                        >
                            Back
                        </button>
                    </>
                )}

                {status === 'searching' && (
                    <div className="searching-state">
                        <div className="spinner"></div>
                        <p>Searching for opponent...</p>
                        <button
                            className="cancel-btn"
                            onClick={handleCancel}
                        >
                            Cancel
                        </button>
                    </div>
                )}

                {status === 'found' && matchInfo && (
                    <div className="match-found-state">
                        <div className="success-icon">✓</div>
                        <h3>Match Found!</h3>
                        <p>You are <strong>{matchInfo.playerId}</strong></p>
                        <p className="starting-text">Starting game...</p>
                    </div>
                )}

                {error && (
                    <div className="error-message">
                        ⚠️ {error}
                    </div>
                )}
            </div>
        </div>
    );
};

export default MultiplayerLobby;
