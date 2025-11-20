// src/services/gameRecordingService.ts
/**
 * Game Recording Service
 * 
 * Records human gameplay for ML training data collection.
 * Integrates with backend API to store games and moves.
 */

import { GameState, PlayerAction } from '../game/types';

const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:3000';

export interface GameRecording {
    gameId: string;
    moveNumber: number;
    isRecording: boolean;
}

class GameRecordingService {
    private currentRecording: GameRecording | null = null;
    private deviceId: string;

    constructor() {
        // Get or create device ID
        this.deviceId = this.getOrCreateDeviceId();
    }

    private getOrCreateDeviceId(): string {
        let deviceId = localStorage.getItem('deviceId');
        if (!deviceId) {
            deviceId = crypto.randomUUID();
            localStorage.setItem('deviceId', deviceId);
        }
        return deviceId;
    }

    /**
     * Start recording a new game
     */
    async startRecording(
        initialState: GameState,
        gameMode: 'pvp' | 'pva' = 'pva',
        player2Device?: string
    ): Promise<string> {
        const gameId = crypto.randomUUID();

        try {
            const response = await fetch(`${API_URL}/api/game/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gameId,
                    initialState,
                    player1Device: this.deviceId,
                    player2Device: player2Device || 'AI',
                    gameMode
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to create game: ${response.statusText}`);
            }

            this.currentRecording = {
                gameId,
                moveNumber: 0,
                isRecording: true
            };

            console.log(`🎮 Started recording game: ${gameId} (${gameMode})`);
            return gameId;
        } catch (error) {
            console.error('Failed to start game recording:', error);
            // Don't fail the game if recording fails
            this.currentRecording = null;
            throw error;
        }
    }

    async recordMove(
        playerId: string,
        action: PlayerAction,
        stateBefore: GameState
    ): Promise<void> {
        if (!this.currentRecording || !this.currentRecording.isRecording) {
            return; // Silently skip if not recording
        }

        try {
            const response = await fetch(
                `${API_URL}/api/game/${this.currentRecording.gameId}/move`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        moveNumber: this.currentRecording.moveNumber,
                        playerId,
                        action,
                        stateBefore // Important for training quality!
                    })
                }
            );

            if (!response.ok) {
                console.warn(`Failed to record move ${this.currentRecording.moveNumber}`);
            }

            this.currentRecording.moveNumber++;
        } catch (error) {
            console.error('Failed to record move:', error);
            // Don't fail the game if recording fails
        }
    }

    /**
     * Complete the current game recording
     */
    async completeRecording(winner: string | null, totalMoves: number): Promise<void> {
        if (!this.currentRecording) {
            return;
        }

        try {
            const response = await fetch(
                `${API_URL}/api/game/${this.currentRecording.gameId}/complete`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        winner,
                        totalMoves
                    })
                }
            );

            if (!response.ok) {
                console.warn('Failed to complete game recording');
            }

            console.log(`✅ Game recording completed: ${this.currentRecording.gameId}, Winner: ${winner || 'Draw'}`);
        } catch (error) {
            console.error('Failed to complete game recording:', error);
        } finally {
            this.currentRecording = null;
        }
    }

    /**
     * Cancel current recording
     */
    cancelRecording(): void {
        if (this.currentRecording) {
            console.log(`❌ Cancelled recording: ${this.currentRecording.gameId}`);
            this.currentRecording = null;
        }
    }

    /**
     * Check if currently recording
     */
    isRecording(): boolean {
        return this.currentRecording !== null && this.currentRecording.isRecording;
    }

    /**
     * Get current recording info
     */
    getCurrentRecording(): GameRecording | null {
        return this.currentRecording;
    }
}

// Export singleton instance
export const gameRecorder = new GameRecordingService();
