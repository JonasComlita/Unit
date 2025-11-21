// src/services/storageService.ts
// Handles offline storage and sync

import { GameLog } from './gameLogger';
import { apiClient } from './apiClient';

const STORAGE_KEYS = {
    PENDING_GAMES: 'unit_game_pending_uploads',
    USER_STATS: 'unit_game_user_stats',
    SETTINGS: 'unit_game_settings'
} as const;

class StorageService {
    /**
     * Save game locally for later upload
     */
    savePendingGame(gameLog: GameLog): void {
        if (!gameLog) return;
        try {
            const pending = this.getPendingGames();
            pending.push(gameLog);
            localStorage.setItem(STORAGE_KEYS.PENDING_GAMES, JSON.stringify(pending));
            console.log(`Saved game ${gameLog.gameId} locally (${pending.length} pending)`);
        } catch (error) {
            console.error('Failed to save pending game:', error);
        }
    }

    /**
     * Get all games waiting to be uploaded
     */
    getPendingGames(): GameLog[] {
        try {
            const stored = localStorage.getItem(STORAGE_KEYS.PENDING_GAMES);
            return stored ? JSON.parse(stored).filter((g: GameLog | null) => !!g) : [];
        } catch (error) {
            console.error('Failed to get pending games:', error);
            return [];
        }
    }

    /**
     * Try to upload all pending games
     */
    async syncPendingGames(): Promise<{ uploaded: number; failed: number }> {
        const pending = this.getPendingGames();

        if (pending.length === 0) {
            return { uploaded: 0, failed: 0 };
        }

        console.log(`Syncing ${pending.length} pending games...`);

        let uploaded = 0;
        let failed = 0;
        const stillPending: GameLog[] = [];

        for (const game of pending) {
            if (!game) continue;

            const result = await apiClient.uploadGame(game);

            if (result.success) {
                uploaded++;
                console.log(`✓ Uploaded game ${game.gameId}`);
            } else {
                failed++;
                stillPending.push(game);
                console.log(`✗ Failed to upload game ${game.gameId}`);
            }

            // Small delay to avoid rate limiting
            await this.sleep(100);
        }

        // Update storage with only failed games
        localStorage.setItem(STORAGE_KEYS.PENDING_GAMES, JSON.stringify(stillPending));

        console.log(`Sync complete: ${uploaded} uploaded, ${failed} failed`);
        return { uploaded, failed };
    }

    /**
     * Clear all pending games (use carefully!)
     */
    clearPendingGames(): void {
        localStorage.removeItem(STORAGE_KEYS.PENDING_GAMES);
        console.log('Cleared all pending games');
    }

    /**
     * Save user settings
     */
    saveSettings(settings: any): void {
        try {
            localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }

    /**
     * Load user settings
     */
    loadSettings(): any {
        try {
            const stored = localStorage.getItem(STORAGE_KEYS.SETTINGS);
            return stored ? JSON.parse(stored) : this.getDefaultSettings();
        } catch (error) {
            console.error('Failed to load settings:', error);
            return this.getDefaultSettings();
        }
    }

    /**
     * Default settings
     */
    private getDefaultSettings() {
        return {
            soundEnabled: true,
            musicEnabled: false,
            vibrationEnabled: true,
            aiDifficulty: 'medium',
            theme: 'dark',
            tutorialCompleted: false
        };
    }

    /**
     * Update local user statistics
     */
    updateUserStats(winner: string | null, moves: number): void {
        try {
            const stats = this.getUserStats();

            stats.gamesPlayed++;
            stats.totalMoves += moves;

            if (winner === 'Player1') {
                stats.wins++;
            } else if (winner === 'Player2') {
                stats.losses++;
            } else {
                stats.draws++;
            }

            stats.averageMoves = Math.round(stats.totalMoves / stats.gamesPlayed);
            stats.winRate = (stats.wins / stats.gamesPlayed * 100).toFixed(1);
            stats.lastPlayed = Date.now();

            localStorage.setItem(STORAGE_KEYS.USER_STATS, JSON.stringify(stats));

        } catch (error) {
            console.error('Failed to update user stats:', error);
        }
    }

    /**
     * Get user statistics
     */
    getUserStats(): any {
        try {
            const stored = localStorage.getItem(STORAGE_KEYS.USER_STATS);
            return stored ? JSON.parse(stored) : {
                gamesPlayed: 0,
                wins: 0,
                losses: 0,
                draws: 0,
                totalMoves: 0,
                averageMoves: 0,
                winRate: '0.0',
                lastPlayed: null
            };
        } catch (error) {
            console.error('Failed to get user stats:', error);
            return null;
        }
    }

    /**
     * Clear all local data
     */
    clearAll(): void {
        Object.values(STORAGE_KEYS).forEach(key => {
            localStorage.removeItem(key);
        });
        console.log('Cleared all local storage');
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Singleton instance
export const storageService = new StorageService();