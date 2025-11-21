// src/services/gameLogger.ts
import { PlayerAction } from '../game/types';
import { apiClient } from './apiClient';
import { storageService } from './storageService';

export interface GameLog {
    gameId: string;
    startTime: number;
    endTime?: number;
    platform: 'web' | 'ios' | 'android';
    winner: string | null;
    moves: MoveLog[];
    totalMoves: number;
}

export interface MoveLog {
    moveNumber: number;
    playerId: string;
    action: PlayerAction;
    thinkingTime: number;
    timestamp: number;
}

class GameDataLogger {
    private currentGame: GameLog | null = null;
    private moveStartTime: number = 0;
    private isOnline: boolean = true;

    constructor() {
        this.isOnline = navigator.onLine;

        window.addEventListener('online', () => {
            console.log('📶 Connection restored');
            this.isOnline = true;
            this.syncPendingGames();
        });

        window.addEventListener('offline', () => {
            console.log('📵 Connection lost');
            this.isOnline = false;
        });

        this.syncPendingGames();
    }

    startGame(platform: 'web' | 'ios' | 'android' = 'web'): void {
        const gameId = this.generateGameId();

        this.currentGame = {
            gameId,
            startTime: Date.now(),
            platform,
            winner: null,
            moves: [],
            totalMoves: 0
        };

        this.moveStartTime = Date.now();
        console.log(`🎮 Started logging game: ${gameId}`);
    }

    logMove(action: PlayerAction, currentPlayer: string): void {
        if (!this.currentGame) {
            console.warn('No game in progress, skipping move log');
            return;
        }

        const thinkingTime = Date.now() - this.moveStartTime;

        const moveLog: MoveLog = {
            moveNumber: this.currentGame.moves.length + 1,
            playerId: currentPlayer,
            action,
            thinkingTime,
            timestamp: Date.now()
        };

        this.currentGame.moves.push(moveLog);
        this.currentGame.totalMoves++;
        this.moveStartTime = Date.now();
    }

    async endGame(winner: string | null): Promise<void> {
        const game = this.currentGame;
        if (!game) {
            console.warn('No game in progress to end');
            return;
        }

        // Prevent double ending
        if (game.endTime) {
            return;
        }

        game.endTime = Date.now();
        game.winner = winner;

        const duration = game.endTime - game.startTime;
        console.log(`🏁 Game ${game.gameId} ended`);
        console.log(`   Winner: ${winner || 'Draw'}`);
        console.log(`   Moves: ${game.totalMoves}`);
        console.log(`   Duration: ${Math.round(duration / 1000)}s`);

        storageService.updateUserStats(winner, game.totalMoves);

        if (this.isOnline) {
            const result = await apiClient.uploadGame(game);

            if (result.success) {
                console.log('✓ Game uploaded successfully');
            } else {
                console.log('✗ Upload failed, saving locally');
                storageService.savePendingGame(game);
            }
        } else {
            console.log('📵 Offline, saving locally');
            storageService.savePendingGame(game);
        }

        // Only clear if it's still the same game
        if (this.currentGame === game) {
            this.currentGame = null;
        }
    }

    async syncPendingGames(): Promise<void> {
        if (!this.isOnline) {
            console.log('Cannot sync: offline');
            return;
        }

        const result = await storageService.syncPendingGames();

        if (result.uploaded > 0 || result.failed > 0) {
            console.log(`Sync result: ${result.uploaded} uploaded, ${result.failed} failed`);
        }
    }

    getCurrentGame(): GameLog | null {
        return this.currentGame;
    }

    private generateGameId(): string {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substring(2, 11);
        return `game_${timestamp}_${random}`;
    }

    getStats(): any {
        return {
            currentGame: this.currentGame ? {
                gameId: this.currentGame.gameId,
                moves: this.currentGame.totalMoves,
                duration: Date.now() - this.currentGame.startTime
            } : null,
            pendingGames: storageService.getPendingGames().length,
            isOnline: this.isOnline,
            userStats: storageService.getUserStats()
        };
    }
}

export const gameLogger = new GameDataLogger();