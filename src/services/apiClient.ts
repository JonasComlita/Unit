// src/services/apiClient.ts
// Centralized API communication

import { API_CONFIG } from '../game/constants';
import { GameLog } from './gameLogger';

class APIClient {
    private baseURL: string;
    private retryAttempts: number = 3;
    private retryDelay: number = 1000; // ms

    constructor() {
        this.baseURL = API_CONFIG.baseURL;
    }

    /**
     * Upload completed game to server
     */
    async uploadGame(gameLog: GameLog): Promise<{ success: boolean; gameId?: string; error?: string }> {
        try {
            const response = await this.fetchWithRetry(`${this.baseURL}${API_CONFIG.endpoints.games}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(gameLog)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return { success: true, gameId: data.gameId };

        } catch (error) {
            console.error('Failed to upload game:', error);
            return { 
                success: false, 
                error: error instanceof Error ? error.message : 'Unknown error' 
            };
        }
    }

    /**
     * Get AI's best move for current game state
     */
    async getAIMove(gameState: any, difficulty: 'easy' | 'medium' | 'hard' | 'expert' = 'medium'): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}${API_CONFIG.endpoints.aiMove}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ gameState, difficulty })
            });

            if (!response.ok) {
                throw new Error(`AI move request failed: ${response.status}`);
            }

            const data = await response.json();
            return data.action;

        } catch (error) {
            console.error('Failed to get AI move:', error);
            // Return a safe default
            return { type: 'endTurn' };
        }
    }

    /**
     * Get game statistics
     */
    async getStats(): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}${API_CONFIG.endpoints.stats}`);
            
            if (!response.ok) {
                throw new Error('Failed to fetch stats');
            }

            return await response.json();

        } catch (error) {
            console.error('Failed to get stats:', error);
            return null;
        }
    }

    /**
     * Fetch with automatic retry logic
     */
    private async fetchWithRetry(url: string, options: RequestInit, attempt: number = 1): Promise<Response> {
        try {
            const response = await fetch(url, options);
            
            // If rate limited or server error, retry
            if ((response.status === 429 || response.status >= 500) && attempt < this.retryAttempts) {
                await this.sleep(this.retryDelay * attempt);
                return this.fetchWithRetry(url, options, attempt + 1);
            }

            return response;

        } catch (error) {
            if (attempt < this.retryAttempts) {
                await this.sleep(this.retryDelay * attempt);
                return this.fetchWithRetry(url, options, attempt + 1);
            }
            throw error;
        }
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Check if API is reachable
     */
    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseURL}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            return response.ok;
        } catch (error) {
            console.warn('API health check failed:', error);
            return false;
        }
    }
}

// Singleton instance
export const apiClient = new APIClient();