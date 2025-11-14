// src/game/constants.ts

export const BOARD_CONFIG = {
    layout: [3, 5, 7, 5, 3],
    layerSpacing: 2.5,
    boardWidth: 10,
    layerGravity: [1.0, 2.0, 3.0, 2.0, 1.0]
} as const;

export const OCCUPATION_REQUIREMENTS = [
    { minPieces: 1, minEnergy: 1, minForce: 1 }, // 3x3 layers
    { minPieces: 1, minEnergy: 1, minForce: 4 }, // 5x5 layers
    { minPieces: 1, minEnergy: 1, minForce: 9 }, // 7x7 layer
] as const;

export const GAME_RULES = {
    forceCapMax: 10,
    maxPincerParticipants: 6,
    initialReinforcements: 1,
    reinforcementsPerTurn: 1
} as const;

export const API_CONFIG = {
    baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3000',
    endpoints: {
        games: '/api/games',
        moves: '/api/moves',
        stats: '/api/games/stats',
        aiMove: '/api/ai/move'
    }
} as const;

export const COLORS = {
    player1: {
        r: 0.29, g: 0.56, b: 0.89,
        hex: '#4A90E2'
    },
    player2: {
        r: 0.82, g: 0.13, b: 0.11,
        hex: '#D0021B'
    },
    actions: {
        placement: '#50C878',
        infusion: '#FFBF00',
        movement: '#00CED1',
        attack: '#DC143C',
        pincer: '#8B00FF'
    }
} as const;

export const MOBILE_CONFIG = {
    minTapTargetSize: 44,
    cameraDefaults: {
        alpha: -Math.PI / 2.5,
        beta: Math.PI / 3,
        radius: 40,
        minRadius: 20,
        maxRadius: 80
    }
} as const;

export function getOccupationRequirement(layer: number) {
    if (layer === 0 || layer === 4) return OCCUPATION_REQUIREMENTS[0];
    if (layer === 1 || layer === 3) return OCCUPATION_REQUIREMENTS[1];
    return OCCUPATION_REQUIREMENTS[2];
}