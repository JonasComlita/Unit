// src/game/types.ts
import { Vector3 } from '@babylonjs/core';

export type PlayerId = 'Player1' | 'Player2';

export interface Piece {
    id: string;
    player: PlayerId;
}

export interface Vertex {
    id: string;
    position: Vector3;
    layer: number;
    layerSize: number;
    adjacencies: string[];
    stack: Piece[];
    vertexSpacing: number;
    energy: number;
}

export interface PlayerState {
    id: PlayerId;
    reinforcements: number;
}

export interface TurnState {
    hasPlaced: boolean;
    hasInfused: boolean;
    hasMoved: boolean;
    turnNumber: number;
}

export interface GameState {
    vertices: Record<string, Vertex>;
    players: Record<PlayerId, PlayerState>;
    currentPlayerId: PlayerId;
    turn: TurnState;
    homeCorners: Record<PlayerId, string[]>;
    winner: PlayerId | null;
    
    // UI State
    selectedVertexId: string | null;
    validPlacementVertices: string[];
    validInfusionVertices: string[];
    validAttackTargets: string[];
    validPincerTargets: Record<string, string[]>;
    validMoveOrigins: string[];
    validMoveTargets: string[];
}

export type PlayerAction = 
    | { type: 'select'; vertexId: string | null }
    | { type: 'place'; vertexId: string }
    | { type: 'infuse'; vertexId: string }
    | { type: 'move'; fromId: string; toId: string }
    | { type: 'attack'; vertexId: string; targetId: string }
    | { type: 'pincer'; targetId: string; originIds: string[] }
    | { type: 'endTurn' };

export type ActionPhase = 'placement' | 'infusion' | 'movement' | 'combat' | null;