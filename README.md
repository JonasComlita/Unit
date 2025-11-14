{
  "gameObjective": {
    "description": "The ultimate goal is to eliminate all of the opponent's pieces from the board. A player wins when the opponent has no pieces left in play."
  },
  "board": {
    "description": "A 3D game space of 5 layers, with each layer being a grid of vertices.",
    "layout": [
      { "layer": 0, "size": "3x3" },
      { "layer": 1, "size": "5x5" },
      { "layer": 2, "size": "7x7" },
      { "layer": 3, "size": "5x5" },
      { "layer": 4, "size": "3x3" }
    ],
    "connectivity": "Vertices are connected to adjacent vertices on the same layer (with wrap-around) and to corresponding vertices on adjacent layers."
  },
  "coreConcepts": {
    "occupationRequirement": {
      "description": "A vertex can only be moved to if the source vertex the player is moving from meets the occupation requirements.",
      "requirementsByLayer": [
        { "layerType": "3x3", "minPieces": 1, "minEnergy": 1, "minForce": 1 },
        { "layerType": "5x5", "minPieces": 1, "minEnergy": 1, "minForce": 4 },
        { "layerType": "7x7", "minPieces": 1, "minEnergy": 1, "minForce": 9 }
      ]
    },
    "force": {
      "description": "The strength of an occupied stack, used for combat.",
      "calculation": "Force = (Number of Pieces in Stack * Vertex Energy) / Layer Gravity",
      "cap": 10,
      "condition": "A vertex only exerts force if it is 'occupied'."
    },
    "gravity": {
      "description": "A debuff applied to force calculation. Symmetrical across the top and bottom layers.",
      "valuesByLayer": [1.0, 2.0, 3.0, 2.0, 1.0]
    }
  },
  "playerState": {
    "reinforcements": {
      "description": "A pool of pieces available to each player for placement.",
      "accrual": "Players receive 1 reinforcement piece at the start of their turn."
    }
  },
  "turnStructure": {
    "description": "Each turn, a player MUST perform both a Place action and an Infuse action. The order is not important.",
    "startOfTurn": [
      "Receive 1 reinforcement piece.",
      "Receive 1 energy point to distribute."
    ],
    "mandatoryActions": [ "Place", "Infuse", "Move" ],
    "optionalActions": [ "Attack", "End Turn" ]
  },
  "actions": {
    "place": {
      "description": "Deploy one piece from the reinforcement pool to one of the player's two designated home corners.",
      "cost": "1 reinforcement piece.",
      "condition": "The target vertex must be one of the player's two home corners. This action is mandatory each turn."
    },
    "infuse": {
      "description": "Increase the energy of any vertex containing at least one of the player's pieces.",
      "cost": "1 energy point per use.",
      "effect": "Adds 1 energy to the target vertex.",
      "condition": "Cannot be used if the resulting force would exceed the cap of 10. This action is mandatory each turn."
    },
    "move": {
      "description": "Move an existing stack of pieces to an adjacent, empty vertex.",
      "condition": ["The target vertex must be empty and adjacent to the source.",
                    "The source vertex must meet the occupation requirements of the target's occupationRequirement.",
                    "A move transfers the entire stack and the full energy value from the source vertex to the target vertex — partial moves or splitting a stack are not permitted."]
    },
    "attack": {
      "description": "Use an occupied vertex to attack an adjacent enemy vertex. Combat is always attritional.",
      "condition": "The attacking vertex must be 'occupied'.",
      "combatResolution": {
        "attackerWins": {
          "condition": "Attacker's Force > Defender's Force",
          "outcome": [
            "The defender's stack is destroyed.",
            "The attacker's stack moves to the defender's vertex.",
            "The new stack size for the victor is abs(attacker_pieces - defender_pieces).",
            "The new vertex energy for the victor is abs(attacker_energy - defender_energy)."
          ]
        },
        "defenderWinsOrDraw": {
          "condition": "Attacker's Force <= Defender's Force",
          "outcome": [
            "The attacker's stack is destroyed.",
            "The defender's stack remains on their vertex.",
            "The new stack size for the victor is abs(defender_pieces - attacker_pieces).",
            "The new vertex energy for the victor is abs(defender_energy - attacker_energy)."
          ]
        }
      },
      "turnEnds": "An attack immediately ends the player's turn."
    }
  }
}