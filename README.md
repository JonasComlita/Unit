
# Unit Strategy Game

## Game Objective
The ultimate goal is to eliminate all of the opponent's pieces from the board. A player wins when the opponent has no pieces left in play.

## Board
- **Description:** A 3D game space of 5 layers, each layer is a grid of vertices.
- **Layout:**
	- Layer 0: 3x3
	- Layer 1: 5x5
	- Layer 2: 7x7
	- Layer 3: 5x5
	- Layer 4: 3x3
- **Connectivity:** Vertices are connected to adjacent vertices on the same layer (with wrap-around) and to corresponding vertices on adjacent layers.

## Core Concepts

### Occupation Requirement
- A vertex can only be moved to if the source vertex meets the occupation requirements.
- **Requirements by Layer:**
	- 3x3: minPieces=1, minEnergy=1, minForce=1
	- 5x5: minPieces=1, minEnergy=1, minForce=4
	- 7x7: minPieces=1, minEnergy=1, minForce=9

### Force
- The strength of an occupied stack, used for combat.
- **Calculation:**
	- `Force = (Number of Pieces in Stack * Vertex Energy) / Layer Gravity`
- **Cap:** 10
- **Condition:** A vertex only exerts force if it is 'occupied'.

### Gravity
- A debuff applied to force calculation. Symmetrical across the top and bottom layers.
- **Values by Layer:** `[1.0, 2.0, 3.0, 2.0, 1.0]`

## Player State

### Reinforcements
- A pool of pieces available to each player for placement.
- **Accrual:** Players receive 1 reinforcement piece at the start of their turn.

## Turn Structure
- Each turn, a player MUST perform both a Place action and an Infuse action. The order is not important.
- **Start of Turn:**
	- Receive 1 reinforcement piece.
	- Receive 1 energy point to distribute.
- **Mandatory Actions:** Place, Infuse, Move
- **Optional Actions:** Attack, End Turn

## Actions

### Place
- Deploy one piece from the reinforcement pool to one of the player's two designated home corners.
- **Cost:** 1 reinforcement piece.
- **Condition:** The target vertex must be one of the player's two home corners. This action is mandatory each turn.

### Infuse
- Increase the energy of any vertex containing at least one of the player's pieces.
- **Cost:** 1 energy point per use.
- **Effect:** Adds 1 energy to the target vertex.
- **Condition:** Cannot be used if the resulting force would exceed the cap of 10. This action is mandatory each turn.

### Move
- Move an existing stack of pieces to an adjacent, empty vertex.
- **Conditions:**
	- The target vertex must be empty and adjacent to the source.
	- The source vertex must meet the occupation requirements of the target's occupationRequirement.
	- A move transfers the entire stack and the full energy value from the source vertex to the target vertex — partial moves or splitting a stack are not permitted.

### Attack
- Use an occupied vertex to attack an adjacent enemy vertex. Combat is always attritional.
- **Condition:** The attacking vertex must be 'occupied'.
- **Combat Resolution:**
	- **Attacker Wins:**
		- Condition: Attacker's Force > Defender's Force
		- Outcome:
			- The defender's stack is destroyed.
			- The attacker's stack moves to the defender's vertex.
			- The new stack size for the victor is `abs(attacker_pieces - defender_pieces)`.
			- The new vertex energy for the victor is `abs(attacker_energy - defender_energy)`.
	- **Defender Wins or Draw:**
		- Condition: Attacker's Force <= Defender's Force
		- Outcome:
			- The attacker's stack is destroyed.
			- The defender's stack remains on their vertex.
			- The new stack size for the victor is `abs(defender_pieces - attacker_pieces)`.
			- The new vertex energy for the victor is `abs(defender_energy - attacker_energy)`.
- **Turn Ends:** An attack immediately ends the player's turn.

---

# Running Self-Play and Training

## 1. Generate Self-Play Data

Run the self-play generator to create training data shards:

```bash
python -m self_play.main --file-writer --shard-format parquet --shard-move-mode compressed --trim-states --random-start --shard-dir shards/v1_model_data --use-model --model-path checkpoints/best_model.pt --model-device cuda --game-version v1-nn
```

- **Shards Output Location:**
	- Shard files will be written to the directory specified by `--shard-dir` (default: `shards/v1_model_data`).
	- Each shard contains a batch of self-play games in Parquet format.

## 2. Train the Model on Shards

Use the training pipeline to train a neural network on the generated shards:

```bash
python training_pipeline.py train --data-dir shards/v1_model_data --epochs 100 --batch-size 256
```

- **Training Data:**
	- The pipeline will automatically load all shard files in the specified directory.

- **Model Output Location:**
	- Model checkpoints are saved in the `checkpoints/` directory.
	- The best-performing model is saved as `checkpoints/best_model.pt`.

## 3. Full Pipeline Example

To run the full pipeline (generate data, train, and evaluate):

```bash
python training_pipeline.py full --games 10000 --epochs 50
```

---