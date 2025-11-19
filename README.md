
# Unit Strategy Game

## Game Objective
The ultimate goal is to eliminate all of the opponent's pieces from the board. A player wins when the they occupy both of the opponent's home corners.

## Board
- **Description:** A 3D game space of 5 layers, each layer is a grid of vertices.
- **Layout:**
	- Layer 0: 3x3
	- Layer 1: 5x5
	- Layer 2: 7x7
	- Layer 3: 5x5
	- Layer 4: 3x3
- **Connectivity:** Vertices are connected to adjacent vertices on the same layer and to corresponding vertices on adjacent layers.
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
- **Condition:** This action is mandatory each turn. If the resulting force after infusion is greater than 10, the force cap is enforced and the force at that vertex equals 10.

### Move
- Move an existing stack of pieces to an adjacent, empty vertex.
- **Conditions:**
	- The target vertex must be empty and adjacent to the source.
	- The source vertex must meet the occupation requirements of the target's occupationRequirement.
	- A move transfers the entire stack and the full energy value from the source vertex to the target vertex — partial moves or splitting a stack are not permitted.
- ** Can not move and attack on the same turn.

### Attack
- Use an occupied vertex to attack an adjacent enemy vertex. Combat is always attritional.
- **Condition:** The attacking vertex must be 'occupied'.
- **Combat Resolution:**
	- **Attacker Wins:**
		- Condition: Attacker's Force > Defender's Force
		- Outcome:
			- The defender's stack is destroyed.
			- The attacker's stack moves to the defender's vertex.
			- The new stack size for the attacker is `abs(attacker_pieces - defender_pieces)`.
			- The new vertex energy for the attacker is `abs(attacker_energy - defender_energy)`.
	- **Defender Wins:**
		- Condition: Attacker's Force < Defender's Force
		- Outcome:
			- The attacker's stack is destroyed.
			- The defender's stack remains on their vertex.
			- The new stack size for the defender is `abs(defender_pieces - attacker_pieces)`.
			- The new vertex energy for the defender is `abs(defender_energy - attacker_energy)`.
	- **Draw:**
		- Condition: Attacker's Force = Defender's Force
		- Outcome:
			- The new stack size for the greater of the two is `abs(more_pieces - less_pieces)`.
			- The new vertex energy for the greater of the two is `abs(more_energy - less_energy)`.
			- Both player's remain at their original positions with their new forces.

- **Turn Ends:** An attack ends the player's turn.
- **Note:** You cannot both move and attack in the same turn.

### Pincer Attack
- Use multiple vertices to attack a shared vertex that is adjacent to each attacking vertex. (A max of 6 vertices can share one adjacent vertex.) 
- **Condition:** The attacking vertices must be occupied by the attacker.
- **Mechanics:**
	- Compute attacker and defender strengths same as a normal attack: `force = (pieces * energy) / gravity of layer.`
	- The pincer attack multiplies all of the attacking vertices forces together, creating an exponential outcome for the attacker.
	- The new stack size for the victor is the same as when an attacker or defender wins.

- **Turn Ends:** A pincer attack immediately ends the player's turn.
- **Note:** As with regular attacks, you cannot both move and attack in the same turn.

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


### Basic Macro Strategies

### 1. The Economy of Force
- **Efficiency**: Since `Force = (Pieces * Energy) / Gravity`, and Gravity increases towards the center (Layers 1, 2, 3), your units are most efficient on the top and bottom layers (Layers 0 and 4). Fighting in the center requires significantly more investment.
- **Occupation Thresholds**: Memorize the `P * E` (Pieces times Energy) targets to make a vertex "occupied" (able to attack):
	- **3x3 Layers**: `P * E >= 1` (Easy)
	- **5x5 Layers**: `P * E >= 8` (Moderate)
	- **7x7 Layer**: `P * E >= 27` (Hard)

### 2. Combat Arithmetic
- **Overwhelming Victory**: Combat is attritional. If you attack a stack of 5 pieces with 6 pieces, you win but are left with only 1 piece. Always aim to attack with significantly more pieces than your opponent to preserve your army size.
- **Energy Conservation**: Similar to pieces, you keep the difference in energy. High-energy units are valuable; protect them from attrition by low-energy enemy spam.

### 3. Positioning and Movement
- **The Pincer Maneuver**: Pincer attacks multiply the force of all attacking stacks. This is the only way to defeat a fully fortified "Death Star" stack (Force 10) without sacrificing your own massive stack. Position multiple small stacks to surround and eliminate large threats.
- **Stacking**: You can move onto your own pieces to merge stacks. Use this to ferry reinforcements from your spawn point to the front lines.
- **Home Defense**: You lose if the opponent occupies your home corners. Never leave them completely vulnerable while pushing for an offensive.

### 4. Opening Moves
- **Corner Rush**: Quickly expanding on your home layer (3x3) is cheap. Securing your layer gives you a platform to launch attacks into the expensive center layers.
- **The Deep Strike**: Building a single massive stack to punch through the high-gravity center can catch an opponent off guard, but it's a high-risk "all-in" strategy.

### 5. Advanced Tactics
- **Kamikaze / Suicide Bomber**: Sometimes it is worth making a losing attack to whittle down an opponent's vertex. Even if you lose the battle, the attrition mechanic means the defender loses pieces equal to your army size. Use this to soften up a strong enemy before a final killing blow.
- **Banking / Storing**: The center layers (Layers 1, 2, 3) have higher gravity, which means they can hold more pieces and energy before hitting the Force Cap of 10. Use these high-gravity zones as "vaults" to store massive armies that would otherwise be capped on the outer layers.
- **The Spread**: The opposite of banking. Spreading your pieces and energy across as many vertices as possible increases your board control and sets up opportunities for **Pincer Attacks**. A wide net allows you to catch enemy units in crossfires.