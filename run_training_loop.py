#!/usr/bin/env python3
"""
Iterative Training Loop for Unit Game AI.

This script automates the cycle of:
1. Generating self-play data (using current best model)
2. Training a new model on that data
3. Evaluating the new model against the previous best
4. Promoting the new model if it wins significantly
"""

import os
import subprocess
import logging
import shutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_loop.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainingLoop")

# Configuration
ITERATIONS = 5
GAMES_PER_ITERATION = 1024
CONCURRENT_GAMES = 16
EPOCHS = 50
BATCH_SIZE = 256
DATA_DIR_BASE = "shards/iter_{}"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
CANDIDATE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "candidate_model.pt")

def run_command(cmd):
    """Run a shell command and check for errors."""
    logger.info(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        raise

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Ensure we have a base model (optional, can start from scratch/random)
    if not os.path.exists(BEST_MODEL_PATH):
        logger.info("No existing best model found. Starting from scratch (random/heuristic).")
    
    for i in range(1, ITERATIONS + 1):
        logger.info(f"=== Starting Iteration {i}/{ITERATIONS} ===")
        
        data_dir = DATA_DIR_BASE.format(i)
        os.makedirs(data_dir, exist_ok=True)
        
        # 1. Generate Data
        logger.info(f"Generating {GAMES_PER_ITERATION} games into {data_dir}...")
        
        gen_cmd = (
            f"./.venv311/bin/python -m self_play.main "
            f"--use-gpu-inference-server "
            f"--num-workers 12 "
            f"--shard-dir {data_dir} "
            f"--shard-move-mode compressed "
            f"--file-writer "
            f"--batch-only "
            f"--trim-states "
            f"--random-start "
            f"--use-model "
            f"--model-path {BEST_MODEL_PATH} "
            f"--model-device cuda "
            f"--game-version v1-nn"
        )
        
        if os.path.exists(BEST_MODEL_PATH):
             logger.info(f"Using best model {BEST_MODEL_PATH} for generation.")
        else:
             logger.info("No best model found; using random/heuristic generation.")
        
        run_command(gen_cmd)
        
        # 2. Train Model
        logger.info(f"Training model on {data_dir}...")
        train_cmd = (
            f"./.venv311/bin/python -m self_play.training_pipeline train "
            f"--data-dir {data_dir} "
            f"--epochs {EPOCHS} "
            f"--batch-size {BATCH_SIZE}"
        )
        run_command(train_cmd)
        
        # Find the latest checkpoint from this training run
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt") and "epoch" in f]
        if not checkpoints:
            logger.error("No checkpoints found after training!")
            continue
            
        # Sort by modification time to get the most recent one
        latest_ckpt = max([os.path.join(CHECKPOINT_DIR, c) for c in checkpoints], key=os.path.getmtime)
        logger.info(f"Latest checkpoint: {latest_ckpt}")
        shutil.copy(latest_ckpt, CANDIDATE_MODEL_PATH)
        
        # 3. Evaluate
        logger.info("Evaluating candidate model...")
        eval_cmd = (
            f"./.venv311/bin/python -m self_play.training_pipeline evaluate "
            f"--model {CANDIDATE_MODEL_PATH}"
        )
        run_command(eval_cmd)
        
        # 4. Promote
        # TODO: Parse evaluation output to conditionally promote. 
        # For now, we assume iterative improvement and always promote to keep the loop moving.
        logger.info(f"Promoting {CANDIDATE_MODEL_PATH} to {BEST_MODEL_PATH}")
        shutil.copy(CANDIDATE_MODEL_PATH, BEST_MODEL_PATH)
        
        logger.info(f"Iteration {i} complete.")

if __name__ == "__main__":
    main()
