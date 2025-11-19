import logging
from self_play.agent_utils import initialize_game
from self_play.mcts import MCTSAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_mcts():
    print("Initializing game...")
    state = initialize_game()
    
    print("Creating MCTS Agent (sims=50, depth=3)...")
    agent = MCTSAgent(simulations=50, rollout_depth=3)
    
    print("Selecting move...")
    move = agent.select_move(state)
    
    print(f"Selected move: {move}")
    print("Test passed!")

if __name__ == "__main__":
    test_mcts()
