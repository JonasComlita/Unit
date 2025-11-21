from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import sys
import os
import logging
import stripe
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure we can import from self_play
sys.path.append(os.getcwd())

from self_play.random_algorithm import select_move as random_select
from self_play.greedy_algorithm import select_move as greedy_select
from self_play.greedy_spreader import select_move as spreader_select
from self_play.greedy_banker import select_move as banker_select
from self_play.greedy_aggressor import select_move as aggressor_select
from self_play.dynamic_agent import select_move as dynamic_select
from self_play.alpha_beta_agent import select_move as alpha_beta_select
from self_play.mcts import MCTSAgent
from database import init_database, get_or_create_user, update_premium_status, check_premium_status
from game_recorder import init_game_tables, create_game, record_move, complete_game, export_games_to_training_format
from multiplayer_server import init_multiplayer

# Initialize MCTS agents (reused across requests)
mcts_agent_20 = MCTSAgent(simulations=20, rollout_depth=3)
mcts_agent_50 = MCTSAgent(simulations=50, rollout_depth=6)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
STRIPE_PRICE_UNLIMITED = os.getenv('STRIPE_PRICE_UNLIMITED', 'price_unlimited_placeholder')
STRIPE_PRICE_MULTIPLAYER = os.getenv('STRIPE_PRICE_MULTIPLAYER', 'price_multiplayer_placeholder')

# Initialize database
init_database()
init_game_tables()
logger.info("Database initialized")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/api/ai/move', methods=['POST'])
def get_ai_move():
    try:
        data = request.json
        game_state = data.get('gameState')
        difficulty = data.get('difficulty', 1)  # Default to level 5
        
        if not game_state:
            return jsonify({"error": "Missing gameState"}), 400

        logger.info(f"Received move request. Difficulty Level: {difficulty}")

        # Convert old string difficulties to numbers for backward compatibility
        difficulty_map = {
            'very_easy': 1,
            'easy': 3,
            'medium': 5,
            'hard': 6,
            'very_hard': 7,
            'expert': 8
        }
        
        if isinstance(difficulty, str):
            difficulty = difficulty_map.get(difficulty, 5)
        
        # Ensure difficulty is an integer
        try:
            difficulty = int(difficulty)
        except (ValueError, TypeError):
            difficulty = 5
        
        # Clamp to valid range
        difficulty = max(1, min(10, difficulty))

        # Dispatch to appropriate AI agent based on level
        if difficulty == 1:
            # Level 1: Random
            move = random_select(game_state)
        elif difficulty == 2:
            # Level 2: Greedy (Basic)
            move = greedy_select(game_state)
        elif difficulty == 3:
            # Level 3: Aggressor
            move = aggressor_select(game_state)
        elif difficulty == 4:
            # Level 4: Banker
            move = banker_select(game_state)
        elif difficulty == 5:
            # Level 5: Spreader
            move = spreader_select(game_state)
        elif difficulty == 6:
            # Level 6: Dynamic
            move = dynamic_select(game_state)
        elif difficulty == 7:
            # Level 7: Alpha-Beta (Depth 2)
            move = alpha_beta_select(game_state)
        else: # difficulty == 8:
            # Level 8: Alpha-Beta (Depth 3)
            import self_play.alpha_beta_agent as ab
            old_depth = ab.MAX_DEPTH
            ab.MAX_DEPTH = 3
            move = alpha_beta_select(game_state)
            ab.MAX_DEPTH = old_depth
            
        return jsonify({"action": move})

    except Exception as e:
        logger.error(f"Error generating AI move: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/user/premium-status', methods=['GET'])
def get_premium_status():
    """Check if a user has premium status."""
    try:
        device_id = request.args.get('deviceId')
        
        if not device_id:
            return jsonify({"error": "Missing deviceId"}), 400
        
        status = check_premium_status(device_id)
        user = get_or_create_user(device_id)
        
        return jsonify({
            "isPremium": status['is_premium'],
            "premiumType": status['premium_type'],
            "deviceId": device_id,
            "userId": user['id']
        })
    
    except Exception as e:
        logger.error(f"Error checking premium status: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/payment/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """Create a Stripe Checkout session for premium upgrade."""
    try:
        data = request.json
        device_id = data.get('deviceId')
        plan_type = data.get('planType', 'unlimited')  # 'unlimited' or 'multiplayer'
        
        if not device_id:
            return jsonify({"error": "Missing deviceId"}), 400
        
        # Get or create user
        user = get_or_create_user(device_id)
        
        # Determine price and mode based on plan type
        if plan_type == 'multiplayer':
            price_id = STRIPE_PRICE_MULTIPLAYER
            mode = 'subscription'
        else:
            price_id = STRIPE_PRICE_UNLIMITED
            mode = 'payment'
        
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode=mode,
            success_url=request.host_url + '?payment=success',
            cancel_url=request.host_url + '?payment=cancelled',
            client_reference_id=device_id,  # Store device_id for webhook
            metadata={
                'device_id': device_id,
                'user_id': user['id'],
                'plan_type': plan_type
            }
        )
        
        logger.info(f"Created checkout session for device_id: {device_id}, plan: {plan_type}")
        
        return jsonify({
            "sessionId": checkout_session.id,
            "url": checkout_session.url
        })
    
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/payment/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events."""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        return jsonify({"error": "Invalid signature"}), 400
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        device_id = session.get('client_reference_id') or session['metadata'].get('device_id')
        plan_type = session['metadata'].get('plan_type', 'unlimited')
        
        if device_id:
            # Update user to premium
            stripe_customer_id = session.get('customer')
            update_premium_status(device_id, True, stripe_customer_id, premium_type=plan_type)
            logger.info(f"Payment successful for device_id: {device_id}, plan: {plan_type}")
        else:
            logger.warning("Webhook received but no device_id found")
    
    elif event['type'] == 'payment_intent.succeeded':
        logger.info("Payment intent succeeded")
    
    elif event['type'] == 'payment_intent.payment_failed':
        logger.warning("Payment failed")
    
    return jsonify({"status": "success"}), 200


@app.route('/api/game/create', methods=['POST'])
def create_game_endpoint():
    """Create a new game record to start recording moves."""
    try:
        data = request.json
        game_id = data.get('gameId')
        initial_state = data.get('initialState')
        player1_device = data.get('player1Device') or data.get('deviceId')
        player2_device = data.get('player2Device', 'AI')
        game_mode = data.get('gameMode', 'pvp')  # 'pvp' or 'pva' (vs AI)
        
        if not game_id or not initial_state or not player1_device:
            return jsonify({"error": "Missing required fields"}), 400
        
        game = create_game(game_id, initial_state, player1_device, player2_device, game_mode)
        return jsonify(game), 201
    
    except Exception as e:
        logger.error(f"Error creating game: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/game/<game_id>/move', methods=['POST'])
def record_move_endpoint(game_id):
    """Record a move in an ongoing game."""
    try:
        data = request.json
        move_number = data.get('moveNumber')
        player_id = data.get('playerId')
        action = data.get('action')
        state_before = data.get('stateBefore')  # Optional, but recommended for training
        
        if move_number is None or not player_id or not action:
            return jsonify({"error": "Missing required fields"}), 400
        
        record_move(game_id, move_number, player_id, action, state_before)
        return jsonify({"status": "recorded"}), 200
    
    except Exception as e:
        logger.error(f"Error recording move: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/game/<game_id>/complete', methods=['POST'])
def complete_game_endpoint(game_id):
    """Mark a game as complete with final result."""
    try:
        data = request.json
        winner = data.get('winner')  # 'Player1', 'Player2', or None for draw
        total_moves = data.get('totalMoves')
        
        if total_moves is None:
            return jsonify({"error": "Missing totalMoves"}), 400
        
        complete_game(game_id, winner, total_moves)
        return jsonify({"status": "completed"}), 200
    
    except Exception as e:
        logger.error(f"Error completing game: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/export-games', methods=['GET'])
def export_training_data():
    """Export human games for training (admin/internal use)."""
    try:
        limit = int(request.args.get('limit', 1000))
        compress = request.args.get('compress', 'true').lower() == 'true'
        
        games = export_games_to_training_format(limit=limit, compress_moves=compress)
        
        return jsonify({
            "count": len(games),
            "games": games
        }), 200
    
    except Exception as e:
        logger.error(f"Error exporting training data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Initialize database
init_database()
init_game_tables()

# Initialize multiplayer WebSocket handlers
init_multiplayer(socketio)

logger.info("Database and multiplayer initialized")



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
