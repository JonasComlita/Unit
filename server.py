from flask import Flask, request, jsonify
from flask_cors import CORS
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
from self_play.greedy_spreader import select_move as spreader_select
from self_play.greedy_banker import select_move as banker_select
from self_play.greedy_algorithm import select_move as greedy_select
from self_play.greedy_aggressor import select_move as aggressor_select
from database import init_database, get_or_create_user, update_premium_status, check_premium_status

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID')  # You'll need to create a product/price in Stripe dashboard

# Initialize database
init_database()
logger.info("Database initialized")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/api/ai/move', methods=['POST'])
def get_ai_move():
    try:
        data = request.json
        game_state = data.get('gameState')
        difficulty = data.get('difficulty', 'medium')
        
        if not game_state:
            return jsonify({"error": "Missing gameState"}), 400

        logger.info(f"Received move request. Difficulty: {difficulty}")

        # Dispatch to appropriate AI agent
        if difficulty == 'very_easy':
            move = random_select(game_state)
        elif difficulty == 'easy':
            move = spreader_select(game_state)
        elif difficulty == 'medium':
            # Greedy (Basic) is now Medium
            move = greedy_select(game_state)
        elif difficulty == 'hard':
            # Banker is now Hard
            move = banker_select(game_state)
        elif difficulty == 'very_hard':
            # Placeholder: Aggressor
            move = aggressor_select(game_state)
        elif difficulty == 'expert':
             # Placeholder: Aggressor (until Neural Net is ready)
            move = aggressor_select(game_state)
        else:
            # Default to Medium (Greedy)
            move = greedy_select(game_state)
            
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
        
        is_premium = check_premium_status(device_id)
        user = get_or_create_user(device_id)
        
        return jsonify({
            "isPremium": is_premium,
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
        
        if not device_id:
            return jsonify({"error": "Missing deviceId"}), 400
        
        # Get or create user
        user = get_or_create_user(device_id)
        
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': STRIPE_PRICE_ID,
                'quantity': 1,
            }],
            mode='payment',  # One-time payment. Use 'subscription' for recurring
            success_url=request.host_url + '?payment=success',
            cancel_url=request.host_url + '?payment=cancelled',
            client_reference_id=device_id,  # Store device_id for webhook
            metadata={
                'device_id': device_id,
                'user_id': user['id']
            }
        )
        
        logger.info(f"Created checkout session for device_id: {device_id}")
        
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
        
        if device_id:
            # Update user to premium
            stripe_customer_id = session.get('customer')
            update_premium_status(device_id, True, stripe_customer_id)
            logger.info(f"Payment successful for device_id: {device_id}")
        else:
            logger.warning("Webhook received but no device_id found")
    
    elif event['type'] == 'payment_intent.succeeded':
        logger.info("Payment intent succeeded")
    
    elif event['type'] == 'payment_intent.payment_failed':
        logger.warning("Payment failed")
    
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)

