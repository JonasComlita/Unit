import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'game.db')


def init_database():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT UNIQUE NOT NULL,
            stripe_customer_id TEXT,
            is_premium BOOLEAN DEFAULT 0,
            premium_type TEXT,
            premium_until TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if premium_type column exists (migration for existing db)
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'premium_type' not in columns:
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN premium_type TEXT')
            logger.info("Added premium_type column to users table")
        except Exception as e:
            logger.error(f"Error adding premium_type column: {e}")
    
    # Create index on device_id for faster lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_device_id ON users(device_id)
    ''')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def get_connection():
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def get_or_create_user(device_id: str) -> Dict[str, Any]:
    """
    Get user by device_id, or create if doesn't exist.
    
    Args:
        device_id: Unique device identifier from frontend
        
    Returns:
        User data as dictionary
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Try to get existing user
    cursor.execute(
        'SELECT id, device_id, stripe_customer_id, is_premium, premium_until, premium_type FROM users WHERE device_id = ?',
        (device_id,)
    )
    row = cursor.fetchone()
    
    if row:
        user = {
            'id': row[0],
            'device_id': row[1],
            'stripe_customer_id': row[2],
            'is_premium': bool(row[3]),
            'premium_until': row[4],
            'premium_type': row[5]
        }
    else:
        # Create new user
        cursor.execute(
            'INSERT INTO users (device_id) VALUES (?)',
            (device_id,)
        )
        conn.commit()
        user_id = cursor.lastrowid
        user = {
            'id': user_id,
            'device_id': device_id,
            'stripe_customer_id': None,
            'is_premium': False,
            'premium_until': None,
            'premium_type': None
        }
        logger.info(f"Created new user with device_id: {device_id}")
    
    conn.close()
    return user


def update_premium_status(device_id: str, is_premium: bool, stripe_customer_id: Optional[str] = None, premium_until: Optional[str] = None, premium_type: Optional[str] = None):
    """
    Update user's premium status.
    
    Args:
        device_id: Unique device identifier
        is_premium: Premium status
        stripe_customer_id: Stripe customer ID (optional)
        premium_until: Expiration timestamp (optional, for subscriptions)
        premium_type: Type of premium ('unlimited' or 'multiplayer')
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    update_fields = ['is_premium = ?', 'updated_at = ?']
    params = [is_premium, datetime.now().isoformat()]
    
    if stripe_customer_id:
        update_fields.append('stripe_customer_id = ?')
        params.append(stripe_customer_id)
    
    if premium_until:
        update_fields.append('premium_until = ?')
        params.append(premium_until)
        
    if premium_type:
        update_fields.append('premium_type = ?')
        params.append(premium_type)
    
    params.append(device_id)
    
    query = f"UPDATE users SET {', '.join(update_fields)} WHERE device_id = ?"
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    logger.info(f"Updated premium status for device_id {device_id}: is_premium={is_premium}, type={premium_type}")


def check_premium_status(device_id: str) -> Dict[str, Any]:
    """
    Check if a user has premium status.
    
    Args:
        device_id: Unique device identifier
        
    Returns:
        Dict with is_premium and premium_type
    """
    user = get_or_create_user(device_id)
    
    # Check if premium_until is set and hasn't expired
    if user['premium_until']:
        try:
            premium_until = datetime.fromisoformat(user['premium_until'])
            if datetime.now() > premium_until:
                # Premium expired, update status
                update_premium_status(device_id, False)
                return {'is_premium': False, 'premium_type': None}
        except (ValueError, TypeError):
            pass
    
    return {
        'is_premium': user['is_premium'],
        'premium_type': user['premium_type']
    }


def get_user_by_stripe_customer(stripe_customer_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user by Stripe customer ID.
    
    Args:
        stripe_customer_id: Stripe customer ID
        
    Returns:
        User data or None
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, device_id, stripe_customer_id, is_premium, premium_until FROM users WHERE stripe_customer_id = ?',
        (stripe_customer_id,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'device_id': row[1],
            'stripe_customer_id': row[2],
            'is_premium': bool(row[3]),
            'premium_until': row[4]
        }
    return None
