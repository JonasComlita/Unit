// src/services/paymentService.ts
import { v4 as uuidv4 } from 'uuid';

const API_BASE_URL = 'http://localhost:3000';

/**
 * Get or create a unique device ID for this browser/device
 */
export function getDeviceId(): string {
    const DEVICE_ID_KEY = 'unit_game_device_id';

    let deviceId: string | null = localStorage.getItem(DEVICE_ID_KEY);

    if (!deviceId) {
        deviceId = uuidv4();
        localStorage.setItem(DEVICE_ID_KEY, deviceId);
    }

    return deviceId;
}

/**
 * Check if the current user has premium status
 */
export async function checkPremiumStatus(): Promise<boolean> {
    try {
        const deviceId = getDeviceId();
        const response = await fetch(`${API_BASE_URL}/api/user/premium-status?deviceId=${encodeURIComponent(deviceId)}`);

        if (!response.ok) {
            throw new Error('Failed to check premium status');
        }

        const data = await response.json();
        return data.isPremium || false;
    } catch (error) {
        console.error('Error checking premium status:', error);
        return false;
    }
}

/**
 * Create a Stripe checkout session and redirect to payment
 */
export async function createCheckoutSession(): Promise<void> {
    try {
        const deviceId = getDeviceId();

        const response = await fetch(`${API_BASE_URL}/api/payment/create-checkout-session`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ deviceId }),
        });

        if (!response.ok) {
            throw new Error('Failed to create checkout session');
        }

        const data = await response.json();

        // Redirect to Stripe Checkout
        if (data.url) {
            window.location.href = data.url;
        } else {
            throw new Error('No checkout URL returned');
        }
    } catch (error) {
        console.error('Error creating checkout session:', error);
        throw error;
    }
}

/**
 * Check if user has played today
 */
export function hasPlayedToday(): boolean {
    const lastPlayedDate = localStorage.getItem('unit_game_last_played_date');
    const today = new Date().toDateString();
    return lastPlayedDate === today;
}

/**
 * Record that user played today
 */
export function recordPlayToday(): void {
    const today = new Date().toDateString();
    localStorage.setItem('unit_game_last_played_date', today);
}
