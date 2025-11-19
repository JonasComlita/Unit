// src/components/PaymentModal.tsx
import React, { useState } from 'react';
import { createCheckoutSession } from '../services/paymentService';

interface PaymentModalProps {
    isOpen: boolean;
    onClose: () => void;
    gamesPlayed: number;
}

const PaymentModal: React.FC<PaymentModalProps> = ({ isOpen, onClose, gamesPlayed }) => {
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    if (!isOpen) return null;

    const handleUpgrade = async () => {
        setIsProcessing(true);
        setError(null);

        try {
            await createCheckoutSession();
            // User will be redirected to Stripe Checkout
        } catch (err) {
            setError('Failed to start payment process. Please try again.');
            setIsProcessing(false);
        }
    };

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
        }}>
            <div style={{
                background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
                borderRadius: '20px',
                padding: '40px',
                maxWidth: '500px',
                width: '90%',
                boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
                border: '2px solid rgba(255, 255, 255, 0.1)',
            }}>
                {/* Header */}
                <div style={{
                    textAlign: 'center',
                    marginBottom: '30px',
                }}>
                    <div style={{
                        fontSize: '48px',
                        marginBottom: '15px',
                    }}>
                        🎮
                    </div>
                    <h2 style={{
                        color: 'white',
                        fontSize: '28px',
                        margin: '0 0 10px 0',
                        fontWeight: 'bold',
                    }}>
                        Upgrade to Premium
                    </h2>
                    <p style={{
                        color: 'rgba(255, 255, 255, 0.7)',
                        fontSize: '16px',
                        margin: 0,
                    }}>
                        You've played your free game for today
                    </p>
                </div>

                {/* Features */}
                <div style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                    padding: '25px',
                    marginBottom: '30px',
                }}>
                    <h3 style={{
                        color: 'white',
                        fontSize: '20px',
                        margin: '0 0 20px 0',
                    }}>
                        Premium Benefits
                    </h3>
                    <ul style={{
                        listStyle: 'none',
                        padding: 0,
                        margin: 0,
                    }}>
                        {[
                            '♾️ Unlimited games',
                            '🎯 All difficulty levels',
                            '💾 Save game progress',
                            '🎨 Exclusive themes',
                            '🏆 Leaderboard access',
                        ].map((feature, index) => (
                            <li key={index} style={{
                                color: 'rgba(255, 255, 255, 0.9)',
                                fontSize: '16px',
                                padding: '10px 0',
                                borderBottom: index < 4 ? '1px solid rgba(255, 255, 255, 0.1)' : 'none',
                            }}>
                                {feature}
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Pricing */}
                <div style={{
                    textAlign: 'center',
                    marginBottom: '25px',
                }}>
                    <div style={{
                        fontSize: '48px',
                        fontWeight: 'bold',
                        color: '#4CAF50',
                        marginBottom: '5px',
                    }}>
                        $4.99
                    </div>
                    <div style={{
                        color: 'rgba(255, 255, 255, 0.6)',
                        fontSize: '14px',
                    }}>
                        One-time payment • Lifetime access
                    </div>
                </div>

                {/* Error Message */}
                {error && (
                    <div style={{
                        background: 'rgba(220, 53, 69, 0.2)',
                        border: '1px solid rgba(220, 53, 69, 0.5)',
                        borderRadius: '8px',
                        padding: '12px',
                        marginBottom: '20px',
                        color: '#ff6b6b',
                        fontSize: '14px',
                        textAlign: 'center',
                    }}>
                        {error}
                    </div>
                )}

                {/* Buttons */}
                <div style={{
                    display: 'flex',
                    gap: '12px',
                }}>
                    <button
                        onClick={onClose}
                        disabled={isProcessing}
                        style={{
                            flex: 1,
                            padding: '15px',
                            background: 'rgba(255, 255, 255, 0.1)',
                            border: '2px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '10px',
                            color: 'white',
                            fontSize: '16px',
                            fontWeight: 'bold',
                            cursor: isProcessing ? 'not-allowed' : 'pointer',
                            opacity: isProcessing ? 0.5 : 1,
                            transition: 'all 0.2s ease',
                        }}
                    >
                        Maybe Later
                    </button>
                    <button
                        onClick={handleUpgrade}
                        disabled={isProcessing}
                        style={{
                            flex: 1,
                            padding: '15px',
                            background: isProcessing
                                ? 'linear-gradient(135deg, rgba(76, 175, 80, 0.5) 0%, rgba(56, 142, 60, 0.5) 100%)'
                                : 'linear-gradient(135deg, #4CAF50 0%, #388E3C 100%)',
                            border: 'none',
                            borderRadius: '10px',
                            color: 'white',
                            fontSize: '16px',
                            fontWeight: 'bold',
                            cursor: isProcessing ? 'not-allowed' : 'pointer',
                            boxShadow: isProcessing ? 'none' : '0 4px 15px rgba(76, 175, 80, 0.4)',
                            transition: 'all 0.2s ease',
                        }}
                    >
                        {isProcessing ? 'Processing...' : 'Upgrade Now'}
                    </button>
                </div>

                {/* Security Badge */}
                <div style={{
                    textAlign: 'center',
                    marginTop: '20px',
                    color: 'rgba(255, 255, 255, 0.5)',
                    fontSize: '12px',
                }}>
                    🔒 Secure payment powered by Stripe
                </div>
            </div>
        </div>
    );
};

export default PaymentModal;
