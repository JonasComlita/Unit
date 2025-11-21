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

    const handleUpgrade = async (planType: 'unlimited' | 'multiplayer') => {
        setIsProcessing(true);
        setError(null);

        try {
            await createCheckoutSession(planType);
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
                maxWidth: '900px',
                width: '95%',
                boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
                border: '2px solid rgba(255, 255, 255, 0.1)',
                maxHeight: '90vh',
                overflowY: 'auto',
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
                        fontSize: '32px',
                        margin: '0 0 10px 0',
                        fontWeight: 'bold',
                    }}>
                        Choose Your Plan
                    </h2>
                    <p style={{
                        color: 'rgba(255, 255, 255, 0.7)',
                        fontSize: '16px',
                        margin: 0,
                    }}>
                        Unlock the full potential of Unit Strategy
                    </p>
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

                <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '20px',
                    justifyContent: 'center',
                }}>
                    {/* Unlimited Plan */}
                    <div style={{
                        flex: '1 1 300px',
                        background: 'rgba(255, 255, 255, 0.05)',
                        borderRadius: '16px',
                        padding: '30px',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        display: 'flex',
                        flexDirection: 'column',
                    }}>
                        <h3 style={{ color: 'white', margin: '0 0 10px 0' }}>Unlimited</h3>
                        <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#4CAF50', marginBottom: '5px' }}>$1.00</div>
                        <div style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '14px', marginBottom: '20px' }}>One-time payment</div>

                        <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 30px 0', flex: 1 }}>
                            {[
                                '♾️ Unlimited single-player games',
                                '🎯 All difficulty levels',
                                '💾 Save game progress',
                                '🚫 No ads',
                            ].map((feature, i) => (
                                <li key={i} style={{ color: 'rgba(255, 255, 255, 0.9)', padding: '8px 0', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>{feature}</li>
                            ))}
                        </ul>

                        <button
                            onClick={() => handleUpgrade('unlimited')}
                            disabled={isProcessing}
                            style={{
                                width: '100%',
                                padding: '15px',
                                background: 'rgba(255, 255, 255, 0.1)',
                                border: '2px solid rgba(76, 175, 80, 0.5)',
                                borderRadius: '10px',
                                color: '#4CAF50',
                                fontSize: '16px',
                                fontWeight: 'bold',
                                cursor: isProcessing ? 'not-allowed' : 'pointer',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            {isProcessing ? 'Processing...' : 'Get Unlimited'}
                        </button>
                    </div>

                    {/* Multiplayer Plan */}
                    <div style={{
                        flex: '1 1 300px',
                        background: 'linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%)',
                        borderRadius: '16px',
                        padding: '30px',
                        border: '2px solid #4CAF50',
                        display: 'flex',
                        flexDirection: 'column',
                        position: 'relative',
                    }}>
                        <div style={{
                            position: 'absolute',
                            top: '-12px',
                            right: '20px',
                            background: '#4CAF50',
                            color: 'white',
                            padding: '4px 12px',
                            borderRadius: '12px',
                            fontSize: '12px',
                            fontWeight: 'bold',
                        }}>
                            BEST VALUE
                        </div>
                        <h3 style={{ color: 'white', margin: '0 0 10px 0' }}>Multiplayer Pro</h3>
                        <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#4CAF50', marginBottom: '5px' }}>$5.00</div>
                        <div style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '14px', marginBottom: '20px' }}>Per month</div>

                        <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 30px 0', flex: 1 }}>
                            {[
                                '🌍 Online Multiplayer',
                                '♾️ Everything in Unlimited',
                                '🏆 Global Leaderboards',
                                '🎨 Exclusive Themes',
                                '⚡ Priority Server Access',
                            ].map((feature, i) => (
                                <li key={i} style={{ color: 'rgba(255, 255, 255, 0.9)', padding: '8px 0', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>{feature}</li>
                            ))}
                        </ul>

                        <button
                            onClick={() => handleUpgrade('multiplayer')}
                            disabled={isProcessing}
                            style={{
                                width: '100%',
                                padding: '15px',
                                background: '#4CAF50',
                                border: 'none',
                                borderRadius: '10px',
                                color: 'white',
                                fontSize: '16px',
                                fontWeight: 'bold',
                                cursor: isProcessing ? 'not-allowed' : 'pointer',
                                boxShadow: '0 4px 15px rgba(76, 175, 80, 0.4)',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            {isProcessing ? 'Processing...' : 'Go Pro'}
                        </button>
                    </div>
                </div>

                <div style={{ textAlign: 'center', marginTop: '30px' }}>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: 'rgba(255, 255, 255, 0.5)',
                            cursor: 'pointer',
                            textDecoration: 'underline',
                        }}
                    >
                        No thanks, I'll stick to free daily games
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
