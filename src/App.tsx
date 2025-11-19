// src/App.tsx
import React from 'react';
import './App.css';
import UnitGame from './components/UnitGame';
import PaymentModal from './components/PaymentModal';
import { checkPremiumStatus, hasPlayedToday, recordPlayToday } from './services/paymentService';

const App: React.FC = () => {
    const [isPremium, setIsPremium] = React.useState<boolean>(false);
    const [showPaywall, setShowPaywall] = React.useState<boolean>(false);
    const [hasPlayedDaily, setHasPlayedDaily] = React.useState<boolean>(false);
    const [isLoading, setIsLoading] = React.useState<boolean>(true);

    // Check premium status on mount
    React.useEffect(() => {
        const checkStatus = async () => {
            try {
                const premium = await checkPremiumStatus();
                setIsPremium(premium);

                if (!premium) {
                    const playedToday = hasPlayedToday();
                    setHasPlayedDaily(playedToday);
                }
            } catch (error) {
                console.error('Error checking premium status:', error);
            } finally {
                setIsLoading(false);
            }
        };

        checkStatus();
    }, []);

    // Check for payment success/cancel in URL
    React.useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const paymentStatus = urlParams.get('payment');

        if (paymentStatus === 'success') {
            // Payment successful, re-check premium status
            checkPremiumStatus().then(premium => {
                setIsPremium(premium);
                if (premium) {
                    alert('🎉 Payment successful! You now have unlimited access!');
                }
            });
            // Clean up URL
            window.history.replaceState({}, document.title, window.location.pathname);
        } else if (paymentStatus === 'cancelled') {
            alert('Payment was cancelled. You can try again anytime!');
            // Clean up URL
            window.history.replaceState({}, document.title, window.location.pathname);
        }
    }, []);

    const handleGameStart = () => {
        if (isPremium) {
            return true; // Premium users can always play
        }

        // Free users: check daily limit
        if (hasPlayedDaily) {
            setShowPaywall(true);
            return false; // Prevent start
        }

        // Record that user played today
        recordPlayToday();
        setHasPlayedDaily(true);
        return true; // Allow start
    };

    if (isLoading) {
        return (
            <div className="app-container" style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100vh',
                color: 'white',
                fontSize: '20px',
            }}>
                Loading...
            </div>
        );
    }

    return (
        <div className="app-container">
            <UnitGame
                onGameStart={handleGameStart}
                isPremium={isPremium}
            />

            <PaymentModal
                isOpen={showPaywall}
                onClose={() => setShowPaywall(false)}
                gamesPlayed={hasPlayedDaily ? 1 : 0}
            />
        </div>
    );
};

export default App;