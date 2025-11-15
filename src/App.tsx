// src/App.tsx
import React from 'react';
import './App.css';
import UnitGame from './components/UnitGame';

const App: React.FC = () => {
    return (
        <div className="app-container">
            <UnitGame />
        </div>
    );
};

export default App;