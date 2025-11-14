import React, { useState, useEffect } from 'react';

const AdminDashboard = () => {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStats();
    // Refresh every 30 seconds
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('https://unitgame-api.herokuapp.com/api/games/stats');
      if (!response.ok) throw new Error('Failed to fetch stats');
      const data = await response.json();
      setStats(data);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center' }}>
        <div style={{ fontSize: '48px' }}>⏳</div>
        <p>Loading statistics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#ff4444' }}>
        <div style={{ fontSize: '48px' }}>❌</div>
        <p>Error: {error}</p>
        <button onClick={fetchStats}>Retry</button>
      </div>
    );
  }

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'system-ui, sans-serif'
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '40px' }}>
        📊 Unit Game Analytics
      </h1>

      {/* Overview Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '20px',
        marginBottom: '40px'
      }}>
        <Card
          icon="🎮"
          title="Total Games"
          value={stats?.total_games?.toLocaleString() || '0'}
          color="#4A90E2"
        />
        <Card
          icon="🌐"
          title="Web Games"
          value={stats?.by_platform?.web?.games?.toLocaleString() || '0'}
          color="#50C878"
        />
        <Card
          icon="📱"
          title="iOS Games"
          value={stats?.by_platform?.ios?.games?.toLocaleString() || '0'}
          color="#007AFF"
        />
        <Card
          icon="🤖"
          title="Android Games"
          value={stats?.by_platform?.android?.games?.toLocaleString() || '0'}
          color="#3DDC84"
        />
      </div>

      {/* Win Rate */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '30px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '40px'
      }}>
        <h2 style={{ marginTop: 0 }}>🏆 Win Distribution</h2>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <WinBar
            label="Player 1"
            wins={stats?.by_winner?.Player1 || 0}
            total={stats?.total_games || 1}
            color="#4A90E2"
          />
          <WinBar
            label="Player 2"
            wins={stats?.by_winner?.Player2 || 0}
            total={stats?.total_games || 1}
            color="#D0021B"
          />
          <WinBar
            label="Draws"
            wins={stats?.by_winner?.Draw || 0}
            total={stats?.total_games || 1}
            color="#999"
          />
        </div>
      </div>

      {/* Platform Breakdown */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '30px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
      }}>
        <h2 style={{ marginTop: 0 }}>📱 Platform Details</h2>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #eee' }}>
              <th style={{ padding: '12px', textAlign: 'left' }}>Platform</th>
              <th style={{ padding: '12px', textAlign: 'right' }}>Games</th>
              <th style={{ padding: '12px', textAlign: 'right' }}>Avg Moves</th>
              <th style={{ padding: '12px', textAlign: 'right' }}>Share</th>
            </tr>
          </thead>
          <tbody>
            {stats?.by_platform && Object.entries(stats.by_platform).map(([platform, data]: [string, any]) => (
              <tr key={platform} style={{ borderBottom: '1px solid #f5f5f5' }}>
                <td style={{ padding: '12px', fontWeight: 'bold', textTransform: 'capitalize' }}>
                  {platform === 'web' && '🌐'} {platform === 'ios' && '📱'} {platform === 'android' && '🤖'} {platform}
                </td>
                <td style={{ padding: '12px', textAlign: 'right' }}>
                  {data.games?.toLocaleString() || 0}
                </td>
                <td style={{ padding: '12px', textAlign: 'right' }}>
                  {data.avg_moves?.toFixed(1) || 'N/A'}
                </td>
                <td style={{ padding: '12px', textAlign: 'right' }}>
                  {((data.games / stats.total_games) * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ textAlign: 'center', marginTop: '40px', color: '#999', fontSize: '14px' }}>
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

const Card = ({ icon, title, value, color }: any) => (
  <div style={{
    background: 'white',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    borderLeft: `4px solid ${color}`
  }}>
    <div style={{ fontSize: '32px', marginBottom: '8px' }}>{icon}</div>
    <div style={{ fontSize: '14px', color: '#666', marginBottom: '8px' }}>{title}</div>
    <div style={{ fontSize: '32px', fontWeight: 'bold', color }}>{value}</div>
  </div>
);

const WinBar = ({ label, wins, total, color }: any) => {
  const percentage = (wins / total * 100).toFixed(1);
  return (
    <div style={{ flex: 1 }}>
      <div style={{ marginBottom: '8px', fontSize: '14px', fontWeight: 'bold' }}>
        {label}: {wins} ({percentage}%)
      </div>
      <div style={{
        height: '30px',
        background: '#f0f0f0',
        borderRadius: '15px',
        overflow: 'hidden'
      }}>
        <div style={{
          height: '100%',
          width: `${percentage}%`,
          background: color,
          transition: 'width 0.3s ease'
        }} />
      </div>
    </div>
  );
};

export default AdminDashboard;