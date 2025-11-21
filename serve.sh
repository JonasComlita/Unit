#!/bin/bash

# This script serves both backend and frontend with ngrok
# For multiple tunnels, you need ngrok's paid plan OR use the config approach

echo "========================================="
echo "Unit Strategy Game - Server Setup"
echo "========================================="
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "ERROR: build directory not found!"
    echo "Please run 'npm run build' first to create the production build."
    exit 1
fi

echo "Starting services..."
echo ""

# Start the Python backend server on port 3000
echo "[1/3] Starting Flask backend on port 3000..."
python3 server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend server on port 5000
echo "[2/3] Starting frontend on port 5000..."
npx serve -s build -l 5000 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 2

echo ""
echo "========================================="
echo "✓ Services started successfully!"
echo "========================================="
echo ""
echo "Backend:  http://localhost:3000"
echo "Frontend: http://localhost:5000"
echo ""
echo "========================================="
echo "[3/3] Starting ngrok tunnel..."
echo "========================================="
echo ""
echo "NOTE: Free ngrok accounts support 1 tunnel."
echo "Choose which service to expose:"
echo ""
echo "  1) Frontend only (recommended for testing)"
echo "  2) Backend only (for API testing)"
echo "  3) Both (requires ngrok paid plan)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting ngrok tunnel for FRONTEND (port 5000)..."
        echo "Your frontend will be publicly accessible."
        echo "Backend will remain local at http://localhost:3000"
        echo ""
        ngrok http 5000
        ;;
    2)
        echo ""
        echo "Starting ngrok tunnel for BACKEND (port 3000)..."
        echo "Your API will be publicly accessible."
        echo "Frontend will remain local at http://localhost:5000"
        echo ""
        ngrok http 3000
        ;;
    3)
        echo ""
        echo "Starting ngrok tunnels for BOTH services..."
        echo "This requires a paid ngrok plan."
        echo ""
        echo "Creating ngrok config..."
        
        # Create temporary ngrok config
        cat > /tmp/ngrok-config.yml <<EOF
version: "2"
authtoken: YOUR_NGROK_AUTH_TOKEN
tunnels:
  backend:
    proto: http
    addr: 3000
  frontend:
    proto: http
    addr: 5000
EOF
        
        echo "Please edit /tmp/ngrok-config.yml and add your auth token"
        echo "Then run: ngrok start --all --config /tmp/ngrok-config.yml"
        echo ""
        echo "Press Ctrl+C when done to stop all services."
        wait
        ;;
    *)
        echo "Invalid choice. Exiting..."
        kill $BACKEND_PID $FRONTEND_PID
        exit 1
        ;;
esac

# When ngrok exits (user presses Ctrl+C), clean up
echo ""
echo "Stopping services..."
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
echo "All services stopped."