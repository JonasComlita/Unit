# Premium User System - Quick Start Guide

## ✅ Completed
- [x] Database module created
- [x] Backend endpoints implemented
- [x] Frontend payment service created
- [x] Payment modal designed
- [x] App.tsx integrated
- [x] Dependencies installed (npm & pip)

## 🔧 Next Steps

### 1. Create Stripe Product & Price

1. Go to: https://dashboard.stripe.com/test/products
2. Click "Add Product"
3. Fill in:
   - **Name:** Unit Strategy Game - Premium
   - **Price:** $4.99 (one-time payment)
4. Save and copy the **Price ID** (starts with `price_`)

### 2. Update .env File

Add this line to your `.env` file:

```env
STRIPE_PRICE_ID=price_xxxxxxxxxxxxx
```

Replace `price_xxxxxxxxxxxxx` with your actual Price ID from step 1.

### 3. Set Up Webhook (For Testing)

Open a new terminal and run:

```bash
stripe listen --forward-to localhost:3000/api/payment/webhook
```

This will output a webhook secret like `whsec_xxxxx`. Add it to `.env`:

```env
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxx
```

**Keep this terminal running while testing!**

### 4. Start the Backend

In a new terminal:

```bash
cd /home/jonas/Documents/unit-strategy-game
python server.py
```

You should see:
```
Database initialized
 * Running on http://0.0.0.0:3000
```

### 5. Start the Frontend

The frontend is already running with `npm run build`. If you want to run in dev mode:

```bash
npm start
```

## 🧪 Testing

1. **Open the app** in your browser
2. **Play 1 game** - this is your free game for today
3. **Try to start another game** - payment modal should appear
4. **Click "Upgrade Now"**
5. **Use test card:** `4242 4242 4242 4242`
6. **Complete payment**
7. **Verify unlimited access** - you can now play multiple games

**Note:** The daily limit resets at midnight. To test again without waiting, clear localStorage:
```javascript
localStorage.removeItem('unit_game_last_played_date');
```

## 📊 Check Database

```bash
sqlite3 game.db "SELECT * FROM users;"
```

You should see your device_id with `is_premium = 1`.

## 🐛 Troubleshooting

### Stripe CLI not installed?
```bash
# macOS
brew install stripe/stripe-cli/stripe

# Linux
wget https://github.com/stripe/stripe-cli/releases/download/v1.19.4/stripe_1.19.4_linux_x86_64.tar.gz
tar -xvf stripe_1.19.4_linux_x86_64.tar.gz
sudo mv stripe /usr/local/bin/
```

### Port 3000 already in use?
Change the port in `server.py`:
```python
port = int(os.environ.get('PORT', 3001))  # Changed to 3001
```

And update the API URL in `paymentService.ts`:
```typescript
const API_BASE_URL = 'http://localhost:3001';
```

## 📝 Current Configuration

- **Free games limit:** 1 game per day
- **Price:** $4.99 (one-time payment)
- **Database:** SQLite (`game.db`)
- **Backend port:** 3000
- **Frontend:** React dev server

## 🎯 Ready to Test!

Once you complete steps 1-5 above, the payment system will be fully functional!
