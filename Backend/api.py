from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import re

app = FastAPI()

# Demo portfolio, replace with your DB logic as needed!
portfolio = {
    "cash": 10000.0,
    "stocks": {}  # e.g., { "AAPL": { "qty": 2, "avg_price": 180.0 } }
}

# --- CORS so frontend can talk to this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For production, set to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Command parser ---
def parse_command(msg):
    text = msg.lower().strip()
    buy = re.match(r"buy\s+(\d+)\s+([\w\.\-]+)", text)
    sell = re.match(r"sell\s+(\d+)\s+([\w\.\-]+)", text)
    price = re.match(r"(get\s*)?price\s*(of)?\s*([\w\.\-]+)", text)
    if "portfolio" in text or "holdings" in text:
        return "portfolio", {}
    if buy:
        return "buy", {"qty": int(buy.group(1)), "symbol": buy.group(2).upper()}
    if sell:
        return "sell", {"qty": int(sell.group(1)), "symbol": sell.group(2).upper()}
    if price:
        return "price", {"symbol": price.group(3).upper()}
    return "unknown", {}

# --- Main chat endpoint ---
@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    intent, args = parse_command(user_message)

    if intent == "buy":
        symbol, qty = args["symbol"], args["qty"]
        info = yf.Ticker(symbol).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return {"result": f"❌ Could not get live price for {symbol}."}
        total = qty * price
        if total > portfolio["cash"]:
            return {"result": f"❌ Not enough cash. You have ₹{portfolio['cash']:.2f}."}
        # Update portfolio
        pos = portfolio["stocks"].get(symbol, {"qty": 0, "avg_price": 0})
        new_qty = pos["qty"] + qty
        new_avg = ((pos["qty"] * pos["avg_price"]) + (qty * price)) / new_qty if new_qty else price
        portfolio["stocks"][symbol] = {"qty": new_qty, "avg_price": new_avg}
        portfolio["cash"] -= total
        return {"result": f"✅ Bought {qty} {symbol} at ₹{price:.2f} each. Cash left: ₹{portfolio['cash']:.2f}"}

    elif intent == "sell":
        symbol, qty = args["symbol"], args["qty"]
        pos = portfolio["stocks"].get(symbol)
        if not pos or pos["qty"] < qty:
            return {"result": f"❌ Not enough {symbol} shares to sell."}
        info = yf.Ticker(symbol).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return {"result": f"❌ Could not get live price for {symbol}."}
        proceeds = qty * price
        pos["qty"] -= qty
        if pos["qty"] == 0:
            del portfolio["stocks"][symbol]
        else:
            portfolio["stocks"][symbol] = pos
        portfolio["cash"] += proceeds
        return {"result": f"✅ Sold {qty} {symbol} at ₹{price:.2f} each. New cash: ₹{portfolio['cash']:.2f}"}

    elif intent == "portfolio":
        lines = [f"💰 Cash: ₹{portfolio['cash']:.2f}"]
        if not portfolio["stocks"]:
            lines.append("No holdings.")
        else:
            lines.append("📊 Holdings:")
            for s, d in portfolio["stocks"].items():
                lines.append(f"• {s}: {d['qty']} @ avg ₹{d['avg_price']:.2f}")
        return {"result": "\n".join(lines)}

    elif intent == "price":
        symbol = args["symbol"]
        info = yf.Ticker(symbol).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            return {"result": f"❌ Could not get live price for {symbol}."}
        return {"result": f"📈 {symbol} price: ₹{price:.2f}"}

    else:
        return {"result": "🤖 Sorry, I didn't get that. Try: Buy 2 TCS.NS, Sell 1 AAPL, Portfolio, Price of RELIANCE.NS"}

