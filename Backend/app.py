import streamlit as st
from realtime_bot import get_current_price, buy_stock, sell_stock, get_portfolio_status, portfolio

st.set_page_config(page_title="Stock Bot", page_icon=":chart_with_upwards_trend:")

# ---- SIDEBAR ----
st.sidebar.title("Portfolio & Features")

# Show current cash balance
st.sidebar.markdown(f"**Cash Balance:** {portfolio.cash_balance:.2f}")

# Show portfolio holdings in a table
if portfolio.holdings:
    st.sidebar.subheader("Your Holdings")
    data = []
    for symbol, holding in portfolio.holdings.items():
        data.append({
            "Symbol": symbol,
            "Quantity": holding['quantity'],
            "Avg Price": holding['avg_price'],
        })
    st.sidebar.table(data)
else:
    st.sidebar.info("No current holdings.")

# Optional sidebar features
if st.sidebar.button("Show Portfolio Status"):
    st.sidebar.text(get_portfolio_status())

# ---- MAIN PAGE ----
st.title("📈 Real-Time Stock Analysis Bot")

st.write(
    """
    Example commands:  
    `get price of TCS.NS`  
    `get price of AAPL`  
    `buy 5 TCS.NS`  
    `sell 3 AAPL`  
    `get portfolio status`
    """
)

user_input = st.text_input("Enter your command:", "")

if st.button("Submit"):
    command = user_input.strip().lower()
    if command.startswith("get price of"):
        try:
            symbol = user_input.split("of", 1)[1].strip().upper()
            price = get_current_price(symbol)
            if price:
                st.success(f"Current price of {symbol}: {price:.2f}")
            else:
                st.error("Could not fetch price. Check the symbol.")
        except Exception as e:
            st.error(f"Error: {e}")
    elif command.startswith("buy"):
        try:
            parts = user_input.strip().split()
            if len(parts) >= 3 and parts[1].isdigit():
                quantity = int(parts[1])
                symbol = parts[2].upper()
                result = buy_stock(symbol, quantity)
                st.info(result)
            else:
                st.warning("Format should be: buy <quantity> <symbol>. Example: buy 5 TCS.NS")
        except Exception as e:
            st.error(f"Error: {e}")
    elif command.startswith("sell"):
        try:
            parts = user_input.strip().split()
            if len(parts) >= 3 and parts[1].isdigit():
                quantity = int(parts[1])
                symbol = parts[2].upper()
                result = sell_stock(symbol, quantity)
                st.info(result)
            else:
                st.warning("Format should be: sell <quantity> <symbol>. Example: sell 3 AAPL")
        except Exception as e:
            st.error(f"Error: {e}")
    elif "portfolio" in command:
        try:
            status = get_portfolio_status()
            st.text(status)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Command not recognized. Try: 'get price of TCS.NS', 'buy 5 TCS.NS', 'sell 3 AAPL', or 'get portfolio status'.")
