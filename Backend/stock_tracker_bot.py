import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!*")

import json
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import random
import os

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

# Google Gemini and LangSmith imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

# State definition for the agent
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio: Dict[str, Dict]
    cash_balance: float
    last_action: Optional[str]

# Dummy Portfolio Manager
class DummyPortfolio:
    def __init__(self, initial_cash: float = 10000.0):
        self.cash_balance = initial_cash
        self.holdings = {}
        self.transaction_history = []

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        stock_value = sum(
            holding["quantity"] * current_prices.get(symbol, holding["avg_price"])
            for symbol, holding in self.holdings.items()
        )
        return self.cash_balance + stock_value

    def buy_stock(self, symbol: str, quantity: int, price: float) -> Dict:
        total_cost = quantity * price
        if total_cost > self.cash_balance:
            return {"success": False, "message": "Insufficient funds"}

        self.cash_balance -= total_cost
        if symbol in self.holdings:
            old_qty = self.holdings[symbol]["quantity"]
            old_avg = self.holdings[symbol]["avg_price"]
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            self.holdings[symbol] = {"quantity": new_qty, "avg_price": new_avg}
        else:
            self.holdings[symbol] = {"quantity": quantity, "avg_price": price}

        transaction = {
            "type": "BUY",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
        self.transaction_history.append(transaction)

        return {"success": True, "message": f"Bought {quantity} shares of {symbol} at ${price:.2f}"}

    def sell_stock(self, symbol: str, quantity: int, price: float) -> Dict:
        if symbol not in self.holdings or self.holdings[symbol]["quantity"] < quantity:
            return {"success": False, "message": "Insufficient shares to sell"}

        total_value = quantity * price
        self.cash_balance += total_value
        self.holdings[symbol]["quantity"] -= quantity

        if self.holdings[symbol]["quantity"] == 0:
            del self.holdings[symbol]

        transaction = {
            "type": "SELL",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
        self.transaction_history.append(transaction)

        return {"success": True, "message": f"Sold {quantity} shares of {symbol} at ${price:.2f}"}

portfolio = DummyPortfolio()

MOCK_STOCK_DATA = {
    "AAPL": {"price": 175.43, "change": 2.15, "volume": 45000000},
    "GOOGL": {"price": 2841.32, "change": -12.45, "volume": 1200000},
    "MSFT": {"price": 378.92, "change": 5.67, "volume": 28000000},
    "TSLA": {"price": 248.73, "change": -8.92, "volume": 75000000},
    "AMZN": {"price": 3342.88, "change": 15.23, "volume": 3500000},
    "NVDA": {"price": 875.28, "change": 22.45, "volume": 55000000},
    "META": {"price": 489.57, "change": -3.21, "volume": 18000000},
    "NFLX": {"price": 445.82, "change": 7.89, "volume": 8500000}
}

def get_current_price(symbol: str) -> float:
    base_price = MOCK_STOCK_DATA.get(symbol.upper(), {}).get("price", 100.0)
    variation = random.uniform(-0.05, 0.05)
    return round(base_price * (1 + variation), 2)

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current price, change, and volume for a given stock symbol."""
    symbol = symbol.upper()
    if symbol not in MOCK_STOCK_DATA:
        return f"Stock symbol {symbol} not found in our database"
    current_price = get_current_price(symbol)
    stock_data = MOCK_STOCK_DATA[symbol]
    change = stock_data["change"]
    volume = stock_data["volume"]
    return f"""
Stock: {symbol}
Current Price: ${current_price:.2f}
Change: ${change:.2f} ({change/current_price*100:.2f}%)
Volume: {volume:,}
"""

@tool
def get_portfolio_status() -> str:
    """Get the current portfolio status, including cash, total value, and all holdings."""
    current_prices = {symbol: get_current_price(symbol) for symbol in portfolio.holdings.keys()}
    total_value = portfolio.get_portfolio_value(current_prices)
    status = f"""
Portfolio Status:
Cash Balance: ${portfolio.cash_balance:.2f}
Total Portfolio Value: ${total_value:.2f}
Holdings:
"""
    if portfolio.holdings:
        for symbol, holding in portfolio.holdings.items():
            current_price = current_prices[symbol]
            current_value = holding["quantity"] * current_price
            profit_loss = (current_price - holding["avg_price"]) * holding["quantity"]
            status += f"- {symbol}: {holding['quantity']} shares @ avg ${holding['avg_price']:.2f} | Current: ${current_price:.2f} | Value: ${current_value:.2f} | P/L: ${profit_loss:.2f}\n"
    else:
        status += "No current holdings\n"
    return status

@tool
def buy_stock(symbol: str, quantity: int) -> str:
    """Buy a specific quantity of a stock at the current market price."""
    symbol = symbol.upper()
    current_price = get_current_price(symbol)
    result = portfolio.buy_stock(symbol, quantity, current_price)
    if result["success"]:
        return f"✅ {result['message']}\nTotal cost: ${quantity * current_price:.2f}\nRemaining cash: ${portfolio.cash_balance:.2f}"
    else:
        return f"❌ {result['message']}"

@tool
def sell_stock(symbol: str, quantity: int) -> str:
    """Sell a specific quantity of a stock at the current market price."""
    symbol = symbol.upper()
    current_price = get_current_price(symbol)
    result = portfolio.sell_stock(symbol, quantity, current_price)
    if result["success"]:
        return f"✅ {result['message']}\nTotal received: ${quantity * current_price:.2f}\nNew cash balance: ${portfolio.cash_balance:.2f}"
    else:
        return f"❌ {result['message']}"

@tool
def get_stock_recommendation(symbol: str) -> str:
    """Get a buy/sell/hold recommendation for a given stock symbol."""
    symbol = symbol.upper()
    if symbol not in MOCK_STOCK_DATA:
        return f"Stock symbol {symbol} not found"
    current_price = get_current_price(symbol)
    change = MOCK_STOCK_DATA[symbol]["change"]
    if change > 10:
        rec = "STRONG BUY"
    elif change > 5:
        rec = "BUY"
    elif change > -5:
        rec = "HOLD"
    elif change > -10:
        rec = "SELL"
    else:
        rec = "STRONG SELL"
    return f"""
Stock Analysis for {symbol}:
Current Price: ${current_price:.2f}
Recent Change: ${change:.2f}
Recommendation: {rec}
Risk Level: {'High' if abs(change) > 10 else 'Medium' if abs(change) > 5 else 'Low'}
"""

tools = [get_stock_price, get_portfolio_status, buy_stock, sell_stock, get_stock_recommendation]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True, google_api_key=os.getenv("GOOGLE_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

class StockTradingAgent:
    def __init__(self):
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.system_prompt = """
You are a helpful and decisive stock trading assistant.

Your goals:
1. Help users check current stock prices, portfolio status, and get recommendations.
2. If the user says to buy or sell a stock (e.g., "buy 10 AAPL"), immediately execute it using the appropriate tool — do NOT ask for confirmation.
3. Always show the result of any trade (cost or earnings, cash balance, and holdings).
4. If the user gives unclear instructions (e.g., just "yes" or "okay"), politely ask them to rephrase with full intent (e.g., "buy 5 NVDA").
5. Remind the user that this is a demo with mock data, and not real financial advice.

Examples:
- If user says: "Buy 10 TSLA", you should:
   a) Get current price of TSLA
   b) Use buy_stock tool immediately
   c) Respond with transaction summary

Be clear, helpful, and direct. Always take action when intent is obvious.
"""

    def should_continue(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"

    def call_model(self, state: AgentState) -> Dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], AIMessage):
            system_message = AIMessage(content=self.system_prompt)
            messages = [system_message] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [*messages, response]}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

@traceable
def run_agent_conversation(app, initial_state):
    state = initial_state
    while True:
        result = app.invoke(state)
        messages = result["messages"]
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_fn = next((t for t in tools if t.name == tool_name), None)
                tool_result = tool_fn(**tool_args) if tool_fn else f"Tool {tool_name} not found."
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
            state = {**state, "messages": messages}
        else:
            if isinstance(last_message, AIMessage):
                print(f"\n🤖 Assistant: {last_message.content}\n")
            elif isinstance(last_message, ToolMessage):
                print(f"\n📊 Result: {last_message.content}\n")
            break
    return state

def main():
    agent = StockTradingAgent()
    app = agent.build_graph()
    initial_state = {
        "messages": [],
        "portfolio": portfolio.holdings,
        "cash_balance": portfolio.cash_balance,
        "last_action": None
    }
    print("🤖 Stock Trading Assistant initialized!")
    print("💰 Starting cash balance: $10,000")
    print("📊 Available stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX")
    print("\nType 'quit' to exit\n")
    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Thanks for using the Stock Trading Assistant!")
            break
        if user_input.lower() in ["yes", "ok", "okay", "sure"]:
            print("🤖 Assistant: Please clarify what you'd like to do. For example, 'Buy 5 NVDA' or 'Sell 3 AAPL'.")
            continue
        if not user_input:
            continue
        current_state = {
            "messages": [HumanMessage(content=user_input)],
            "portfolio": portfolio.holdings,
            "cash_balance": portfolio.cash_balance,
            "last_action": None
        }
        try:
            run_agent_conversation(app, current_state)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
