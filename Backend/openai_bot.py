import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!*")

import os
import random
import time
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    portfolio: Dict[str, Dict]
    cash_balance: float
    last_action: Optional[str]
    context_injected: Optional[bool]  # New flag for RAG loop fix

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
        return f"Stock symbol {symbol} not found in our database."
    current_price = get_current_price(symbol)
    stock_data = MOCK_STOCK_DATA[symbol]
    change = stock_data["change"]
    volume = stock_data["volume"]
    return (
        f"Stock: {symbol}\n"
        f"Current Price: ${current_price:.2f}\n"
        f"Change: ${change:.2f} ({change/current_price*100:.2f}%)\n"
        f"Volume: {volume:,}\n"
    )

@tool
def get_portfolio_status() -> str:
    """Get the current portfolio status, including cash, total value, and all holdings."""
    current_prices = {symbol: get_current_price(symbol) for symbol in portfolio.holdings.keys()}
    total_value = portfolio.get_portfolio_value(current_prices)
    status = (
        f"Portfolio Status:\n"
        f"Cash Balance: ${portfolio.cash_balance:.2f}\n"
        f"Total Portfolio Value: ${total_value:.2f}\n"
        f"Holdings:\n"
    )
    if portfolio.holdings:
        for symbol, holding in portfolio.holdings.items():
            current_price = current_prices[symbol]
            current_value = holding["quantity"] * current_price
            profit_loss = (current_price - holding["avg_price"]) * holding["quantity"]
            status += (
                f"- {symbol}: {holding['quantity']} shares @ avg ${holding['avg_price']:.2f} | "
                f"Current: ${current_price:.2f} | Value: ${current_value:.2f} | P/L: ${profit_loss:.2f}\n"
            )
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
        return (
            f"✅ {result['message']}\n"
            f"Total cost: ${quantity * current_price:.2f}\n"
            f"Remaining cash: ${portfolio.cash_balance:.2f}"
        )
    else:
        return f"❌ {result['message']}"

@tool
def sell_stock(symbol: str, quantity: int) -> str:
    """Sell a specific quantity of a stock at the current market price."""
    symbol = symbol.upper()
    current_price = get_current_price(symbol)
    result = portfolio.sell_stock(symbol, quantity, current_price)
    if result["success"]:
        return (
            f"✅ {result['message']}\n"
            f"Total received: ${quantity * current_price:.2f}\n"
            f"New cash balance: ${portfolio.cash_balance:.2f}"
        )
    else:
        return f"❌ {result['message']}"

@tool
def get_stock_recommendation(symbol: str) -> str:
    """Get a buy/sell/hold recommendation for a given stock symbol."""
    symbol = symbol.upper()
    if symbol not in MOCK_STOCK_DATA:
        return f"Stock symbol {symbol} not found."
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
    risk = "High" if abs(change) > 10 else "Medium" if abs(change) > 5 else "Low"
    return (
        f"Stock: {symbol}\n"
        f"Current Price: ${current_price:.2f}\n"
        f"Change: ${change:.2f} ({change/current_price*100:.2f}%)\n"
        f"Volume: {MOCK_STOCK_DATA[symbol]['volume']:,}\n"
        f"Recommendation: {rec} (Risk Level: {risk})\n"
    )

@tool
def convert_usd_to_inr(amount: float) -> str:
    """Convert USD amount to INR using a fixed exchange rate, and show both values and the rate."""
    usd_to_inr = 83.0
    inr = amount * usd_to_inr
    return (
        f"{amount:.2f} USD is equal to {inr:.2f} INR. "
        f"(Conversion rate used: 1 USD = {usd_to_inr:.2f} INR)"
    )

@tool
def get_total_profit_loss() -> str:
    """Return the total profit or loss of the current portfolio."""
    current_prices = {symbol: get_current_price(symbol) for symbol in portfolio.holdings.keys()}
    total_pl = 0.0
    details = ""
    for symbol, holding in portfolio.holdings.items():
        current_price = current_prices[symbol]
        profit_loss = (current_price - holding["avg_price"]) * holding["quantity"]
        total_pl += profit_loss
        details += f"{symbol}: {profit_loss:.2f} USD\n"
    return f"Total Profit/Loss: {total_pl:.2f} USD\n{details}"

@tool
def convert_portfolio_to_inr() -> str:
    """Convert your entire portfolio (cash + holdings) value to INR."""
    current_prices = {symbol: get_current_price(symbol) for symbol in portfolio.holdings.keys()}
    total_value_usd = portfolio.get_portfolio_value(current_prices)
    usd_to_inr = 83.0
    total_value_inr = total_value_usd * usd_to_inr
    return (
        f"Your entire portfolio is worth ${total_value_usd:.2f} USD, which is {total_value_inr:.2f} INR. "
        f"(1 USD = {usd_to_inr:.2f} INR)"
    )

@tool
def convert_profit_to_inr() -> str:
    """Convert your total portfolio profit or loss to INR."""
    current_prices = {symbol: get_current_price(symbol) for symbol in portfolio.holdings.keys()}
    total_profit_usd = 0.0
    for symbol, holding in portfolio.holdings.items():
        current_price = current_prices[symbol]
        total_profit_usd += (current_price - holding["avg_price"]) * holding["quantity"]
    usd_to_inr = 83.0
    total_profit_inr = total_profit_usd * usd_to_inr
    return (
        f"Your total portfolio profit/loss is ${total_profit_usd:.2f} USD, which is {total_profit_inr:.2f} INR. "
        f"(1 USD = {usd_to_inr:.2f} INR)"
    )

tools = [
    get_stock_price,
    get_portfolio_status,
    buy_stock,
    sell_stock,
    get_stock_recommendation,
    convert_usd_to_inr,
    get_total_profit_loss,
    convert_portfolio_to_inr,
    convert_profit_to_inr
]

if os.path.exists("knowledge.txt"):
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
    print("RAG Knowledge base loaded!")
else:
    vectorstore = None
    print("No knowledge.txt found! RAG disabled.")

def retrieve_relevant_context(query):
    if not vectorstore:
        return ""
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results]) if results else ""

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    streaming=False,
)
llm_with_tools = llm.bind_tools(tools)

class StockTradingAgent:
    def __init__(self):
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.system_prompt = (
            "You are a helpful and decisive stock trading assistant.\n\n"
            "Your goals:\n"
            "1. Help users check current stock prices, portfolio status, and get recommendations.\n"
            "2. If the user says to buy or sell a stock (e.g., 'buy 10 AAPL'), ALWAYS ask for confirmation ('Are you sure you want to buy 10 AAPL?'). Only proceed when user confirms (e.g., 'yes' or 'confirm').\n"
            "3. Always show the result of any trade (cost or earnings, cash balance, and holdings).\n"
            "4. If the user gives unclear instructions (e.g., just 'yes' or 'okay'), politely ask them to rephrase with full intent (e.g., 'buy 5 NVDA').\n"
            "5. If a user requests a currency conversion (like USD to INR), use the appropriate conversion tool.\n"
            "6. If the user asks about total profit/loss, use the appropriate tool.\n"
            "7. If the user requests 'convert into INR' but does not specify what to convert (cash, profit, portfolio, or a specific amount), politely ask for clarification on what they want to convert.\n"
            "8. If you use info from the knowledge base, clearly state so.\n"
            "9. Always respond in plain text. Do not use Markdown or LaTeX formatting.\n"
            "10. Remind the user that this is a demo with mock data, and not real financial advice.\n\n"
            "Examples:\n"
            "- If user says: 'Buy 10 TSLA', you should:\n"
            "   a) Ask: 'Are you sure you want to buy 10 TSLA?'\n"
            "   b) If the user says yes, then get current price and execute the trade\n"
            "   c) Respond with transaction summary\n\n"
            "Be clear, helpful, and direct. Only take action after explicit confirmation."
        )

    def should_continue(self, state: AgentState) -> str:
        # Only return "tools" if the last message includes explicit tool calls (prevents loop!)
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"

    def call_model(self, state: AgentState) -> Dict:
        messages = state["messages"]
        # Only inject RAG context if not already injected (fixes recursion loop!)
        if not state.get("context_injected", False):
            user_query = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
            context = retrieve_relevant_context(user_query)
            if context:
                messages.append(HumanMessage(content=f"(For context, here is some relevant info from the docs:)\n{context}"))
            state["context_injected"] = True  # Mark as injected
        response = llm_with_tools.invoke(messages)
        return {"messages": [*messages, response], "context_injected": state.get("context_injected", False)}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

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
            state = {**state, "messages": messages, "context_injected": True}
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
        "last_action": None,
        "context_injected": False
    }
    print("🤖 Stock Trading Assistant with RAG initialized!")
    print("💰 Starting cash balance: $10,000")
    print("📊 Available stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, NFLX")
    print("\nType 'quit' to exit\n")
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting Stock Trading Assistant. Goodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Thanks for using the Stock Trading Assistant!")
            break
        if user_input.lower() in ["yes", "ok", "okay", "sure"]:
            print("🤖 Assistant: Please clarify what you'd like to do. For example, 'Buy 5 NVDA' or 'Sell 3 AAPL'.")
            continue
        if not user_input:
            continue

        # Loading animation!
        print("⏳ Thinking", end="", flush=True)
        for _ in range(3):
            time.sleep(0.4)
            print(".", end="", flush=True)
        print("\r" + " " * 25 + "\r", end="", flush=True)  # clear line

        current_state = {
            "messages": [HumanMessage(content=user_input)],
            "portfolio": portfolio.holdings,
            "cash_balance": portfolio.cash_balance,
            "last_action": None,
            "context_injected": False  # Reset for each user input!
        }
        try:
            run_agent_conversation(app, current_state)
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
