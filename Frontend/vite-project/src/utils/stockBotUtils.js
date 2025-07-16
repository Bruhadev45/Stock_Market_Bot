export function parseTradeIntent(message) {
  const buySellPattern = /(buy|purchase|add|sell|remove)[^\d]*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)?[^\w]*([a-zA-Z\.]{2,10}|\bapple\b|\btcs\b|\bgoogle\b|\bamazon\b|\btata\b|\breliance\b)/i;
  const wordsToNums = { one:1, two:2, three:3, four:4, five:5, six:6, seven:7, eight:8, nine:9, ten:10 };
  const match = message.match(buySellPattern);
  if (!match) return null;
  let [, action, qty, symbol] = match;
  action = action.toLowerCase().includes("buy") || action.toLowerCase().includes("purchase") || action.toLowerCase().includes("add") ? "buy" : "sell";
  qty = qty ? (isNaN(qty) ? wordsToNums[qty.toLowerCase()] : parseInt(qty)) : 1;
  symbol = symbol.toUpperCase();
  const symbolMap = {
    APPLE: "AAPL",
    GOOGLE: "GOOGL",
    AMAZON: "AMZN",
    TCS: "TCS.NS",
    RELIANCE: "RELIANCE.NS",
    TATA: "TATAMOTORS.NS"
  };
  if (symbolMap[symbol]) symbol = symbolMap[symbol];
  return { action, quantity: qty, symbol };
}
