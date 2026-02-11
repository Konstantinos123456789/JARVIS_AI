import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.sentiment import SentimentIntensityAnalyzer
import schedule
import datetime
import time
import pandas as pd

# Load the stock data for multiple stocks
stock_symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'INTC', 'CSCO', 'JPM', 'BAC', 'AXP', 'KO', 'NKE']
stock_data = {}

def update_stock_data():
    """Update stock data for all symbols"""
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    for symbol in stock_symbols:
        try:
            ticker = yf.Ticker(symbol)
            stock_data[symbol] = ticker.history(period='max', start='2020-01-01', end=end_date)
            stock_data[symbol].dropna(inplace=True)
            print(f"Updated data for {symbol}")
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
    print("Stock data update complete.")

# Manually update the stock data initially
try:
    update_stock_data()
except Exception as e:
    print(f"Error during initial stock data update: {e}")

# Preprocess the data for each stock
X = {}
y = {}
models = {}

for symbol in stock_symbols:
    if symbol in stock_data and len(stock_data[symbol]) > 0:
        try:
            X[symbol] = stock_data[symbol][['Open', 'High', 'Low', 'Volume']]
            y[symbol] = stock_data[symbol]['Close']
            
            # Check if we have enough data
            if len(X[symbol]) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X[symbol], y[symbol], test_size=0.2, random_state=42
                )
                
                # Train a random forest regressor model for each stock
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                models[symbol] = model

                # Evaluate the model for each stock
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print(f'MSE for {symbol}: {mse:.2f}')
            else:
                print(f"Not enough data for {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    else:
        print(f"No data available for {symbol}")

# Define a function to generate stock recommendations
def generate_recommendations(user_input):
    """Generate stock recommendations based on user investment goals
    
    Args:
        user_input (str): User's investment goals and risk tolerance
        
    Returns:
        list: List of recommended stock symbols
    """
    try:
        # Analyze the user input using NLP
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(user_input)
        print(f'Sentiment: {sentiment}')

        # Determine the user's investment goals and risk tolerance
        goals = ['growth', 'income', 'capital preservation']
        risk_tolerance = ['conservative', 'moderate', 'aggressive']
        user_goals = [goal for goal in goals if goal.lower() in user_input.lower()]
        user_risk_tolerance = [risk for risk in risk_tolerance if risk.lower() in user_input.lower()]

        # Generate a list of recommended stocks
        recommended_stocks = []
        for symbol in stock_symbols:
            if symbol in models and symbol in stock_data:
                try:
                    model = models[symbol]
                    stock_df = stock_data[symbol]
                    
                    if len(stock_df) == 0:
                        continue
                    
                    # Use recent data for prediction
                    recent_data = stock_df.tail(10)
                    
                    for i in range(len(recent_data)):
                        features = recent_data.iloc[[i]][['Open', 'High', 'Low', 'Volume']]
                        expected_return = model.predict(features)[0]
                        volatility = stock_df['Close'].pct_change().std()

                        # Evaluate the stock based on the user's goals and risk tolerance
                        if 'growth' in user_goals and 'aggressive' in user_risk_tolerance:
                            if expected_return > 0.05 and volatility < 0.1:
                                recommended_stocks.append(symbol)
                                break
                        elif 'income' in user_goals and 'conservative' in user_risk_tolerance:
                            if expected_return > 0.03 and volatility < 0.05:
                                recommended_stocks.append(symbol)
                                break
                        elif 'growth' in user_goals and 'moderate' in user_risk_tolerance:
                            if expected_return > 0.04 and volatility < 0.08:
                                recommended_stocks.append(symbol)
                                break
                except Exception as e:
                    print(f"Error evaluating {symbol}: {e}")
                    continue

        # If no specific criteria matched, return popular tech stocks
        if not recommended_stocks:
            recommended_stocks = ['AAPL', 'MSFT', 'GOOGL']
            
        return list(set(recommended_stocks))  # Remove duplicates
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

# Schedule the update_stock_data function to run daily
schedule.every().day.at("08:00").do(update_stock_data)  # Run at 8:00 AM every day

# Confirm the scheduler is set up correctly
print("Scheduler setup successful.")

# Note: To keep the scheduler running, you would need to add:
# while True:
#     schedule.run_pending()
#     time.sleep(60)
# However, this would block the main program, so it's commented out.
