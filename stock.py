import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.sentiment import SentimentIntensityAnalyzer
import schedule
import datetime
import time

# Load the stock data for multiple stocks
stock_symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'INTC', 'CSCO', 'JPM', 'BAC', 'AXP', 'KO', 'NKE']
stock_data = {}


def fetch_stock_stooq(symbol: str) -> pd.DataFrame:
    """Fetch historical stock data from Stooq (free, no API key needed).

    Args:
        symbol (str): Stock ticker symbol e.g. 'AAPL'

    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, Close, Volume columns
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    try:
        df = pd.read_csv(url, parse_dates=['Date'])
        df = df.rename(columns=str.title)           # normalize column names
        df = df.sort_values('Date').reset_index(drop=True)
        df = df[df['Date'] >= '2020-01-01']
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Stooq fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def update_stock_data():
    """Update stock data for all symbols"""
    for symbol in stock_symbols:
        try:
            df = fetch_stock_stooq(symbol)
            if not df.empty:
                stock_data[symbol] = df
                print(f"Updated data for {symbol} ({len(df)} rows)")
            else:
                print(f"No data returned for {symbol}")
        except Exception as e:
            print(f"Error updating {symbol}: {e}")
    print("Stock data update complete.")


# Load data on startup
try:
    update_stock_data()
except Exception as e:
    print(f"Error during initial stock data update: {e}")

# Preprocess and train a model for each stock
X = {}
y = {}
models = {}

for symbol in stock_symbols:
    if symbol in stock_data and len(stock_data[symbol]) > 0:
        try:
            X[symbol] = stock_data[symbol][['Open', 'High', 'Low', 'Volume']]
            y[symbol] = stock_data[symbol]['Close']

            if len(X[symbol]) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X[symbol], y[symbol], test_size=0.2, random_state=42
                )

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                models[symbol] = model

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                print(f'MSE for {symbol}: {mse:.2f}')
            else:
                print(f"Not enough data for {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    else:
        print(f"No data available for {symbol}")


def generate_recommendations(user_input: str) -> list:
    """Generate stock recommendations based on user investment goals.

    Args:
        user_input (str): User's investment goals and risk tolerance

    Returns:
        list: List of recommended stock symbols
    """
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(user_input)
        print(f'Sentiment: {sentiment}')

        goals               = ['growth', 'income', 'capital preservation']
        risk_tolerance      = ['conservative', 'moderate', 'aggressive']
        user_goals          = [g for g in goals if g.lower() in user_input.lower()]
        user_risk_tolerance = [r for r in risk_tolerance if r.lower() in user_input.lower()]

        # Infer risk tolerance from sentiment if user didn't specify
        if not user_risk_tolerance:
            if sentiment['compound'] >= 0.3:
                user_risk_tolerance = ['aggressive']
            elif sentiment['compound'] <= -0.3:
                user_risk_tolerance = ['conservative']
            else:
                user_risk_tolerance = ['moderate']

        recommended_stocks = []
        for symbol in stock_symbols:
            if symbol in models and symbol in stock_data:
                try:
                    model    = models[symbol]
                    stock_df = stock_data[symbol]

                    if len(stock_df) == 0:
                        continue

                    volatility  = stock_df['Close'].pct_change().std()
                    recent_data = stock_df.tail(10)

                    for i in range(len(recent_data)):
                        features        = recent_data.iloc[[i]][['Open', 'High', 'Low', 'Volume']]
                        expected_return = model.predict(features)[0]

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
                        else:
                            # Generic: positive expected return and low volatility
                            if expected_return > 0 and volatility < 0.15:
                                recommended_stocks.append(symbol)
                                break
                except Exception as e:
                    print(f"Error evaluating {symbol}: {e}")
                    continue

        if not recommended_stocks:
            recommended_stocks = ['AAPL', 'MSFT', 'GOOG']

        return list(set(recommended_stocks))

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []


# Schedule daily refresh at 8:00 AM
schedule.every().day.at("08:00").do(update_stock_data)
print("Scheduler setup successful.")

# Note: To keep the scheduler running, you would need:
# while True:
#     schedule.run_pending()
#     time.sleep(60)
