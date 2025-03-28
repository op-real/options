import flask
from flask import Flask, jsonify, request
from flask_caching import Cache
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from dotenv import load_dotenv # Optional: for managing configs like Redis URL

# --- Configuration ---
load_dotenv() # Optional: Load .env file if you have one

# Cache configuration (using Redis is recommended for production)
# Make sure Redis server is running (e.g., `redis-server`)
CACHE_CONFIG = {
    "CACHE_TYPE": "RedisCache",  # Use 'SimpleCache' for in-memory testing
    "CACHE_DEFAULT_TIMEOUT": 3600,  # 1 hour default timeout
    "CACHE_REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/0") # Get from env or default
}

# Option Pricing Configuration
RISK_FREE_RATE = 0.05  # Placeholder - fetch dynamically for higher accuracy (e.g., ^TNX)
DAYS_AHEAD = 7          # Calculate options for the next 7 calendar days
VOLATILITY_WINDOW = 60 # Days of historical data to calculate volatility
MIN_VOLUME_OPEN_INTEREST = 10 # Filter out thinly traded options

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App and Cache ---
app = Flask(__name__)
app.config.from_mapping(CACHE_CONFIG)
cache = Cache(app)

# --- Helper Functions ---

@cache.memoize(timeout=86400) # Cache SP500 list for a day
def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    logging.info("Fetching S&P 500 tickers from Wikipedia...")
    try:
        # Be cautious: Wikipedia structure can change. Error handling is important.
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Add headers to mimic a browser, sometimes helps avoid blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        tables = pd.read_html(response.text)
        sp500_table = tables[0]
        # Adjust column name if Wikipedia changes it
        tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
        logging.info(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Wikipedia page: {e}")
        return [] # Return empty list on error
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing Wikipedia table: {e}. Structure might have changed.")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_sp500_tickers: {e}")
        return []

@cache.memoize(timeout=600) # Cache market data for 10 minutes
def get_market_data(ticker_symbol):
    """
    Fetches current price, historical data for volatility, and option expiration dates.
    Returns None if fetching fails.
    """
    logging.debug(f"Fetching market data for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)

        # 1. Current Price
        hist = ticker.history(period="5d") # Get recent history for last close
        if hist.empty:
            logging.warning(f"No recent history found for {ticker_symbol}")
            return None
        current_price = hist['Close'].iloc[-1]
        if pd.isna(current_price):
             logging.warning(f"Could not get current price for {ticker_symbol}")
             return None

        # 2. Historical Data for Volatility
        hist_data = ticker.history(period=f"{VOLATILITY_WINDOW+5}d") # Fetch slightly more data
        if len(hist_data) < VOLATILITY_WINDOW / 2 : # Need sufficient data points
             logging.warning(f"Insufficient historical data for volatility for {ticker_symbol}")
             return None

        # 3. Option Expiration Dates
        option_expirations = ticker.options
        if not option_expirations:
            logging.warning(f"No option expirations found for {ticker_symbol}")
            return None # No options available

        logging.debug(f"Successfully fetched market data for {ticker_symbol}")
        return {
            "current_price": current_price,
            "historical_data": hist_data,
            "option_expirations": option_expirations
        }
    except Exception as e:
        # yfinance can throw various errors (network, parsing, rate limits)
        logging.error(f"Error fetching market data for {ticker_symbol} with yfinance: {e}")
        return None

def calculate_historical_volatility(historical_data, window=VOLATILITY_WINDOW):
    """Calculates annualized historical volatility from closing prices."""
    if historical_data is None or len(historical_data) < window:
        return None
    try:
        # Use log returns for volatility calculation
        log_returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
        # Calculate rolling standard deviation, drop NaN
        rolling_std = log_returns.rolling(window=window).std().dropna()
        if rolling_std.empty:
            return None
        # Annualize the volatility (using the most recent calculation)
        # Trading days adjustment (approx 252) can be more precise but 365 is common too
        volatility = rolling_std.iloc[-1] * np.sqrt(252)
        return volatility
    except Exception as e:
        logging.error(f"Error calculating volatility: {e}")
        return None

def black_scholes_call_put(S, K, T, r, sigma):
    """
    Calculates Black-Scholes price for European call and put options.

    S: Underlying asset price
    K: Option strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset
    """
    if T <= 0 or sigma <= 0: # Avoid division by zero or invalid inputs
       # For expired options, price is intrinsic value
       call_price = max(0, S - K)
       put_price = max(0, K - S)
       return call_price, put_price

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

        # Ensure prices are not negative due to floating point issues
        call_price = max(call_price, 0)
        put_price = max(put_price, 0)

        return call_price, put_price
    except OverflowError:
        logging.error(f"OverflowError in BSM calculation for S={S}, K={K}, T={T}, r={r}, sigma={sigma}. Inputs might be extreme.")
        return np.nan, np.nan # Return NaN on numerical error
    except Exception as e:
        logging.error(f"Unexpected error in BSM calculation: {e}")
        return np.nan, np.nan


# --- Main Processing Function for a Single Ticker ---

# Note: Caching option chains directly via yf.Ticker(t).option_chain(exp) can be tricky
# because the result object isn't easily hashable by default memoizers.
# We cache the raw data fetching instead (get_market_data).
def process_ticker(ticker_symbol):
    """Fetches data, calculates volatility and option prices for a single ticker."""
    logging.info(f"Processing ticker: {ticker_symbol}")
    start_time = time.time()

    market_data = get_market_data(ticker_symbol)
    if not market_data:
        return ticker_symbol, {"error": f"Failed to fetch market data for {ticker_symbol}"}

    S = market_data["current_price"]
    hist_data = market_data["historical_data"]
    option_expirations = market_data["option_expirations"]

    # Calculate Volatility
    sigma = calculate_historical_volatility(hist_data)
    if sigma is None or pd.isna(sigma) or sigma <= 0:
        return ticker_symbol, {"error": f"Failed to calculate valid volatility for {ticker_symbol}"}

    # Filter Expirations for the next DAYS_AHEAD
    today = datetime.now().date()
    target_date = today + timedelta(days=DAYS_AHEAD)
    relevant_expirations = [
        exp for exp in option_expirations
        if today <= datetime.strptime(exp, '%Y-%m-%d').date() <= target_date
    ]

    if not relevant_expirations:
        return ticker_symbol, {"options": {}, "underlying_price": S, "calculation_timestamp": datetime.now().isoformat(), "message": "No relevant option expirations found within the next 7 days."}

    ticker_results = {
        "underlying_price": S,
        "calculated_volatility": sigma,
        "calculation_timestamp": datetime.now().isoformat(),
        "options": {}
    }
    errors_for_ticker = []

    # Process each relevant expiration date
    for expiry_date_str in relevant_expirations:
        try:
            expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
            # Time to Expiry (in years) - add small epsilon if expiry is today? No, BSM handles T=0
            T = (expiry_date - today).days / 365.25 # Using 365.25 for avg year length

            # Fetch option chain for this specific expiration
            # Adding a small cache here if fetching per expiration is slow, but yfinance often gets all chains at once.
            # Let's rely on the get_market_data cache primarily. Re-calling yf.Ticker might be ok.
            opt_chain = yf.Ticker(ticker_symbol).option_chain(expiry_date_str)
            options_data = []

            # Process Calls
            calls = opt_chain.calls
            # Process Puts (match strikes later)
            puts = opt_chain.puts.set_index('strike') # Index puts by strike for easy lookup

            # Filter and calculate for calls, then lookup corresponding put
            # Filter based on volume or open interest to focus on relevant strikes
            filtered_calls = calls[
                (calls['volume'].fillna(0) >= MIN_VOLUME_OPEN_INTEREST) |
                (calls['openInterest'].fillna(0) >= MIN_VOLUME_OPEN_INTEREST)
            ]

            for _, row in filtered_calls.iterrows():
                K = row['strike']

                # Calculate BSM prices
                call_price_bsm, put_price_bsm = black_scholes_call_put(S, K, T, RISK_FREE_RATE, sigma)

                # Add market prices for comparison if needed (optional)
                # market_call_price = (row['bid'] + row['ask']) / 2 if (row['bid'] > 0 and row['ask'] > 0) else row['lastPrice']
                # market_put_price = np.nan
                # if K in puts.index:
                #    put_row = puts.loc[K]
                #    market_put_price = (put_row['bid'] + put_row['ask']) / 2 if (put_row['bid'] > 0 and put_row['ask'] > 0) else put_row['lastPrice']

                if not pd.isna(call_price_bsm) and not pd.isna(put_price_bsm):
                     options_data.append({
                        "strike": K,
                        "call_price": round(call_price_bsm, 3),
                        "put_price": round(put_price_bsm, 3),
                        # "market_call_price": round(market_call_price, 3) if not pd.isna(market_call_price) else None, # Optional
                        # "market_put_price": round(market_put_price, 3) if not pd.isna(market_put_price) else None, # Optional
                    })

            if options_data:
                 ticker_results["options"][expiry_date_str] = sorted(options_data, key=lambda x: x['strike'])

        except Exception as e:
            logging.error(f"Error processing expiration {expiry_date_str} for {ticker_symbol}: {e}")
            errors_for_ticker.append(f"Error on expiration {expiry_date_str}: {str(e)}")

    if errors_for_ticker:
        ticker_results["processing_errors"] = errors_for_ticker

    processing_time = time.time() - start_time
    logging.info(f"Finished processing {ticker_symbol} in {processing_time:.2f} seconds.")
    return ticker_symbol, ticker_results


# --- Flask API Route ---

@app.route('/cal_options_prices')
@cache.cached(timeout=900) # Cache the final result for 15 minutes
def calculate_all_options_prices():
    """
    API endpoint to calculate option prices for all S&P 500 stocks
    for expirations within the next 7 days.
    """
    logging.info("Received request for /cal_options_prices")
    start_time = time.time()

    tickers = get_sp500_tickers()
    if not tickers:
        return jsonify({"error": "Could not retrieve S&P 500 ticker list."}), 500

    # Limit number of tickers for testing if needed
    # tickers = tickers[:20]

    results = {}
    errors = {}
    # Use ThreadPoolExecutor for parallel I/O-bound tasks (fetching data)
    # Adjust max_workers based on your system and network capacity
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks
        future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}

        # Process completed tasks
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                ticker_symbol, data = future.result()
                if "error" in data:
                    errors[ticker_symbol] = data["error"]
                else:
                    results[ticker_symbol] = data
            except Exception as exc:
                logging.error(f'{ticker} generated an exception: {exc}')
                errors[ticker] = f"Unhandled exception during processing: {str(exc)}"

    total_time = time.time() - start_time
    logging.info(f"Total processing time for {len(tickers)} tickers: {total_time:.2f} seconds.")

    final_response = {
        "calculation_start_time": datetime.fromtimestamp(start_time).isoformat(),
        "total_processing_time_seconds": round(total_time, 2),
        "data": results,
        "errors": errors,
        "metadata": {
            "tickers_processed": len(results),
            "tickers_failed": len(errors),
            "risk_free_rate": RISK_FREE_RATE,
            "volatility_window_days": VOLATILITY_WINDOW,
            "days_ahead": DAYS_AHEAD
        }
    }

    return jsonify(final_response)

@app.route('/')
def index():
    """Basic index route."""
    return "Option Pricing API Server is running. Use /cal_options_prices endpoint."


# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network
    # Use debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=False)