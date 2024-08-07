from ib_insync import *
import numpy as np
import scipy.stats as si
import pandas as pd
from datetime import datetime, timedelta
import pytz
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Connect to IBKR
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# --- Function Definitions ---

def calculate_bs_delta(S, K, T, r, sigma):
    """Calculates the Black-Scholes delta for a call option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1, 0.0, 1.0)

def get_available_strikes(symbol, expiry):
    """Fetches the available strikes for the given symbol and expiry date."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "FOP"
    contract.exchange = "CME"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiry
    contract.right = "C"
    contract.multiplier = "50"
    
    details = ib.reqContractDetails(contract)
    strikes = [detail.contract.strike for detail in details]
    return strikes

def get_closest_strikes(available_strikes, reference_strike):
    """Finds the two closest strikes to the reference strike."""
    sorted_strikes = sorted(available_strikes)
    lower_strike = max([strike for strike in sorted_strikes if strike <= reference_strike], default=None)
    upper_strike = min([strike for strike in sorted_strikes if strike >= reference_strike], default=None)
    
    if lower_strike is None and upper_strike is not None:
        lower_strike = min([strike for strike in sorted_strikes if strike < upper_strike], default=None)
    if upper_strike is None and lower_strike is not None:
        upper_strike = max([strike for strike in sorted_strikes if strike > lower_strike], default=None)

    return lower_strike, upper_strike

def get_closest_delta_options(symbol, expiry, reference_strike, current_price, target_delta=0.159):
    """Finds the two call options with deltas closest to the target_delta."""
    available_strikes = get_available_strikes(symbol, expiry)
    if not available_strikes:
        print(f"No available strikes for expiry {expiry}")
        return pd.DataFrame()
    
    lower_strike, upper_strike = get_closest_strikes(available_strikes, reference_strike)
    if lower_strike is None or upper_strike is None:
        print(f"Could not find appropriate strikes around the reference strike {reference_strike}")
        return pd.DataFrame()

    strikes_to_fetch = [lower_strike, upper_strike]
    contracts = [FuturesOption(symbol, expiry, strike, 'C', 'CME', '50', 'USD') for strike in strikes_to_fetch]
    
    # Qualify contracts
    contracts = ib.qualifyContracts(*contracts)
    
    # Fetch market data for options
    tickers = [ib.reqMktData(contract, '', False, False) for contract in contracts]
    
    ib.sleep(2)
    
    risk_free_rate = 0.04
    est = pytz.timezone('US/Eastern')
    current_time = datetime.now(est)
    expiry_time = datetime.strptime(expiry, '%Y%m%d') + timedelta(hours=16)
    expiry_time = est.localize(expiry_time)
    T = (expiry_time - current_time).total_seconds() / (365 * 24 * 60 * 60)
    
    data = []
    for ticker in tickers:
        market_data = ib.ticker(ticker.contract)
        implied_volatility = market_data.modelGreeks.impliedVol if market_data.modelGreeks else None
        if implied_volatility:
            delta = calculate_bs_delta(current_price, ticker.contract.strike, T, risk_free_rate, implied_volatility)
            data.append((ticker.contract.strike, delta))
            print(f"Strike: {ticker.contract.strike}, Delta: {delta:.5f}, IV: {implied_volatility:.5f}, T: {T:.5f}")
    
    options_df = pd.DataFrame(data, columns=['strike', 'delta'])
    
    options_df['delta'] = pd.to_numeric(options_df['delta'], errors='coerce')
    
    # Calculate the difference from target_delta
    options_df['diff'] = (options_df['delta'] - target_delta).abs()
    options_df['diff'] = pd.to_numeric(options_df['diff'], errors='coerce')
    
    # Find the two closest deltas
    if options_df.empty:
        print(f"No options data available for expiry {expiry}")
        return pd.DataFrame()
    
    closest_options = options_df.nsmallest(2, 'diff')
    
    return closest_options

def interpolate_delta(option1, option2, target_delta):
    """Interpolates to find the strike price for the target delta."""
    delta1, strike1 = option1['delta'], option1['strike']
    delta2, strike2 = option2['delta'], option2['strike']
    
    if delta1 == delta2:
        return strike1
    
    strike = strike1 + (target_delta - delta1) * (strike2 - strike1) / (delta2 - delta1)
    print(f"Interpolated strike for delta {target_delta}: {strike:.2f}, based on strikes {strike1:.2f} and {strike2:.2f}")
    
    return strike

def newton_raphson(symbol, expiry, initial_strike, target_delta, current_price, tolerance=0.01, max_iterations=10):
    """Uses Newton-Raphson method to find the strike price for the target delta."""
    strike = initial_strike
    
    for _ in range(max_iterations):
        closest_options = get_closest_delta_options(symbol, expiry, strike, current_price, target_delta)
        if closest_options.empty or len(closest_options) < 2:
            print(f"Skipping expiry {expiry} due to insufficient options data")
            return None  # Skip if we don't have at least two options to interpolate
        
        delta1, strike1 = closest_options.iloc[0]['delta'], closest_options.iloc[0]['strike']
        delta2, strike2 = closest_options.iloc[1]['delta'], closest_options.iloc[1]['strike']
        
        f_strike = interpolate_delta(closest_options.iloc[0], closest_options.iloc[1], target_delta)
        
        if abs(f_strike - strike) < tolerance:
            return f_strike
        
        strike = f_strike
    
    return strike

def calculate_percent_otm(strike, current_price):
    """Calculates the percent out-of-the-money (OTM) for a call option."""
    return ((strike - current_price) / current_price) * 100

def get_trading_days_to_expiry(current_date, expiry_date):
    """Calculates the number of trading days until the expiry date."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=current_date, end_date=expiry_date)
    return len(schedule)

def calculate_incremental_movement(previous_otm, current_otm, previous_days, current_days):
    """Calculates the incremental daily movement based on cumulative OTM."""
    previous_variance = previous_otm ** 2
    current_variance = current_otm ** 2
    incremental_variance = current_variance - previous_variance
    if incremental_variance < 0:
        incremental_variance = 0
    return np.sqrt(incremental_variance)

# --- Main Calculation Function ---

def calculate_expected_movement(symbol, expiry_dates):
    daily_movements = {}
    cumulative_movements = {}

    future = Future(symbol=symbol, lastTradeDateOrContractMonth='202409', exchange='CME', currency='USD')
    ib.qualifyContracts(future)
    ticker = ib.reqMktData(future)
    ib.sleep(2)
    current_price = ticker.marketPrice()

    previous_otm = None
    previous_days = None
    previous_strike = None

    for expiry in expiry_dates:
        if previous_strike is None:
            reference_strike = current_price
        else:
            reference_strike = previous_strike

        initial_strike = reference_strike
        strike = newton_raphson(symbol, expiry, initial_strike, 0.159, current_price)

        if strike is None:
            continue  # Skip if we couldnt find a valid strike

        call_percent_otm = calculate_percent_otm(strike, current_price)

        current_time = datetime.now(pytz.timezone('US/Eastern'))
        expiry_time = datetime.strptime(expiry, '%Y%m%d') + timedelta(hours=16)  # Set expiry time to 16:00 EST
        expiry_time = pytz.timezone('US/Eastern').localize(expiry_time)
        trading_days_to_expiry = get_trading_days_to_expiry(current_time.date(), expiry_time.date())

        if previous_otm is not None and previous_days is not None:
            daily_movement = calculate_incremental_movement(previous_otm, call_percent_otm, previous_days, trading_days_to_expiry)
        else:
            daily_movement = call_percent_otm / np.sqrt(trading_days_to_expiry) if trading_days_to_expiry > 0 else call_percent_otm

        previous_otm = call_percent_otm
        previous_days = trading_days_to_expiry
        previous_strike = strike

        cumulative_movements[expiry] = call_percent_otm
        daily_movements[expiry] = daily_movement

    ib.disconnect()

    return daily_movements, cumulative_movements

# --- Example Usage ---

expiry_dates = ['20240806', '20240807', '20240808', '20240809', '20240812', '20240813', '20240814', '20240815', '20240816', '20240819', '20240820', '20240821', '20240822', '20240823', '20240826', '20240827', '20240828', '20240829', '20240830']

daily_movements, cumulative_movements = calculate_expected_movement('ES', expiry_dates)

for date in expiry_dates:
    print(f"Expected daily movement on {date}: {daily_movements[date]:.2f}%, Cumulative movement up to {date}: {cumulative_movements[date]:.2f}%")

# Customize font properties
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


dates = [datetime.strptime(date, '%Y%m%d') for date in expiry_dates]

plt.figure(figsize=(14, 8))
plt.plot(dates, list(daily_movements.values()), marker='o', linestyle='-', color='blue', label='Daily Expected Movement')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.title(r'1$\sigma$ Expected Daily Movement of ES')
plt.xlabel('Date')
plt.ylabel('Expected Daily Movement (%)')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)
plt.legend()
plt.figtext(0.5, 0.01, 'Based on Delta-Interpolated Incremental Variance', ha='center', fontsize=10, family='serif')
plt.show()