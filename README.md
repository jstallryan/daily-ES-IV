## Delta-Interpolated IV Calculation for ES Futures
This project calculates the expected daily movement of ES (S&P 500 E-mini) futures based on delta-interpolated options data. The calculations utilize the Black-Scholes model, Newton-Raphson method for strike price interpolation, and data from Interactive Brokers (IBKR).

### Features
#### Connects to Interactive Brokers to fetch market data
#### Calculates Black-Scholes delta for call options
#### Finds the closest strikes and call options with deltas closest to the target delta
#### Uses the Newton-Raphson method to interpolate the strike price for the target delta
#### Calculates percent out-of-the-money (OTM) for the identified strike price
#### Computes expected daily movement and cumulative movement based on delta-interpolated incremental variance
#### Visualizes the expected daily movement using Matplotlib
