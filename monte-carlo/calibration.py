import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def monte_carlo_simulation(
    symbol: str,
    start_date: str,
    end_date: str,
    lookback_days: int,
    distribution: str,
    simulations: int,
    jump_params=None,
) -> go.Figure:
    """
    Perform Monte Carlo simulation for stock price prediction.
    """
    # get historical stock data
    df_stockdata = yf.download(symbol, start=start_date, end=end_date)

    # calculate returns
    df_stockdata["daily_return"] = df_stockdata["Adj Close"].pct_change().dropna()

    # extract data
    initial_price = df_stockdata["Adj Close"].iloc[-1]
    drift = df_stockdata["daily_return"].mean() * lookback_days
    volatility = df_stockdata["daily_return"].std() * np.sqrt(lookback_days)
    days = 252  # trading days per year

    # generate MC simulations
    np.random.seed(42)
    if distribution == "lognormal":
        daily_returns = np.exp(
            (drift - 0.5 * volatility**2)
            + volatility * np.random.normal(0, 1, size=(days, simulations))
        )
    elif distribution == "normal":
        daily_returns = (drift - 0.5 * volatility**2) + volatility * np.random.normal(
            0, 1, size=(days, simulations)
        )
    elif distribution == "jump_diffusion" and jump_params:
        daily_returns = generate_jump_diffusion_returns(
            jump_params["jump_size_mean"],
            jump_params["jump_size_std"],
            jump_params["jump_intensity"],
            drift,
            volatility,
            days,
            simulations,
        )

    price_simulations = initial_price * np.cumprod(daily_returns, axis=0)
    # cap simulated prices
    price_simulations = np.clip(price_simulations, a_min=0.01, a_max=3000)

    fig = px.line(
        price_simulations,
        labels={"index": "Trading Days", "value": "Stock Price"},
        title=f"Monte Carlo Simulation of {symbol} Stock Prices",
    )
    fig.update(layout_showlegend=False)

    # display initial stock price, drift and volatility
    initial_values_text = f"Initial Stock Price: {initial_price:.2f}<br>Drift: {drift:.6f}<br>Volatility: {volatility:.6f}"
    fig.add_annotation(
        text=initial_values_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.15,
        showarrow=False,
        font=dict(size=12, color="black"),
    )

    return fig


def generate_jump_diffusion_returns(
    jump_size_mean: float,
    jump_size_std: float,
    jump_intensity: float,
    drift: float,
    volatility: float,
    days: int,
    simulations: int,
) -> np.ndarray:
    """
    Generate Jump Diffusion returns for Monte Carlo simulation.
    """
    np.random.seed(42)
    dt = 1 / days
    jump_process = np.random.poisson(jump_intensity * dt, size=(days, simulations))
    jump_returns = (
        np.random.normal(jump_size_mean, jump_size_std, size=(days, simulations))
        * jump_process
    )
    drift_returns = (drift - 0.5 * volatility**2) * dt
    diffusion_returns = volatility * np.random.normal(
        0, np.sqrt(dt), size=(days, simulations)
    )
    total_returns = np.exp(drift_returns + diffusion_returns + jump_returns) - 1
    return 1 + total_returns


# calibrate jump diffusion parameters
def calibrate_jump_diffusion_params(returns: np.array) -> dict:
    """
    Calibrate Jump Diffusion model parameters based on historical returns.
    """
    mean_jump_size = np.mean(returns[returns > 0])
    std_jump_size = np.std(returns[returns > 0])
    jump_intensity = len(returns[returns > 0]) / len(returns)
    return {
        "jump_size_mean": mean_jump_size,
        "jump_size_std": std_jump_size,
        "jump_intensity": jump_intensity,
    }
