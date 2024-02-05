import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime, timedelta

from consts import stock_dict
from calibration import monte_carlo_simulation, calibrate_jump_diffusion_params

# Dash app setup
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Monte Carlo Simulation Dashboard"),
        html.Label("Select Stock"),
        html.Br(),
        dcc.Dropdown(
            id="stock-dropdown",
            options=[{"label": v, "value": k} for k, v in stock_dict.items()],
            value="AAPL",
        ),
        html.Br(),
        html.Label("Or Enter Stock Symbol (e.g. GOOGL)"),
        html.Br(),
        dcc.Input(id="stock-symbol", type="text", value=""),
        html.Br(),
        html.Br(),
        html.Label("Start Date"),
        html.Br(),
        dcc.Input(
            id="start-date",
            type="text",
            value=(datetime.today() - timedelta(days=252)).strftime("%Y-%m-%d"),
        ),
        html.Br(),
        html.Br(),
        html.Label("End Date"),
        html.Br(),
        dcc.Input(
            id="end-date", type="text", value=datetime.today().strftime("%Y-%m-%d")
        ),
        html.Br(),
        html.Br(),
        html.Label("Lookback [Trading Days]"),
        dcc.Slider(id="lookback-days", min=21, max=252, step=21, value=21),
        html.Br(),
        html.Label("Distribution"),
        dcc.Dropdown(
            id="distribution",
            options=[
                {"label": "GBM (Lognormal)", "value": "lognormal"},
                {"label": "Jump Diffusion", "value": "jump_diffusion"},
            ],
            value="lognormal",
        ),
        html.Br(),
        html.Label("Number of Simulations"),
        dcc.Slider(id="simulations", min=10, max=100, step=10, value=30),
        dcc.Graph(id="monte-carlo-plot"),
    ]
)


# Callback to update the plot based on user inputs
@app.callback(
    Output("monte-carlo-plot", "figure"),
    [
        Input("stock-dropdown", "value"),
        Input("stock-symbol", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("lookback-days", "value"),
        Input("distribution", "value"),
        Input("simulations", "value"),
    ],
)
def update_plot(
    selected_stock,
    custom_stock: str,
    start_date: str,
    end_date: str,
    lookback_days: int,
    distribution: str,
    simulations: int,
):
    symbol = custom_stock.upper() if custom_stock else selected_stock

    # calibrate jump diffusion parameters with historical data
    if distribution == "jump_diffusion":
        df_stockdata = yf.download(symbol, start=start_date, end=end_date)
        returns = df_stockdata["Adj Close"].pct_change().dropna().values
        jump_params = calibrate_jump_diffusion_params(returns)
    else:
        jump_params = None

    return monte_carlo_simulation(
        symbol,
        start_date,
        end_date,
        int(lookback_days),
        distribution,
        simulations,
        jump_params,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
