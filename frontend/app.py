from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
import os

app = Dash(__name__)
data_path = "/Users/cindy/PycharmProjects/NBIStockForecast/inputs"
stocks = ["A", "B", "C", "D", "E"]
periods = [str(i) for i in range(1, 16)]
features = ["bidVolume", "bidPrice", "askVolume", "askPrice", "volume", "price"]

app.layout = html.Div([
    html.H4(
        'Stock Price Analysis',
        style={
            'textAlign': 'center',
            'fontSize': '36px',
            'fontWeight': 'bold',
            'marginTop': '20px'
        }
    ),
    dcc.Graph(
        id="time-series-chart",
        style={
            'height': '700px',
            'width': '100%',
            'margin': 'auto'
        }
    ),
    dcc.Graph(
        id="std-dev-chart",
        style={
            'height': '400px',
            'width': '100%',
            'margin': 'auto',
            'marginTop': '20px'
        }
    ),
    html.Div(
        style={
            'display': 'flex',  # Flexbox layout
            'justifyContent': 'space-between',  # Spread out left and right sections
            'alignItems': 'flex-start',  # Align top of both sections
            'marginTop': '20px'
        },
        children=[
            # Left Section
            html.Div(
                style={'width': '45%'},  # Take up 45% of the width
                children=[
                    html.P("Select stock:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="ticker",
                        options=[{"label": stock, "value": stock} for stock in stocks],
                        value="A",
                        clearable=False,
                    ),
                    html.P("Select period:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="period",
                        options=[{"label": f"Period {p}", "value": p} for p in periods],
                        value="1",
                        clearable=False,
                    ),
                ]
            ),
            # Right Section
            html.Div(
                style={'width': '50%'},  # Take up 50% of the width
                children=[
                    html.P("Select feature to plot:", style={'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id="features",
                        options=[{"label": feature, "value": feature} for feature in features],
                        value=["price"],
                        inline=True,
                        style={'fontSize': '16px', 'marginBottom': '10px'}
                    ),
                    html.P("Select standard deviation:", style={'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id="std-dev-options",
                        options=[
                            {"label": "30-sec Std Dev", "value": "std_30"},
                            {"label": "60-sec Std Dev", "value": "std_60"}
                        ],
                        value=[],
                        inline=True,
                        style={'fontSize': '16px', 'marginBottom': '10px'}
                    ),
                    html.P("Highlight features:", style={'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id="highlight-options",
                        options=[
                            {"label": "High of Day", "value": "high_day"},
                            {"label": "Low of Day", "value": "low_day"},
                            {"label": "Trades", "value": "trades"}
                        ],
                        value=[],
                        inline=True,
                        style={'fontSize': '16px'}
                    ),
                    html.P("Predictions:", style={'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id="label-toggle",
                        options=[{"label": "Show Label", "value": "show_label"}],
                        value=[],  # Default: label not shown
                        inline=True,
                        style={'fontSize': '16px', 'marginBottom': '10px'}
                    ),
                ]
            )
        ]
    )
], style={'fontFamily': 'Arial, sans-serif'})

def load_period_data(stock, period):
    """Loads a specific period CSV for the selected stock."""
    period_file = os.path.join(data_path, stock, f"{period}.csv")
    if os.path.exists(period_file):
        df = pd.read_csv(period_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        return pd.DataFrame()

@app.callback(
    [Output("time-series-chart", "figure"), Output("std-dev-chart", "figure")],
    [
        Input("ticker", "value"),
        Input("period", "value"),
        Input("features", "value"),
        Input("std-dev-options", "value"),
        Input("highlight-options", "value"),
        Input("label-toggle", "value"),
    ]
)
def update_graph(ticker, period, selected_features, std_dev_options, highlight_options, label_toggle):
    df = load_period_data(ticker, period)

    if df.empty:
        return (
            go.Figure().add_annotation(
                text=f"No data available for Stock {ticker} - Period {period}",
                xref="paper", yref="paper", showarrow=False
            ),
            go.Figure().add_annotation(
                text="No standard deviation data available",
                xref="paper", yref="paper", showarrow=False
            )
        )

    # Ensure the timestamp is set as the index for time-based calculations
    df = df.set_index('timestamp')

    # Calculate standard deviation for 30-second and 60-second intervals
    std_30 = None
    std_60 = None
    if 'price' in df.columns:
        std_30 = df['price'].resample('30S').std().dropna()  # Drop NaN values
        std_60 = df['price'].resample('60S').std().dropna()  # Drop NaN values

    # Main time-series chart
    main_fig = go.Figure()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Distinct colors for features
    for i, feature in enumerate(selected_features):
        if feature in df.columns:
            main_fig.add_trace(go.Scatter(
                x=df.index,
                y=df[feature],
                mode='lines',
                name=feature,
                line=dict(color=colors[i % len(colors)])  # Cycle through colors
            ))

    # Add the "label" feature if toggled on
    if "show_label" in label_toggle and "label" in df.columns:
        label_colors = {0: "red", 1: "blue", 2: "green"}
        label_text = {0: "Down", 1: "Hold", 2: "Up"}
        main_fig.add_trace(go.Scatter(
            x=df.index,
            y=df["label"],
            mode="markers+lines",
            name="Label",
            line=dict(color="cyan"),
            marker=dict(
                size=8,
                color=df["label"].map(label_colors),  # Map label values to colors
                symbol="circle"
            ),
            hovertemplate=(
                "Timestamp: %{x}<br>" +
                "Label: %{y}<br>" +
                "Action: %{text}<extra></extra>"
            ),
            text=df["label"].map(label_text),  # Display label meaning in hover text
            yaxis="y2"  # Assign to secondary y-axis
        ))

    # Highlight high/low of the day
    if "high_day" in highlight_options and 'price' in df.columns:
        high_value = df['price'].max()
        high_time = df['price'].idxmax()
        main_fig.add_trace(go.Scatter(
            x=[high_time],
            y=[high_value],
            mode='markers+text',
            marker=dict(color='red', size=10),
            name="High of Day"
        ))
    if "low_day" in highlight_options and 'price' in df.columns:
        low_value = df['price'].min()
        low_time = df['price'].idxmin()
        main_fig.add_trace(go.Scatter(
            x=[low_time],
            y=[low_value],
            mode='markers+text',
            marker=dict(color='blue', size=10),
            name="Low of Day"
        ))

    # Add trades if selected
    if "trades" in highlight_options and 'volume' in df.columns:
        trade_times = df.index[df['volume'] > 0]
        trade_prices = df['price'][df['volume'] > 0]
        main_fig.add_trace(go.Scatter(
            x=trade_times,
            y=trade_prices,
            mode='markers',
            marker=dict(color='green', size=8),
            name="Trades"
        ))

    # Standard deviation chart
    std_fig = go.Figure()
    if "std_30" in std_dev_options and std_30 is not None:
        std_fig.add_trace(go.Scatter(
            x=std_30.index,
            y=std_30,
            mode='markers',
            name="30-sec Std Dev",
            marker=dict(symbol='circle', size=6, color='blue')
        ))
    if "std_60" in std_dev_options and std_60 is not None:
        std_fig.add_trace(go.Scatter(
            x=std_60.index,
            y=std_60,
            mode='markers',
            name="60-sec Std Dev",
            marker=dict(symbol='square', size=6, color='orange')
        ))

    std_fig.update_layout(
        title="Standard Deviation Chart",
        xaxis_title="Timestamp",
        yaxis_title="Standard Deviation",
        legend_title="Std Dev Measures",
        xaxis=dict(
            tickformat="%H:%M:%S",  # Display only the time
        )
    )

    main_fig.update_layout(
        title={
            'text': f"{', '.join(selected_features)} Over Time for Stock {ticker} During Period {period}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Features",
        xaxis=dict(
            tickformat="%H:%M:%S",  # Display only the time
            rangeslider=dict(visible=True)  # Enable range slider
        ),
        yaxis2=dict(
            #title="Label (Red = Down, Blue = Hold, Green = Up)",
            overlaying="y",
            side="right",
            range=[-0.5, 2.5]  # Set range for the label
        )
    )

    return main_fig, std_fig

if __name__ == '__main__':
    app.run(debug=True)