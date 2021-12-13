import dash
import pandas as pd, numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import resample_ohlc
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


app = dash.Dash(__name__)
server = app.server

colors = {"background": "#ffffff", "text": "##263c5c"}

# Read in COT data and create features
cot = pd.read_csv("cot_btc_12DEC2021.csv", index_col="date", parse_dates=True)

# Read in BTC price data (30-minute) and create features
btc_30m = pd.read_csv("btc_prices_12DEC2021.csv", index_col="date", parse_dates=True)
btc_1d = resample_ohlc(btc_30m, "D")
btc_cst = (
    btc_30m[btc_30m.index.hour > 16].tz_localize("UTC").tz_convert("America/Chicago")
)
btc_cst = btc_cst[(btc_cst.index.hour == 15) & (btc_cst.index.minute == 30)]
btc_cst.index = pd.to_datetime(btc_cst.index.date)
settle_prices = btc_cst["close"].to_frame() / 100
settle_prices["shifted"] = settle_prices["close"].shift(-3)
cot["settle"] = settle_prices.loc[cot.index, "shifted"]
cot["returns"] = cot["settle"].pct_change().shift(-1)

# Get Long and Short positioning as percentage of total float
spread_cols = [col for col in cot.columns if "Spread" in col]
long_cols = [
    "Dealer Longs",
    "Asset Manager Longs",
    "Leveraged Funds Longs",
    "Non Reportable Longs",
    "Other Reportable Longs",
]
short_cols = [
    "Dealer Shorts",
    "Asset Manager Shorts",
    "Leveraged Funds Shorts",
    "Non Reportable Shorts",
    "Other Reportable Shorts",
]
cot["oi_less_spreads"] = cot["Open Interest"] - cot[spread_cols].sum(axis=1)
short_pos = cot[short_cols].divide(cot["oi_less_spreads"], axis=0)
long_pos = cot[long_cols].divide(cot["oi_less_spreads"], axis=0)
combined_pos = pd.concat([short_pos, long_pos], axis=1)

# Create clustering model
n_components = 4
n_clusters = 5

pca = PCA(n_components=n_components)
decomp = pca.fit_transform(combined_pos)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(decomp)
cot["cluster"] = kmeans.labels_
combined_pos["cluster"] = kmeans.labels_
grp_cluster = combined_pos.groupby("cluster").mean()

# Create clustering chart
fig_cluster = px.scatter(
    cot.reset_index(),
    x="date",
    y="settle",
    color="cluster",
    height=600,
    title="Clusters vs Price",
)
# fig_cluster.add_trace(go.Scatter(x=cot.index, y=cot["settle"], mode="lines", name=None))

cluster = 0
df_mean = grp_cluster.loc[cluster].to_frame()
df_mean.columns = ["% of Total"]
df_mean.index.name = "Category"
df_mean.reset_index(inplace=True)
fig_means = px.bar(
    df_mean,
    x="Category",
    y="% of Total",
    title=f"Positioning Mix - Cluster {cluster}",
    height=800,
)


@app.callback(Output("category-plot", "figure"), Input("category_oi", "value"))
def plot_open_interest(category):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=cot.index, y=cot[category], name="Open Interest"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=cot.index, y=cot["settle"], name="Bitcoin Price"), secondary_y=True
    )
    fig.update_layout(
        title_text="CME Open Interest",
        height=600,
    )
    fig.update_yaxes(title_text="Open Interest (Contracts)")
    fig.update_yaxes(title_text="Bitcoin Price (USD)", secondary_y=True)

    return fig


@app.callback(Output("correlation_plot", "figure"), Input("category_cor", "value"))
def plot_correlation(category):
    oi_pct_change = cot[category].pct_change()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oi_pct_change, y=cot["returns"], mode="markers"))
    fig.update_layout(
        title="Price Correlation",
        xaxis_title=f"% Change, {category}",
        yaxis_title="% Return, BTC",
        height=600,
        width=800,
    )

    return fig


@app.callback(Output("correlation_text", "children"), Input("category_cor", "value"))
def calculate_correlation(category):
    df = (
        cot.copy()
        .assign(corr_variable=lambda x: np.log(x[category] + 1).diff())
        .fillna(0)
        .dropna()
    )
    x = df["corr_variable"].values.reshape(-1, 1)
    y = df["returns"].values

    lr = LinearRegression()
    lr.fit(x, y)
    r2 = lr.score(x, y) * 100

    # return df[["returns", "corr_variable"]].head().round(3).to_string()
    return f"Equation: {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}    ||    r^2: {r2:.2f}"


@app.callback(Output("cluster_means", "figure"), Input("category_cluster", "value"))
def plot_clusters(cluster):
    df_mean = grp_cluster.loc[cluster].to_frame()
    df_mean.columns = ["% of Total"]
    df_mean.index.name = "Category"
    df_mean.reset_index(inplace=True)
    fig = px.bar(
        df_mean,
        x="Category",
        y="% of Total",
        title=f"Positioning Mix - Cluster {cluster}",
        height=800,
    )

    return fig


@app.callback(
    Output("tabs_content", "children"),
    Input("app_tabs", "value"),
)
def render_tab(tab):
    if tab == "open_interest":
        return html.Div(
            [
                # html.H3("Tab content 1"),
                dcc.Graph(id="category-plot"),
                html.Div(
                    dcc.Dropdown(
                        id="category_oi",
                        options=[{"label": i, "value": i} for i in cot.columns[:-2]],
                        value="Open Interest",
                    ),
                    style={"width": "50%"},
                ),
            ]
        )
    elif tab == "returns_correlation":
        return html.Div(
            [
                # html.H3("Tab content 2"),
                dcc.Graph(id="correlation_plot"),
                html.Div(id="correlation_text"),
                html.Br(),
                html.Br(),
                html.Div(
                    dcc.Dropdown(
                        id="category_cor",
                        options=[{"label": i, "value": i} for i in cot.columns[:-2]],
                        value="Open Interest",
                    ),
                    style={"width": "50%"},
                ),
            ]
        )
    elif tab == "clustering":
        return html.Div(
            [
                dcc.Graph(id="cluster_plot", figure=fig_cluster),
                html.Div(
                    dcc.Dropdown(
                        id="category_cluster",
                        options=[{"label": i, "value": i} for i in grp_cluster.index],
                        value=0,
                    ),
                    style={"width": "50%"},
                ),
                dcc.Graph(id="cluster_means", figure=fig_means),
            ]
        )


# App Layout
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="COT Bitcoin Analysis",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="Insights from the CFTC Commitment of Traders Report",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Br(),
        # dcc.Graph(id="category-plot"),
        dcc.Tabs(
            id="app_tabs",
            value="open_interest",
            children=[
                dcc.Tab(label="Open Interest", value="open_interest"),
                dcc.Tab(label="Returns Correlation", value="returns_correlation"),
                dcc.Tab(label="Clustering", value="clustering"),
            ],
        ),
        html.Div(id="tabs_content"),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
