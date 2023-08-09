"""
Dash-based HTML code for the selection the ui to be displayed in the \
        browser via the Dash server.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

next_button = html.A(dbc.Row([dbc.Col([dbc.Button("Next", n_clicks=0)],
        width="auto")], align="center", justify="center",
        style={"padding": "20px"}), href="/analyze")


no_data = dbc.Col([dbc.Alert("Please select at least one electrode!",
                          color="danger"),], width="auto")

def select(grid):
    return dbc.Container(
    [
        dbc.Row([dbc.Col([
            html.H6("Select the desired electrodes and time frame"),
            html.Hr()
        ], width="auto")],
            align="center",
            justify="center",
            style={"padding": "25px"}),
        dbc.Container([
            dbc.Row([dbc.Col([html.H6("Select Image File: ")])]),
            dbc.Row([dbc.Col([
                dbc.Input(id="select-image-file-path",
                          type="text", size="85", required=True,
                          placeholder='Please enter the absolute file path!')],
            )], style={"padding": "25px"}),
        ], id="select-electrode-grid-bg"),
        dbc.Row([
            dbc.Col([
                html.Img(src="/assets/xkcd.png", id="select-mea-setup-img")],
                width="auto", align="center"),
            dbc.Col(dcc.Graph(id="select-electrode-grid",
                              figure=grid), width="auto"),
            dbc.Col(
                dbc.Button("Show selected raw signals",
                           id="select-show-signals", n_clicks=0),
                width="auto")
             ],
            align="center",
            justify="center",
            style={"padding": "25px"}),
        dbc.Row([
            dbc.Col([html.H6("Specify a time range for analysis (s:ms:mus): ")],
                    width="auto"),
            dbc.Col([
                dbc.Input(placeholder="Start time (s:ms:mus)", id="select-start"),
                ], width="auto"),
            dbc.Col([
                dbc.Input(placeholder="Stop time (s:ms:mus)", id="select-stop")
                ], width="auto")
        ],
        id="select-time",
        align="center",
        justify="center",
        style={"padding": "25px"}),
        dbc.Row([
            dbc.Col(
                dbc.Button("Apply electrode and time window selection",
                           id="select-apply", n_clicks=0),
                width="auto")
            ],
            id="select-next",
            align="center",
            justify="center",
            style={"padding": "25px"}),
        dbc.Row([], id="select-output-dummy"),
    ],
    style={"padding": "50px"}, fluid=True
)
