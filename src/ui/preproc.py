"""
Dash-based HTML code for the preprocessing the ui to be displayed in the
        browser via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc


preproc = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Row([html.H6("Downsampling:")]),
            dbc.Row([html.Hr(className="my-2"),]),
            dbc.Row([dbc.Input(placeholder="New sampling rate",
                               id="preproc-dwnsmpl-rate")],
                    style={"padding": "50px"}),
            dbc.Row([dbc.Button("Apply", id="preproc-dwnsmpl-apply")],
                    style={"padding": "50px"}),
            dbc.Row([], id="preproc-dwnsmpl-result",
                    style={"padding": "50px"}),
            ], width="auto", style={"padding": "50px"}),
        dbc.Col([
            dbc.Row([html.H6("Remove Electrical Humming:")]),
            dbc.Row([html.Hr(className="my-2"),]),
            dbc.Row([dbc.Button("Apply", id="preproc-humming-apply")],
                    style={"padding": "50px"}),
            dbc.Row([], id="preproc-humming-result",
                    style={"padding": "50px"}),
            ], width="auto"),
        dbc.Col([
            dbc.Row([html.H6("General Filter")]),
            dbc.Row([html.Hr(className="my-2"),]),
            dbc.Row([dbc.Input(placeholder="Lower Frequency",
                               id="preproc-fltr-lower")],
                    style={"padding": "10px"}),
            dbc.Row([dbc.Input(placeholder="Upper Frequency",
                               id="preproc-fltr-upper")],
                    style={"padding": "10px"}),
            dbc.RadioItems(
                id="preproc-fltr-type",
                options=[
                    {"label": "Bandpass", "value": 0},
                    {"label": "Bandstop", "value": 1},
                    ],
                labelCheckedClassName="text-success",
                inputCheckedClassName=("border border-success "
                                       "bg-success"),
                style={"padding": "50px"}
                ),
            dbc.Row([dbc.Button("Apply", id="preproc-fltr-apply")],
                    style={"padding": "50px"}),
            dbc.Row([], id="preproc-fltr-result", \
                    style={"padding": "50px"}),
            ], width="auto", style={"padding": "50px"}),
        ], align="center", justify="center"),
    dbc.Row([dbc.Col([html.A(dbc.Button("Next"), href="analyze")],
                     width="auto", align="center")],
            align="center", justify="center")
    ], style={"padding": "50px"}, fluid=True
)
