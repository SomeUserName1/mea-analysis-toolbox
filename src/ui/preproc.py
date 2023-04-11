"""
Dash-based HTML code for the preprocessing the ui to be displayed in the \
        browser via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc


preproc = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H6("Remove Electrical Humming:")]),
                dbc.Row([html.Hr(className="my-2"),]),
                dbc.Row([dbc.Button("Apply", id="preproc-humming-apply")], \
                        style={"padding": "50px"}),
                dbc.Row([], id="preproc-humming-result", \
                        style={"padding": "50px"}),
            ], width="auto"),
            dbc.Col([
                dbc.Row([html.H6("Bandpass Filter")]),
                dbc.Row([html.Hr(className="my-2"),]),
                dbc.Row([dbc.Input(placeholder="Lower Cut-off Frequency", \
                        id="preproc-bndpss-lower")], style={"padding": "10px"}),
                dbc.Row([dbc.Input(placeholder="Upper Cut-off Frequency", \
                        id="preproc-bndpss-upper")], style={"padding": "10px"}),
                dbc.RadioItems(
                    id="preproc-bndpss-type",
                    options=[
                        {"label": "Butterworth", "value": 0},
                        {"label": "Chebyshev", "value": 1},
                    ],
                    labelCheckedClassName="text-success",
                    inputCheckedClassName="border border-success bg-success", \
                            style={"padding": "50px"}
                ),
                dbc.Row([dbc.Button("Apply", id="preproc-bndpss-apply")], \
                        style={"padding": "50px"}),
                dbc.Row([], id="preproc-bndpss-result", \
                        style={"padding": "50px"}),
            ], width="auto", style={"padding": "50px"}),
            dbc.Col([
                dbc.Row([html.H6("Bandstop Filter")]),
                dbc.Row([html.Hr(className="my-2"),]),
                dbc.Row([dbc.Input(placeholder="Lower Cut-off Frequency", \
                        id="preproc-bndstp-lower")], style={"padding": "10px"}),
                dbc.Row([dbc.Input(placeholder="Upper Cut-off Frequency", \
                        id="preproc-bndstp-upper")], style={"padding": "10px"}),
                dbc.RadioItems(
                    id="preproc-bndstp-type",
                    options=[
                        {"label": "Butterworth", "value": 0},
                        {"label": "Chebyshev", "value": 1},
                    ],
                    labelCheckedClassName="text-success",
                    inputCheckedClassName="border border-success bg-success", \
                            style={"padding": "50px"}
                ),
                dbc.Row([dbc.Button("Apply", id="preproc-bndstp-apply")], \
                        style={"padding": "50px"}),
                dbc.Row([], id="preproc-bndstp-result", \
                        style={"padding": "50px"}),
            ], width="auto", style={"padding": "50px"}),
            dbc.Col([
                dbc.Row([html.H6("Downsampling:")]),
                dbc.Row([html.Hr(className="my-2"),]),
                dbc.Row([dbc.Input(placeholder="New sampling rate", \
                        id="preproc-dwnsmpl-rate")], style={"padding": "50px"}),
                dbc.Row([dbc.Button("Apply", id="preproc-dwnsmpl-apply")], \
                        style={"padding": "50px"}),
                dbc.Row([], id="preproc-dwnsmpl-result", \
                        style={"padding": "50px"}),
            ], width="auto", style={"padding": "50px"}),
        ], align="center", justify="center"),
        dbc.Row([dbc.Col([html.A(dbc.Button("Next"), href="analyze")], \
                width="auto", align="center")], align="center", justify="center")
    ], style={"padding": "50px"}, fluid=True
)
