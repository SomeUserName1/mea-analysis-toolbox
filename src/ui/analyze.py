"""
Dash-based HTML code for the analyze ui to be displayed in the browser via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc

analyze = dbc.Container(
    [
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H4("Available Analysis Sections:")],align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([html.Hr(className="my-2"),],align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([html.Strong("Electrode Amplitude Animation"),
                dbc.Input(placeholder="Playback Speed in FPS",
                                id="analyze-animate-fps"),
                 dbc.Input(placeholder="Slow down from real time",
                                id="analyze-animate-slow-down"),
                 dbc.Button("Generate Video (takes some minutes)", className="fas fa-play",
                            id="analyze-animate-play", n_clicks=0)
                         ],align="center",
            justify="center",
            style={"padding": "25px"}),
            dbc.Row([dbc.Button("PSD", id="analyze-psd")],align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([dbc.Button("Periodic-Aperiodic PSD decomposition",
                id="analyze-aperiodic-periodic")],
             align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([dbc.Button("Spectrogram", id="analyze-spec")],align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([html.Strong("Detect peaks by absolute Amplitude"),
                        dbc.Col(html.H6("Baseline std of same electrode factor:"), width="auto"),
                        dbc.Col([ 
                            dbc.Input(placeholder="2.5", id="analyze-peaks-ampl-loc-thresh",),
                        ], width="auto"),
                        dbc.Col(html.H6("Baseline std of all electrodes factor:"), width="auto"),
                        dbc.Col([
                            dbc.Input(placeholder="0.5", id="analyze-peaks-ampl-glob-thresh",)
                        ], width="auto"),
                         dbc.Button("Start", id="analyze-peaks-ampl")],
                        align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([html.Strong("Detect event by moving deviation measures"),
                        dbc.Col(html.H6("Baseline std of moving std or MAD of same electrode factor:"), width="auto"),
                        dbc.Col(dbc.RadioItems(id="analyze-events-method", value=1,
                                       inline=True,
                                        options=[{"label": "Moving Std",
                                        "value": 1},
                                                {"label": "Moving MAD",
                                        "value": 2},],
                                        ), width="auto"),
                        dbc.Col(dbc.Input(placeholder="1", id="analyze-events-thresh"), width="auto"),
                        dbc.Col(dbc.Checklist(options=[{"label": "Export", "value": 1}],
                                value=[], id="analyze-events-export"), width="auto"),
                        dbc.Col(dbc.Input(placeholder="Enter full file path for file to export", id="analyze-events-fname"), width="auto"),

                         dbc.Button("Start", id="analyze-events")],align="center",
            justify="center",
            style={"padding": "25px"}),
                dbc.Row([], id="analyze-events-stats",align="center",
            justify="center",
            style={"padding": "25px"}),
                ], width="auto", align="center"),
            ],
            align="center",
            justify="center",
            style={"padding": "25px"}),
    ], style={"padding": "50px"}, fluid=True
)
