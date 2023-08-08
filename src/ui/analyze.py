"""
Dash-based HTML code for the analyze ui to be displayed in the browser via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc

compute = dbc.Container(
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


"""
Dash-based HTML code for the preprocessing the ui to be displayed in the
        browser via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc


filters = dbc.Container([
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
