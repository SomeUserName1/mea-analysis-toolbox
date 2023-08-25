"""
Dash-based HTML code for the analyze ui to be displayed in the browser via the Dash server.
"""
from enum import Enum

from dash import html
import dash_bootstrap_components as dbc
import numpy as np

# , style={"padding": "50px"}
# width="auto"
# , align="center", justify="center"
# fluid=True
# , align="center", justify="center", style={"padding": "25px"}


class TimeSeriesPlottable(Enum):
    SIG = 0
    ENV = 1
    DERV = 2
    MV_AVG = 3
    MV_MAD = 4
    MV_VAR = 5
    PEAKS = 6
    BURSTS = 7
    SEIZURE = 8


channels_table = dbc.Card(
    dbc.CardBody([], id="channels-table")
)

peaks_table = dbc.Card(
    dbc.CardBody([], id="peaks-table")
)

bursts_table = dbc.Card(
    dbc.CardBody([], id="bursts-table")
)

seizure_table = dbc.Card(
    dbc.CardBody([], id="seizure-table")
)

network_table = dbc.Card(
    dbc.CardBody([], id="network-table")
)


result_tables = dbc.Col(dbc.Tabs([
    dbc.Tab(channels_table, label="Channels"),
    dbc.Tab(peaks_table, label="Peaks"),
    dbc.Tab(bursts_table, label="Bursts"),
    dbc.Tab(seizure_table, label="Seizure"),
    dbc.Tab(network_table, label="Network")
    ]), width='auto')

#def generate_table(df):
#    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, index=True)


def generate_table(dataframe):
    rows = []
    for i in range(len(dataframe)):
        cols = []
        for col in dataframe.columns:
            field = dataframe.iloc[i][col]
            if type(field) == float:
                cols.append(html.Td(f"{dataframe.iloc[i][col]:.8f}"))
            elif type(field) == np.ndarray:
                cols.append(html.Td(f"shape: {dataframe.iloc[i][col].shape}"))
            else:
                cols.append(html.Td(f"{dataframe.iloc[i][col]}"))

        rows.append(html.Tr(cols))

    return (
        html.Table([
            html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
            html.Tbody(rows)],
            className='table table-bordered table-hover table-responsive')
        )


filters = dbc.AccordionItem([
    # Rereference (to remove noise across all channels)
    # Line Noise
    dbc.Row([
        dbc.Row([dbc.Button("Remove Line Noise (EU)", id="analyze-linenoise-apply")]),
        dbc.Row([], id="analyze-linenoise-result"),
    ], style={"padding": "25px"}, class_name="border rounded-3"),
    # Custom frequencies
    dbc.Row([
        dbc.Row([html.H6("Custom Frequency Filter")]),
        dbc.Row([html.Hr(className="my-2"),]),
        dbc.Row([
            dbc.Col([dbc.Input(placeholder="Lower Frequency", id="analyze-fltr-lower")]),
            dbc.Col([dbc.Input(placeholder="Upper Frequency", id="analyze-fltr-upper")])
            ]),
        dbc.RadioItems(id="analyze-fltr-type", options=[{"label": "Bandpass", "value": 0}, 
                                                        {"label": "Bandstop", "value": 1},],
            labelCheckedClassName="text-success", inputCheckedClassName=("border border-success bg-success"),
            style={"padding": "25px"}),
        dbc.Row([dbc.Button("Apply", id="analyze-fltr-apply")]),
        dbc.Row([], id="analyze-fltr-result"),
        ], style={"padding": "25px"}, class_name="border rounded-3"),
    # Downsample
    dbc.Row([
        dbc.Col([dbc.Input(placeholder="New sampling rate", id="analyze-dwnsmpl-rate")]),
        dbc.Col([dbc.Button("Downsample", id="analyze-dwnsmpl-apply")]),
        dbc.Row([], id="analyze-dwnsmpl-result"),
    ], style={"padding": "25px"}, class_name="border rounded-3"),

], title="Filter")



basics = dbc.AccordionItem([
    # SNR
    dbc.Row([dbc.Button("SNR", id="analyze-snr")], style={"padding": "5px"}),
    # RMS
    dbc.Row([dbc.Button("RMS", id="analyze-rms")], style={"padding": "5px"}),
    # Entropies
    dbc.Row([dbc.Button("Approximate Entropy", id="analyze-ent")], style={"padding": "5px"}),
#    # Envelope
#    dbc.Row([dbc.Button("Envelope", id="analyze-env")], style={"padding": "5px"}),
#    # Derivative
#    dbc.Row([dbc.Button("Derivative", id="analyze-derv")], style={"padding": "5px"}),
#    # Mv mean
#    dbc.Row([dbc.Button("Moving Average", id="analyze-mean")], style={"padding": "5px"}),
#    # mv mad
#    dbc.Row([dbc.Button("Moving Mean Absolute Deviation", id="analyze-mad")], style={"padding": "5px"}),
], title="Basic Properties")

spectral = dbc.AccordionItem([
    # PSD
    dbc.Row([dbc.Button("PSD", id="analyze-psd")], style={"padding": "5px"}),
    # Periodic-Aperiodic decomposition
    dbc.Row([dbc.Button("Periodic-Aperiodic PSD decomposition", id="analyze-fooof")], style={"padding": "5px"}),
    # Detrend PSD
    dbc.Row([dbc.Button("Detrend PSD", id="analyze-dpsd")], style={"padding": "5px"}),
    # Spectrogram
    dbc.Row([dbc.Button("Spectrogram", id="analyze-spec")], style={"padding": "5px"}),
], title="Spectral Analysis")

activity = dbc.AccordionItem([
    # Detect Peaks by MAD
    dbc.Row([
        html.Strong("Detect peaks by MAD"),
        dbc.Col(html.H6("Mean absolute deviation threshold:"), width="auto"),
        dbc.Input(placeholder="6", id="analyze-peaks-ampl-thresh"),
        dbc.Button("Start", id="analyze-peaks-ampl")    
     ], style={"padding": "25px"}),
    # Detect Bursts
    # Detect Events
    dbc.Row([
        html.Strong("Detect event by moving deviation measures"),
        dbc.Row(dbc.RadioItems(id="analyze-events-method", value=1, inline=True, 
                               options=[{"label": "Moving Std", "value": 1}, {"label": "Moving MAD", "value": 2}])),
        dbc.Row([
            dbc.Col(html.H6("Threshold factor")),
            dbc.Col(dbc.Input(placeholder="1", id="analyze-events-thresh")),
        ]),
        dbc.Button("Start", id="analyze-events"),
        dbc.Row([], id="analyze-events-stats"),
    ], style={"padding": "25px"}),
], title="Activity Detection")


network = dbc.AccordionItem([
    # Cross-Correlation
    dbc.Row([dbc.Button("Cross-Correlation", id="analyze-xcorr")], style={"padding": "5px"}),
    # Mutual Info
    dbc.Row([dbc.Button("Mutual Information", id="analyze-mi")], style={"padding": "5px"}),
    # Transfer Entropy
    dbc.Row([dbc.Button("Transfer Entropy", id="analyze-te")], style={"padding": "5px"}),
    # Coherence
    dbc.Row([dbc.Button("Coherence", id="analyze-coh")], style={"padding": "5px"}),
    # Granger Causality
    dbc.Row([dbc.Button("Granger Causality", id="analyze-gc")], style={"padding": "5px"}),
    # Spectral Granger Causality
    dbc.Row([dbc.Button("Spectral Granger Causality", id="analyze-sgc")], style={"padding": "5px"}),
], title="Network Analysis")


visualize = dbc.AccordionItem([
    # Scatter, box, violin
    # single values on grid
    # single values on grid over time
    # time series
    dbc.Row([
        dbc.Row([
            dbc.Label("Choose a bunch"),
            dbc.Checklist(
                options=[
                    {"label": "Signals", "value": TimeSeriesPlottable.SIG.value},
                    {"label": "Envelope", "value": TimeSeriesPlottable.ENV.value},
                    {"label": "Derivative", "value": TimeSeriesPlottable.DERV.value},
                    {"label": "Moving Average", "value": TimeSeriesPlottable.MV_AVG.value},
                    {"label": "Moving Mean Absolute Deviation", "value": TimeSeriesPlottable.MV_MAD.value},
                    {"label": "Peaks", "value": TimeSeriesPlottable.PEAKS.value},
                    {"label": "Bursts", "value": TimeSeriesPlottable.BURSTS.value},
                    {"label": "Seizure", "value": TimeSeriesPlottable.SEIZURE.value},
                ],
                value=[TimeSeriesPlottable.SIG.value],
                id="analyze-plot-ts-input",),
            ]),
        dbc.Row([dbc.Button("Plot Signals", id="analyze-plot-ts")], style={"padding": "5px"}),
    ]),
    # time series with peak scatter
    # time series with bursts overlay
    # time series with seizure overlay
    # PSD & detrended PSD
    # FOOOF
    # Spectrogram
    # Network relation (w. networkx
    # Coherence
    # Current Source Densities
    html.Div(id='analyze-output-dummy', style={'display': 'none'})
], title="Visualize")

export = dbc.AccordionItem([
    # events, bursts, stats, 

    # event stats 
    dbc.Row([
        dbc.Col(dbc.Checklist(options=[{"label": "Export", "value": 1}], value=[], id="analyze-events-export")),
        dbc.Col(dbc.Input(placeholder="Enter full file path for file to export", id="analyze-events-fname")),
    ], style={"padding": "25px"}),

    
    # animations
    dbc.Row([html.H6("Electrode Amplitude Animation"),
        dbc.Input(placeholder="Column (only 1D signals)", id="analyze-animate-value"),
        dbc.Input(placeholder="Playback Speed in FPS", id="analyze-animate-fps"),
        dbc.Input(placeholder="Slow down from real time", id="analyze-animate-slow-down"),
        dbc.Button("Generate Video (takes some minutes)", class_name="fas fa-play", id="analyze-animate-play", n_clicks=0)
     ]),
], title="Export")

compute = dbc.AccordionItem([dbc.Accordion([basics, spectral, activity, network])], title="Compute")

side_bar = dbc.Col([dbc.Accordion([filters, compute, visualize, export], always_open=True)], width=4)

analyze = dbc.Container([dbc.Row([side_bar, result_tables], align='start', justify='start')])
