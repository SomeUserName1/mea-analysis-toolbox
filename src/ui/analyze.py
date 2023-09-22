"""
Dash-based HTML code for the analyze ui to be displayed in the browser via the
Dash server.
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
    """
    Enum class for the different plottable time series.
    """
    SIG = 0
    PEAKS = 1
    EVENTS = 2
    THRESH = 3


def prev_next_rows_buttons(id_string):
    """
    Generate a row with two buttons to go to the next or previous page of a
    table.

    :param id_string: The id string to use for the buttons.
    :type id_string: str

    :return: The row with the buttons.
    :rtype: dash_bootstrap_components.Row
    """
    return dbc.Row([dbc.Col([
            dbc.Button("Prev", n_clicks=0, id=f"{id_string}-prev"),
            dbc.Button("Next", n_clicks=0, id=f"{id_string}-next")
            ], width="auto")],
        align="center", justify="center", style={"padding": "20px"})


channels_table = dbc.Card(
    dbc.CardBody([
        html.Table([], id="channels-table",
                   className=('table table-bordered table-hover '
                              'table-responsive')),
        prev_next_rows_buttons("channels-table")
        ])
)

peaks_table = dbc.Card(
    dbc.CardBody([
        html.Table([], id="peaks-table",
                   className='table table-bordered table-hover '
                             'table-responsive'),
        prev_next_rows_buttons("peaks-table")
        ])

)

events_table = dbc.Card(
    dbc.CardBody([
        html.Table([], id="events-table",
                   className='table table-bordered table-hover '
                             'table-responsive'),
        prev_next_rows_buttons("events-table")
        ])

)

# network_table = dbc.Card(
#     dbc.CardBody([], id="network-table")
# )


result_tables = dbc.Col(dbc.Tabs([
    dbc.Tab(channels_table, label="Channels"),
    dbc.Tab(peaks_table, label="Peaks"),
    dbc.Tab(events_table, label="Events"),
    # dbc.Tab(network_table, label="Network")
    ]), width='auto')


def generate_table(dataframe, from_row=0, max_rows=100):
    """
    Generate a HTML table from a pandas dataframe.

    :param dataframe: The dataframe to generate the table from.
    :type dataframe: pandas.DataFrame

    :param from_row: The row to start from.
    :type from_row: int

    :param to_row: The row to end at.
    :type to_row: int

    :param max_rows: The maximum number of rows to display.

    :return: The HTML table.
    :rtype: dash_html_components.Table
    """
    rows = []
    for i in range(from_row, min(len(dataframe), from_row + max_rows)):
        cols = []
        for col in dataframe.columns:
            field = dataframe.iloc[i][col]
            if isinstance(field, float):
                cols.append(html.Td(f"{dataframe.iloc[i][col]:.4e}"))
            elif isinstance(field, np.ndarray):
                cols.append(html.Td(f"shape: {dataframe.iloc[i][col].shape}"))
            else:
                cols.append(html.Td(f"{dataframe.iloc[i][col]}"))

        rows.append(html.Tr(cols))

    return [
            html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
            html.Tbody(rows)
            ]


filters = dbc.AccordionItem([
    # Rereference (to remove noise across all channels)
    # Line Noise
    dbc.Row([
        dbc.Row([dbc.Button("Remove Line Noise (EU)",
                            id="analyze-linenoise-apply")]),
        dbc.Row([], id="analyze-linenoise-result"),
    ], style={"padding": "25px"}, class_name="border rounded-3"),
    # Custom frequencies
    dbc.Row([
        dbc.Row([html.H6("Custom Frequency Filter")]),
        dbc.Row([html.Hr(className="my-2")]),
        dbc.Row([
            dbc.Col([dbc.Input(placeholder="Lower Frequency",
                               id="analyze-fltr-lower")]),
            dbc.Col([dbc.Input(placeholder="Upper Frequency",
                               id="analyze-fltr-upper")])
            ]),
        dbc.RadioItems(id="analyze-fltr-type",
                       options=[{"label": "Bandpass", "value": 0},
                                {"label": "Bandstop", "value": 1}],
                       labelCheckedClassName="text-success",
                       inputCheckedClassName=("border rounded-3"),
                       style={"padding": "25px"}),

        dbc.Row([dbc.Button("Apply", id="analyze-fltr-apply")]),
        dbc.Row([], id="analyze-fltr-result"),
        ], style={"padding": "25px"}, class_name="border rounded-3"),
    # Downsample
    dbc.Row([
        dbc.Col([dbc.Input(placeholder="New sampling rate",
                           id="analyze-dwnsmpl-rate")]),
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
    dbc.Row([dbc.Button("Approximate Entropy", id="analyze-ent")],
            style={"padding": "5px"}),
], title="Basic Properties")

spectral = dbc.AccordionItem([
    # PSD
    dbc.Row([dbc.Button("PSD", id="analyze-psd")], style={"padding": "5px"}),
    # Spectrogram
    dbc.Row([dbc.Button("Spectrogram", id="analyze-spec")],
            style={"padding": "5px"}),
    #  # Periodic-Aperiodic decomposition
    #  dbc.Row([dbc.Button("Periodic-Aperiodic PSD decomposition",
    #                      id="analyze-fooof")],
    #          style={"padding": "5px"}),
    #  # Detrend PSD
    #  dbc.Row([dbc.Button("Detrend PSD", id="analyze-dpsd")],
    #          style={"padding": "5px"}),

], title="Spectral Analysis")

activity = dbc.AccordionItem([
    # Detect Peaks by MAD
    dbc.Row([
        html.Strong("Detect Activity"),
        dbc.Row([dbc.Row([html.H6("Presets:")])]),
        dbc.Row([
            dbc.RadioItems(
                id="analyze-activity-presets",
                options=[{"label": "Peaks", "value": 0},
                         {"label": "Bursts", "value": 1}],
                labelCheckedClassName="text-success",
                inputCheckedClassName=("border rounded-3"),
                style={"padding": "25px"}),

            ]),
        dbc.Col(html.H6("Parameters:"), width="auto"),
        dbc.Input(placeholder="mad window (50ms)", id="analyze-peaks-mad-win"),
        dbc.Input(placeholder="envelope window (100ms)",
                  id="analyze-peaks-env-win"),
        dbc.Input(placeholder="envelope percentile (5)",
                  id="analyze-peaks-env-percentile"),
        dbc.Input(placeholder="mad treshold (1.5)",
                  id="analyze-peaks-mad-thrsh"),
        dbc.Input(placeholder="envelope treshold (2)",
                  id="analyze-peaks-env-thrsh"),
        dbc.Button("Start", id="analyze-peaks-ampl")
     ], style={"padding": "25px"}),
    dbc.Row([
        html.Strong("Detect Events"),
        dbc.Row([dbc.Row([html.H6("Presets:")])]),
        dbc.Col(html.H6("Parameters:"), width="auto"),
        dbc.Input(placeholder="mad window (50ms)",
                  id="analyze-events-mad-win"),
        dbc.Input(placeholder="envelope window (100ms)",
                  id="analyze-events-env-win"),
        dbc.Input(placeholder="envelope percentile (5)",
                  id="analyze-events-env-percentile"),
        dbc.Input(placeholder="mad treshold (1.5)",
                  id="analyze-events-mad-thrsh"),

        dbc.Button("Start", id="analyze-events")
     ], style={"padding": "25px"}),
], title="Activity Detection")


# network = dbc.AccordionItem([
#     # Cross-Correlation
#     dbc.Row([dbc.Button("Cross-Correlation", id="analyze-xcorr")],
#             style={"padding": "5px"}),
#     # Mutual Info
#     dbc.Row([dbc.Button("Mutual Information", id="analyze-mi")],
#             style={"padding": "5px"}),
#     # Transfer Entropy
#     dbc.Row([dbc.Button("Transfer Entropy", id="analyze-te")],
#             style={"padding": "5px"}),
#     # Coherence
#     dbc.Row([dbc.Button("Coherence", id="analyze-coh")],
#             style={"padding": "5px"}),
#     # Granger Causality
#     dbc.Row([dbc.Button("Granger Causality", id="analyze-gc")],
#             style={"padding": "5px"}),
#     # Spectral Granger Causality
#     dbc.Row([dbc.Button("Spectral Granger Causality", id="analyze-sgc")],
#             style={"padding": "5px"}),
# ], title="Network Analysis")


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
                    {"label": "Signals",
                     "value": TimeSeriesPlottable.SIG.value},
                    {"label": "Peaks",
                     "value": TimeSeriesPlottable.PEAKS.value},
                    {"label": "Events",
                     "value": TimeSeriesPlottable.EVENTS.value},
                    {"label": "Detection Thresholds",
                     "value": TimeSeriesPlottable.THRESH.value},
                ],
                value=[TimeSeriesPlottable.SIG.value],
                id="analyze-plot-ts-input",),
            ]),
        dbc.Row([dbc.Button("Plot Signals", id="analyze-plot-ts")],
                style={"padding": "5px"}),
    ]),
    dbc.Row([dbc.Button("PSD", id="analyze-plot-psds")],
            style={"padding": "5px"}),
    dbc.Row([dbc.Button("Spectrograms", id="analyze-plot-spects")],
            style={"padding": "5px"}),

    # Network relation (w. networkx
    # Coherence
    # Current Source Densities
    html.Div(id='analyze-output-dummy', style={'display': 'none'})
], title="Visualize")

export = dbc.AccordionItem([
    # events, bursts, stats,

    # event stats
    dbc.Row([
        dbc.Col(dbc.Input(placeholder="Enter full file path and base name",
                          id="analyze-export-fname")),
        dbc.Col(dbc.Button("Export Tables", id="analyze-export-tables")),
        dbc.Row([], id="analyze-export-feedback"),
    ], style={"padding": "25px"}),
    # # animations
    # dbc.Row([html.H6("Electrode Amplitude Animation"),
    #         dbc.Input(placeholder="Column (only 1D signals)",
    #                   id="analyze-animate-value"),
    #         dbc.Input(placeholder="Playback Speed in FPS",
    #                   id="analyze-animate-fps"),
    #         dbc.Input(placeholder="Slow down from real time",
    #                   id="analyze-animate-slow-down"),
    #         dbc.Button("Generate Video (takes some minutes)",
    #                    class_name="fas fa-play",
    #                    id="analyze-animate-play",
    #                    n_clicks=0)]
    #         ),
], title="Export")

compute = dbc.AccordionItem([dbc.Accordion([basics,
                                            spectral,
                                            activity,
                                            # network
                                            ])],
                            title="Compute")

side_bar = dbc.Col([dbc.Accordion([filters, compute, visualize, export],
                                  always_open=True)], width=4)

analyze = dbc.Container([dbc.Row([side_bar, result_tables],
                                 align='start', justify='start')])
