"""
Dash-based HTML code for the home/import ui to be displayed in the browser via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc

importer = dbc.Container(
        [
            dbc.Row([dbc.Col([html.H6("Condition Input File Path: ")], width="auto",
                             align="center"),
                     dbc.Col(dbc.Input(id="import-cond-input-file-path", type="text",
                                       size="85", required=True,
                                       placeholder='Please enter the absolute'
                                       'file path!'),
                             width="auto")],
                    align="center",
                    justify="center",
                    style={"padding": "5px"}
                    ),
            dbc.Row([dbc.Col([html.H6("Baseline Input File Path: ")], width="auto",
                             align="center"),
                     dbc.Col(dbc.Input(id="import-base-input-file-path", type="text",
                                       size="85", required=True,
                                       placeholder='Please enter the absolute'
                                       'file path!'),
                             width="auto")],
                    align="center",
                    justify="center",
                    style={"padding": "5px"}
                    ),
            dbc.Row([dbc.Col(html.H6("File type:"), width="auto"),
                     dbc.Col(dbc.RadioItems(id="import-radios", value=0,
                                            inline=True,
                                            options=[{"label": "MCS 256 MEA",
                                                      "value": 0},
                                                     {"label": "MCS CMOS MEA",
                                                      "value": 1},
                                                     {"label": "MCS Multi-Well MEA",
                                                      "value": 2},
                                                     ],
                                            ), width="auto")],
                    align="center",
                    justify="center",
                    style={"padding": "5px"}
                    ),
            dbc.Row([dbc.Col(dbc.Button("Submit",
                                        id="import-submit-input-file-path",
                                        n_clicks=0),
                             width="auto")],
                    align="center",
                    justify="center",
                    style={"padding": "5px"}
                    ),
            dbc.Container(id="import-feedback")
            ],
        style={"padding": "50px"}
        )

home = html.Div(
        dbc.Container(
            [

                dbc.Row(
                    [dbc.Col(html.Img(src="/assets/icon.png", height="75px"),
                             width="auto"),
                     dbc.Col(html.H1("LFP Toolbox", className="display-3"),
                             width="auto")]
                     ),
                html.P(
                    "Please Upload a file",
                    className="lead",
                    ),
                html.Hr(className="my-2"),
                importer],
            fluid=True,
            className="py-3",
            ),
        className="p-3 bg-light rounded-3",
        )


def build_import_infos(infos, success):
    """
    Creates a next button on success and displays the preformatted string info \
            in every case.

        @param info: preformatted string containing either metadata on the \
                data set or an error message
        @param success: a boolean indicating if the import was successfull \
                or not.

        @return: HTML to display the preformatted string and a next button \
                if the import was successful.
    """
    next_button = html.A(dbc.Row([
        dbc.Col([
            dbc.Button("Next", n_clicks=0)
            ], width="auto")
        ],
                                 align="center",
                                 justify="center",
                                 style={"padding": "20px"}),
                         href="/select") if success else None

    return [dbc.Row([dbc.Col([html.Pre(infos)],
                             width="auto")],
                    align="center",
                    justify="center",
                    style={"padding": "20px"}),
            next_button]
