"""
Dash-based HTML code for the navbar of the ui to be displayed in the browser \
        via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc

nav_items = [
        dbc.NavItem(dbc.NavLink("Select", href="/select", active="exact")),
        dbc.NavItem(dbc.NavLink("Preprocess", href="/preproc", active="exact")),
        dbc.NavItem(dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem(dbc.NavLink("RMS per Electrode Animation",
                        href="/rms-animate", active="exact")),
                dbc.DropdownMenuItem(dbc.NavLink("FFT", href="/fft",
                                                 active="exact")),
                dbc.DropdownMenuItem(
                 dbc.NavLink("Granger Causality", href="/granger",
                             active="exact"))
            ],
            label="Explore & Analyze",
            nav=True,
            in_navbar=True,
            menu_variant="dark"),
        ),
    ]


navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [dbc.Col(html.Img(src="/assets/icon.png", height="30px")),
                     dbc.Col(dbc.NavbarBrand("LFP Toolbox", className="ms-2"))]
                ),
                href="/"
            ),
            dbc.Nav(id="nav")
        ],
        fluid=True
    ),
    color="dark", dark=True
)
