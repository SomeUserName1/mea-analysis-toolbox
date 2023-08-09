"""
Dash-based HTML code for the navbar of the ui to be displayed in the browser \
        via the Dash server.
"""
from dash import html
import dash_bootstrap_components as dbc

nav_items = [
        dbc.NavItem(dbc.NavLink("Select", href="/select", active="exact")),
        dbc.NavItem(dbc.NavLink("Analyze", href="/analyze", active="exact")),
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
