from dash import Dash, dcc, html, Input, Output, callback
from pages import home, about, page_not_found, footer
from pages import normal_distribution, exponential_distribution, uniform_distribution, t_distribution, gamma_distribution, beta_distribution
from pages import laplace_distribution, chi2_distribution
from datetime import date
import dash_bootstrap_components as dbc
from datasets import frases


FA = "https://use.fontawesome.com/releases/v6.0.0/css/all.css"

external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.themes.GRID, FA]


app = Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets = external_stylesheets,
                # external_scripts = mathjax,
                meta_tags = [{
                        'name': 'viewport',
                        'content': 'width=device_width',
                        'initial-scale': 1.0
                }])

server = app.server

## Navbar Simples
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Normal", href="/normal-distribution")),
        dbc.NavItem(dbc.NavLink("Gamma", href="/gamma-distribution")),
        dbc.NavItem(dbc.NavLink("Chi2", href="/chi2-distribution")),        
        dbc.NavItem(dbc.NavLink("Laplace", href="/laplace-distribution")),
        dbc.NavItem(dbc.NavLink("Beta", href="/beta-distribution")),
        dbc.NavItem(dbc.NavLink("Exponencial", href="/exponential-distribution")),
        dbc.NavItem(dbc.NavLink("Uniform", href="/uniform-distribution")),
        dbc.NavItem(dbc.NavLink("t-Student", href="/t-student-distribution")),
        dbc.NavItem(dbc.NavLink(
                        html.I(className="fas fa-mug-hot", style={"color": 'yellow'}),
                        href="/about")),
            ],
    brand="Home",
    brand_href="/",
    color="darkgreen",
    dark=True,
)



## Criando o layout
app.layout = html.Div([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(
        id='page-content',
        style={'margin-right': '2.5%', 'margin-left': '2.5%', 'margin-top': '10px', }
        ),
    footer.footer,
], style={"backgroundColor": "ivory"})


@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == "/normal-distribution":
        return normal_distribution.layout
    elif pathname == "/exponential-distribution":
        return exponential_distribution.layout
    elif pathname == "/uniform-distribution":
        return uniform_distribution.layout
    elif pathname == "/t-student-distribution":
        return t_distribution.layout
    elif pathname == "/gamma-distribution":
        return gamma_distribution.layout
    elif pathname == "/beta-distribution":
        return beta_distribution.layout
    elif pathname == "/laplace-distribution":
        return laplace_distribution.layout
    elif pathname == "/chi2-distribution":
        return chi2_distribution.layout
    # elif pathname == "/chi2-distribution":
    #     return chi2_distribution.layout

    elif pathname == '/home':
        return home.layout
    elif pathname == '/about':
        return about.layout
    else:
        return page_not_found.layout

if __name__ == '__main__':
    app.run_server(debug=True)
