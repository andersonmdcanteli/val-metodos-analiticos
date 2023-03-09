from dash import html
import dash_bootstrap_components as dbc






layout = html.Div([
    dbc.Row(
        dbc.Col(
            html.H2("Home page"),
        ), style={"text-align": 'center'}, className="title_spacing",
    ),
    dbc.Row(
        dbc.Col(
            html.H4("Distribuições:")
        ),
        style={"paddingTop": "100px"}
    ),
    dbc.Row(
        dbc.Col([
            html.Ul([
                html.Li([
                    html.A("Distribuição Normal", href="/normal-distribution"),
                    html.Span(";")
                ]),
                html.Li([
                    html.A("Distribuição t de Student", href="/t-student-distribution"),
                    html.Span(";")
                ]),
                html.Li([
                    html.A("Distribuição Exponencial", href="/exponential-distribution"),
                    html.Span(";")
                ]),
                html.Li([
                    html.A("Distribuição Uniforme", href="/uniform-distribution"),
                    html.Span(";")
                ]),
                html.Li([
                    html.A("Distribuição Gamma", href="/gamma-distribution"),
                    html.Span(";")
                ]),
                html.Li([
                    html.A("Distribuição Beta", href="/beta-distribution"),
                    html.Span(";")
                ]),
                html.Li([
                    html.A("Distribuição Laplace", href="/laplace-distribution"),
                    html.Span(";")
                ]),
            ])
        ])
    ),
    dbc.Row(
        style={"paddingTop": "300px"}
    )

])
