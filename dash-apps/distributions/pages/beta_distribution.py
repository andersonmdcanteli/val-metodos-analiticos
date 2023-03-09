### ------ IMPORTS ------ ###
# import base64
import numbers
# import io

# --- dash --- #
from dash import callback, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.dash import no_update
import dash_mantine_components as dmc
# --- Third part --- #
import pandas as pd
import numpy as np
from scipy.stats import beta
from pandas.api.types import is_numeric_dtype

# --- Home Made --- #
from functions import distributions_functions as distfuncs

### ------ datasets ------ ####





### ------ Constantes ------ ###
PERSISTENCE_TYPE = "session"
PERSISTENCE = True


### ------ FUNCTIONS ------ ###



### ------ Configs ------ ###
configuracoes_grafico = {
    'staticPlot': False,     # True, False
    'scrollZoom': True,      # True, False
    'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
    'showTips': True,       # True, False
    'displayModeBar': True,  # True, False, 'hover'
    'watermark': True,
    'modeBarButtonsToRemove': ['lasso2d'],
}

style_data = {
    'whiteSpace': 'normal',
    'height': 'auto'
    }

style_cell = {
    'font_size': 'clamp(1rem, 0.5vw, 0.5vh)',
    'textAlign': 'center',
    'height': 'auto',
    'minWidth': '80px',
    'width': '80px',
    'maxWidth': '80px',
    'whiteSpace': 'normal'
}

style_table = {
    'overflowX': 'auto',
    'overflowY': 'auto',
    'maxHeight': '500px',
    }

style_header = {
    'fontWeight': 'bold', # deixando em negrito
    }


# --- FOR DROPDOWNs --- #



### ------ LAYOUTS ------ ###


# --- MAIN LAYOUT --- #
layout = html.Div([

    dbc.Row([
        dbc.Col(
            sm = 0,
            lg = 1,
            align="center"
        ),
        dbc.Col(
            html.H2("Distribuição Beta"),
            width = {"size": 10},
            align = "center"
        ),
        dbc.Col(
            sm = 0,
            lg = 1,
            align="center"
        ),
    ], style = {'textAlign': 'center', 'paddingTop': '30px', 'paddingBottom': '30px'},),
    dbc.Row([
        # Grafico
        dbc.Col([
            # linha para o gráfico
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id="betadist-plot", mathjax=True, config=configuracoes_grafico),

                )
            ),
            # linha
            dbc.Row(
                dbc.Col(
                    html.H4("Limites do gráfico:"),
                ), style={"paddingTop": "25px", "paddingBottom": "25px"}
            ),
            # linha para opções
            dbc.Row([
                # Opções de limites do gráfico
                dbc.Col(
                    dbc.Row([
                        dbc.Col(
                            html.Label("x min:"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-xmin-input", type="number", value=-.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),
                        dbc.Col(
                            html.Label("x max:"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-xmax-input", type="number", value=1.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),
                        dbc.Col(
                            html.Label("y min:"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-ymin-input", type="number", value=-.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),
                        dbc.Col(
                            html.Label("y max:"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-ymax-input", type="number", value=20, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),
                        ],
                    ),
                    lg = 10
                ),
            ], justify="center",),


            dbc.Row([
                dbc.Col(
                    html.H5("Amostra:"), width="auto", align="center",
                ),
                dbc.Col(
                    dmc.Tooltip(
                        label="Adiciona ou remove os dados amostrais do gráfico",
                        multiline=True, width=200,
                        position="top", transition="slide-left", transitionDuration=300,
                        offset=3,
                        children=dmc.Switch(label="", onLabel="Sim", offLabel="Não", checked=True, size="lg", color="teal",
                                        id="betadist-switch", persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE),
                        color="gray", withArrow=True,
                    ), width="auto", align="center", style={"textAlign": "center"}
                ),
            ], style={"paddingTop": "15px", "paddingBottom": "15px"}),


        ],lg = 8),
        # Inputs para os dados
        dbc.Col([

            dbc.Row([
                dbc.Col(
                    html.H4(
                        dcc.Markdown(
                            "$f \\left(x \\right) = \\frac{\\left(x-a \\right)^{p-1}\\left(b-x\\right)^{q-1}}{B\\left(p,q\\right) \\left(b-a\\right)^{p+q-1}}$",
                            mathjax=True
                            )
                    ), style={"textAlign": "center"}
                ),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Row(
                        dbc.Col(
                            html.Label("onde:")
                        ),
                    ),
                    dbc.Row(
                        dbc.Col([
                            html.Ul([
                                html.Li(
                                    dcc.Markdown(
                                        "$a$ é o limite inferior;",
                                        mathjax=True
                                        )
                                ),
                                html.Li(
                                    dcc.Markdown(
                                        "$b$ é o limite superior;",
                                        mathjax=True
                                        )
                                ),
                                html.Li(
                                    dcc.Markdown(
                                        "$p$ e $q$ são parâmetros de forma;",
                                        mathjax=True
                                        )
                                ),
                                html.Li(
                                    dcc.Markdown(
                                        "$B \\left(p, q \\right) = \\int_{0}^{1} {t^{p-1} \\left(1-t \\right)^{q-1}dt}$",
                                        mathjax=True
                                        )
                                ),
                            ])
                        ])
                    ),
                ], lg=8),
                dbc.Col([
                    dbc.Row(
                        dbc.Col(
                            html.Label("para:")
                        ),
                    ),
                    dbc.Row(
                        dbc.Col([
                            html.Ul([
                                html.Li(
                                    dcc.Markdown(
                                        "$a \\le x \\le b$;",
                                        mathjax=True
                                        )
                                ),
                                html.Li(
                                    dcc.Markdown(
                                        "$q > 0$;",
                                        mathjax=True
                                        )
                                ),
                                html.Li(
                                    dcc.Markdown(
                                        "$p > 0$;",
                                        mathjax=True
                                        )
                                ),
                            ])
                        ])
                    ),
                ],),
            ]),





            dbc.Row(
                dbc.Col(
                    html.H5("Parâmetros da distribuição")
                ), style={"textAlign": "center", "paddingTop": "25px"}
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Row([
                        dbc.Col(
                            dcc.Markdown(
                                "$a =$",
                                mathjax=True
                                ),
                            width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-loc-input", type="number", value=0, step=.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),
                        dbc.Col(
                            dcc.Markdown(
                                "$b =$",
                                mathjax=True
                                ),
                            width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-scale-input", type="number", value=1, min=0, step=.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),



                        ],
                    ),
                ),
            ], justify="center",),
            dbc.Row([
                dbc.Col(
                    dbc.Row([
                        dbc.Col(
                            dcc.Markdown(
                                "$p =$",
                                mathjax=True
                                ),
                            width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-shape-a-input", type="number", value=.5, min=0, step=.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),
                        dbc.Col(
                            dcc.Markdown(
                                "$q =$",
                                mathjax=True
                                ),
                            width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dbc.Input(id="betadist-shape-b-input", type="number", value=.5, min=0, step=.1, autofocus=False, debounce=False,
                                    persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                    )
                                ),
                            ]),

                        ],
                    ),
                ),
            ], justify="center",),
            html.Div([
                dbc.Row(
                    dbc.Col(
                        html.H5("Parâmetros da amostra")
                    ), style={"textAlign": "center", "paddingTop": "25px", "paddingBottom": "15px"}
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Row([
                            dbc.Col(
                                html.Label("Tamanho:"), width="auto", align="center"
                            ),
                            dbc.Col([
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Input(id="betadist-size-input", type="number", value=20, step=1, min=0, autofocus=False, debounce=False,
                                        persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                        )
                                    ),
                                ]),
                            dbc.Col(
                                html.Label("Aleatório"), width="auto", align="center"
                            ),
                            dbc.Col([
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Input(id="betadist-random-input", type="number", value=42, step=1, min=0, autofocus=False, debounce=False,
                                        persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE)
                                        )
                                    ),
                                ]),

                            ],
                        ),
                        lg = 10
                    ),
                ], justify="center",),
            ], id="betadist-param-div"),

            # alert
            dbc.Row(
                dbc.Col(
                    id="betadist-alert"
                )
            )
        ])
    ], justify="center",),

    html.Hr(style={"paddingTop": "25px", "paddingBottom": "25px"}),
    dbc.Row(
        dbc.Col(
            html.H5("Referências")
        )
    ),
    dbc.Row(
        dbc.Col(
            html.Ul([
                html.Li([
                    html.Span("NIST/SEMATECH e-Handbook of Statistical Methods (2012), "),
                    html.A("http://www.itl.nist.gov", href="https://doi.org/10.18434/M32189"),
                    html.Span(", "),
                    html.A("Beta Distribution", href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm"),
                    html.Span(". Acesso em 02/02/2023.")
                ]),
                html.Li([
                    html.Span("Virtanen, P., Gommers, R., Oliphant, T.E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17, 261–272 (2020). DOI: "),
                    html.A("10.1038/s41592-019-0686-2", href="https://doi.org/10.1038/s41592-019-0686-2"),
                    html.Span(", "),
                    html.A("scipy.stats.beta", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html"),
                    html.Span(". Acesso em 02/02/2023.")
                ])
            ])
        )
    ),



],)




### ------ CALLBACKS ------ ###


@callback(
    Output(component_id="betadist-param-div", component_property="style"),
    Input(component_id="betadist-switch", component_property="checked"),
)
def toggle_sample(switch):
    if switch:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@callback(
    Output(component_id="betadist-plot", component_property="figure"),
    Output(component_id="betadist-alert", component_property="children"),
    Input(component_id="betadist-xmin-input", component_property="value"),
    Input(component_id="betadist-xmax-input", component_property="value"),
    Input(component_id="betadist-ymin-input", component_property="value"),
    Input(component_id="betadist-ymax-input", component_property="value"),
    Input(component_id="betadist-loc-input", component_property="value"),
    Input(component_id="betadist-scale-input", component_property="value"),
    Input(component_id="betadist-shape-a-input", component_property="value"),
    Input(component_id="betadist-shape-b-input", component_property="value"),
    Input(component_id="betadist-size-input", component_property="value"),
    Input(component_id="betadist-random-input", component_property="value"),
    Input(component_id="betadist-switch", component_property="checked"),
)
def update_plot(x_min, x_max, y_min, y_max, loc, scale, shape_a, shape_b, size, random, switch):


    if not isinstance(x_min, numbers.Number):
        text = "O valor do limite inferior do eixo x deve ser numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert

    if not isinstance(x_max, numbers.Number):
        text = "O valor do limite superior do eixo x deve ser numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert

    if x_min >= x_max:
        text = "O valor do limite superior deve ser maior do que o limite inferior"
        alert = distfuncs.make_alert(text)
        return no_update, alert

    if not isinstance(y_min, numbers.Number):
        text = "O valor do limite inferior do eixo y deve ser numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert

    if not isinstance(y_max, numbers.Number):
        text = "O valor do limite superior do eixo y deve ser numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert

    if y_min >= y_max:
        text = "O valor do limite superior deve ser maior do que o limite inferior"
        alert = distfuncs.make_alert(text)
        return no_update, alert


    if not isinstance(loc, numbers.Number):
        text = "O limite inferior (a) ser um valor numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert

    if not isinstance(scale, numbers.Number):
        text = "O limite superior (b) deve ser um valor numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert

    if scale <= 0:
        text = "O limite superior (b) deve ser maior do que zero (0)"
        alert = distfuncs.make_alert(text)
        return no_update, alert

    if scale <= loc:
        text = "O limite superior (b) deve ser maior do que o limite inferior (a)"
        alert = distfuncs.make_alert(text)
        return no_update, alert

    if not isinstance(shape_a, numbers.Number):
        text = "O parâmetro de forma (p) deve ser um valor numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert


    if shape_b <= 0:
        text = "O parâmetro de forma (b) deve ser maior do que zero (0)"
        alert = distfuncs.make_alert(text)
        return no_update, alert

    if not isinstance(shape_b, numbers.Number):
        text = "O parâmetro de forma (b) deve ser um valor numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert


    if shape_a <= 0:
        text = "O parâmetro de forma (p) deve ser maior do que zero (0)"
        alert = distfuncs.make_alert(text)
        return no_update, alert


    if not isinstance(size, numbers.Number):
        text = "O tamanho amostral deve ser um valor numérico!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert


    if size < 1:
        text = "A amostra deve ter tamanho maior do que zero (0)."
        alert = distfuncs.make_alert(text)
        return no_update, alert

    if not isinstance(random, numbers.Number):
        text = "A chave para gerar amostras aleatórias deve ser numérica!"
        dica = html.Div([
            html.P("O separador de casas decimais é definido pela sua região (seu país)."),
        ])
        alert = distfuncs.make_alert(text, dica)
        return no_update, alert


    if random < 0:
        text = "A chave para gerar amostras aleatórias deve ser positiva."
        alert = distfuncs.make_alert(text)
        return no_update, alert

    height = 500


    x = np.linspace(beta.ppf(0.001, a=shape_a, b=shape_a, loc=loc, scale=scale), beta.ppf(0.999, a=shape_a, b=shape_b, loc=loc, scale=scale), 1000)

    df_distribution = pd.DataFrame({
        "x": x,
        "Densidade": beta.pdf(x, a=shape_a, b=shape_b, loc=loc, scale=scale)
    })

    try:
        fig = px.line(df_distribution, x=df_distribution.columns[0], y=df_distribution.columns[1], height=height)
    except Exception:
        fig = px.line(df_distribution, x=df_distribution.columns[0], y=df_distribution.columns[1], height=height)

    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name'] = 'Dist. Beta'
    fig['data'][0]['line']['color']='black'

    if switch:
        x_data = beta.rvs(a=shape_a, b=shape_b, loc=loc, scale=scale, size=size, random_state=random)
        df_data = pd.DataFrame({
            "x": x_data,
            "Dados":  beta.pdf(x_data, a=shape_a, b=shape_b, loc=loc, scale=scale)
        })
        # adicionando a reta
        fig.add_trace(
            go.Scatter(
                x = df_data[df_data.columns[0]],
                y = df_data[df_data.columns[1]],
                mode = 'markers',
                marker=dict(size=12, color="rgba(255, 0, 0, 0.5)"),
                name = "Amostra",
                hovertemplate =
                    "x = %{x:.3f}<br>" +
                    "Densidade = %{y:.3f}<br>" +
                    "<extra></extra>", # removendo a caixa que incomoda a visualização),
                )
            )
        fig.add_trace(
            go.Histogram(
                x=df_data[df_data.columns[0]],
                histnorm='probability density',
                name="Histograma amostral",
                # marker_color="rgba(205, 209, 228, 0.5)",
                marker_color="rgba(11, 127, 171, 0.5)",
                hovertemplate =
                    "x = %{x}<br>" +
                    "Densidade = %{y}<br>" +
                    "<extra></extra>", # removendo a caixa que incomoda a visualização),

                ),
            )

    fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="dimgray")
    fig.add_hline(y=0, line_width=2, line_dash="dot", line_color="dimgray")

    # ajustando o layout
    fig.update_layout(
                    template='simple_white',
                    hoverlabel = dict(
                        font_size = 16,
                        font_family = "Rockwell"
                    ),
                    xaxis_range=[x_min, x_max],
                    yaxis_range=[y_min, y_max],
                    legend = dict(
                        font_size = 12,
                        font_family = "Rockwell",
                    ),
                    margin={"r":0,"l":0,"b":0, 't':30}, # removendo margens desnecessárias
                )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig, []





#
