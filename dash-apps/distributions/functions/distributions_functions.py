### ------ IMPORTS ------ ###
import base64
import io


# --- Third part --- #
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html
### ------ Constants ------ ###




### ------ Functions ------ ###


def make_alert(text, dica=""):
    """Esta função cria uma DIV contendo um ALERTA
    """
    if len(dica) > 0:
        dica_header = "Dica"
    else:
        dica_header = ""

    return html.Div([
        dbc.Alert([
                html.H4("Valor inválido!", className="alert-heading"),
                html.P(text),
                html.Hr(),
                html.H5(dica_header),
                html.P(dica, style={"textAlign": "center"})
            ], color="danger", dismissable=True)
    ])
