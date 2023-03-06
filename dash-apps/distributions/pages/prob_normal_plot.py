### ------ IMPORTS ------ ###
import base64
# import datetime
import io

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
from scipy import stats
from pandas.api.types import is_numeric_dtype

# --- Home Made --- #
from functions import prob_plot_functions as p_func

### ------ datasets ------ ####
from datasets import probplot




### ------ Constantes ------ ###
PERSISTENCE_TYPE = "session"
PERSISTENCE = True


### ------ FUNCTIONS ------ ###
def make_column_type(df):
    columns_options = []
    for col in df.columns:
        col_options = {"name": col, "id": col}
        if df[col].dtype != object:
            col_options["type"] = "numeric" # para permitir filtrar numéricamente
            # col_options['format'] = Format(precision=4, scheme=Scheme.fixed)
        columns_options.append(col_options)
    return columns_options


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

ref_options = [
    {'label': 'Sim', 'value': True},
    {'label': 'Não', 'value': False},
]

dataset_options = [
    {'label': 'Genérico', 'value': "Genérico"},
    {'label': 'Left Skewed', 'value': 'Left Skewed'},
    {'label': 'Right Skewed', 'value': 'Right Skewed'},
    {'label': 'Normal', 'value': 'Normal'},
    {'label': 'Usuário', 'value': "Usuário"},
]

randomization_drop_options = [
    {'label': 'Aleatório', 'value': 'None'},
    {'label': '42', 'value': 42},
    {'label': '1', 'value': 1},
    {'label': '2', 'value': 2},
    {'label': '3', 'value': 3},
    {'label': '4', 'value': 4},
    {'label': '5', 'value': 5},
]



### ------ LAYOUTS ------ ###

# --- MENU --- #
offcanvas = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-cog fa-2x contact_icons"),
            id="probplot-open-offcanvas", n_clicks=0, className="btn bg-transparent rounded-circle border-white shadow-none",
            ),
        dbc.Offcanvas(
            children = [
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(
                            html.Label("Dataset"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="probplot-dataset-dropdown",
                                        value = dataset_options[0]['value'],
                                        options = dataset_options,
                                        clearable = False,
                                        persistence=PERSISTENCE,
                                        persistence_type=PERSISTENCE_TYPE,
                                        ), #width=2

                                ), justify="center"
                            ),
                            ]),
                        ],
                    ),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(
                            html.Label("Aleatorizar posição:"), width="auto", align="center"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="probplot-randomization-dropdown",
                                        value = randomization_drop_options[1]['value'],
                                        options = randomization_drop_options,
                                        clearable = False,
                                        persistence=PERSISTENCE,
                                        persistence_type=PERSISTENCE_TYPE,
                                        ), #width=2
                                    )
                                ),
                            ], align="center", ),
                        ], justify="center"
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Label("Revelar:"), width="auto", align="end"
                        ),
                        dbc.Col([
                            dbc.Row(
                                dbc.Col(
                                    dmc.Switch(
                                        label="", onLabel="Sim", offLabel="Não", checked=False, size="lg", color="teal",
                                        id="probplot-revail-switch", persistence_type=PERSISTENCE_TYPE, persistence=PERSISTENCE),
                                    )
                                ),
                            ], align="center", width="auto"),
                        ], justify="center"
                    ),
                    # Entrada de dados
                    html.Hr(),
                    html.Div([
                        dbc.Row(
                            dbc.Col(
                                dcc.Upload(
                                    id='probplot-data-upload',
                                    children=html.Div([
                                        dbc.Button("Arraste e solte ou selecione arquivos", outline=True, color="success", className="me-1", size="lg"),
                                    ]),
                                    style={
                                        'textAlign': 'center',
                                    },
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                            ), style={"paddingBottom": "25px"}
                        ),
                        # Hidden div to show # WARNINGs
                        dbc.Row(
                            dbc.Col(
                                html.Div(id='probplot-data-upload-alert'),
                            ), style={"textAlign": "center", "margin": "10px"}
                        ),
                        # div to shwo user data
                        dbc.Row(
                            dbc.Col(
                                id="probplot-userdata-table"
                            )
                        ),
                        # Img Div to show the file tips
                        dbc.Row(
                            dbc.Col(
                                id="probplot-upload-image-descritption"
                            ), style={"paddingTop": "20px", "paddingBottom": "20px", "textAlign": "center"}
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    [
                                        dbc.Button("Baixe o arquivo csv de exemplo", id="probplot-example-download-button",
                                        outline=True, color="success", className="me-1", size="sm"),
                                        dcc.Download(id="probplot-example-download-csv"),
                                    ], style={"textAlign": "center"}, id="mean-with-constant-download-div"),
                            ), justify="center", style={"textAlign": "center", "paddingBottom": "25px", "paddingTop": "15px"},
                        ),
                    ], id="probplot-data-upload-div"),
                ],
            id="probplot-offcanvas",
            title="Preferências",
            is_open=False,
        ),
    ]
)


# --- MAIN LAYOUT --- #
layout = html.Div([

    dbc.Row([
        dbc.Col(
            sm = 0,
            lg = 1,
            align="center"
        ),
        dbc.Col(
            id="probplot-title",
            width = {"size": 10},
            align = "center"
        ),
        dbc.Col(
            offcanvas,
            width = {"size": 1},
            align="center"
        ),
    ], style = {'textAlign': 'center', 'paddingTop': '30px', 'paddingBottom': '30px'},),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id="probplot-fig-A", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
                dbc.Col(
                    dcc.Graph(id="probplot-fig-B", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
                dbc.Col(
                    dcc.Graph(id="probplot-fig-C", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id="probplot-fig-D", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
                dbc.Col(
                    dcc.Graph(id="probplot-fig-E", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
                dbc.Col(
                    dcc.Graph(id="probplot-fig-F", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
            ], style={"paddingTop": "10px"}),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id="probplot-fig-G", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
                dbc.Col(
                    dcc.Graph(id="probplot-fig-H", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
                dbc.Col(
                    dcc.Graph(id="probplot-fig-I", mathjax=True, config=configuracoes_grafico),
                    lg=4
                ),
            ], style={"paddingTop": "10px"}),
        ], lg=10)
    ], justify="center"),
    dcc.Store(id="probplot-dataset-store"),
    dcc.Store(id="probplot-normal-data-store"),
    dcc.Store(id="probplot-random-list-store"),
],)




### ------ CALLBACKS ------ ###
@callback(
    Output("probplot-example-download-csv", "data"),
    Input("probplot-example-download-button", "n_clicks"),
    prevent_initial_call=True,
)
def func_to_download_example_file(n_clicks):
    df = probplot.df_dataset.copy()
    return dcc.send_data_frame(df.to_csv, "example.csv", index=False)



@callback(
    Output(component_id="probplot-data-upload-div", component_property="style"),
    Input(component_id="probplot-dataset-dropdown", component_property="value")
)
def toggle_dataset_source(dataset):
    if dataset == "Usuário":
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@callback(
    Output(component_id="probplot-dataset-store", component_property="data"),
    Output(component_id="probplot-data-upload-alert", component_property="children"),
    Output(component_id="probplot-title", component_property="children"),
    Input(component_id="probplot-dataset-dropdown", component_property="value"),
    Input('probplot-data-upload', 'contents'),
    State('probplot-data-upload', 'filename'),
)
def toggle_dataset(dropdown_dataset, list_of_contents, list_of_names):
    base = "Gráfico de Probabilidade Normal - "

    if dropdown_dataset == "Genérico":
        df = probplot.df_dataset.copy()
        return df.to_json(date_format='iso', orient='split'), [], html.H2(base + "Dados genéricos")

    elif dropdown_dataset == "Left Skewed":
        df = probplot.df_left_skewed.copy()
        return df.to_json(date_format='iso', orient='split'), [], html.H2(base + "Left Skewed")

    elif dropdown_dataset == "Right Skewed":
        df = probplot.df_right_skewed.copy()
        return df.to_json(date_format='iso', orient='split'), [], html.H2(base + "Right Skewed")

    elif dropdown_dataset == "Normal":
        df = probplot.df_normal.copy()
        return df.to_json(date_format='iso', orient='split'), [], html.H2(base + "Normal")

    else:
        if list_of_names is None:
            df = probplot.df_dataset.copy()
            return df.to_json(date_format='iso', orient='split'), [], html.H2(base + "Dados genéricos")

        else:
            if len(list_of_names) != 1:
                text = f"Foram recebidos {len(list_of_names)} arquivos, mas é aceito apenas um arquivo."
                alert = p_func.make_alert(text)
                return no_update, alert, no_update
            else:
                if len(list_of_contents[0]) < 6:
                    text = f"Ocorreu um erro ao processar os dados"
                    dica = f"O arquivo {list_of_names[0]} não contém dados!"
                    alert = p_func.make_alert(text, dica)
                    return no_update, alert, no_update

                df = p_func.parse_contents(list_of_contents[0], list_of_names[0])

                if type(df) == str:
                    text = f"Ocorreu um erro ao processar o arquivo"
                    dica = "São aceitos apenas arquivos '.csv'"
                    alert = p_func.make_alert(text, dica)
                    return no_update, alert, no_update

                else:
                    if df.shape[1] > 1:
                        textos = ("Muitas colunas!", "O arquivo fornecido ", "contém", "colunas, mas apenas 1 coluna é necessária.", "O teste será realizado utilizando a primeira coluna")

                        alert = dbc.Alert([
                            html.H4(textos[0], className="alert-heading"),
                            html.P(f"{textos[1]} ('{list_of_names[0]}') {textos[2]} {str(df.shape[1])} {textos[3]}"),
                            html.P(f"{textos[4]} ('{df.columns[0]}') "),
                        ], color="warning", dismissable=True)
                        data_name = list_of_names[0]
                        # no return because is just a WARNING
                    else:
                        alert = []



        if not is_numeric_dtype(df[df.columns[0]]):
            text = f"Pelo menos uma entrada da coluna '{df.columns[0]}' não é numérica!"
            dica = "O separador de casas decimais é o ponto ('.')"
            alert = p_func.make_alert(text, dica)
            return no_update, alert, no_update


        if df[df.columns[0]].isnull().values.any():
            text = f"Pelo menos uma entrada da coluna '{df.columns[0]}' esta vazia!"
            alert = p_func.make_alert(text)
            return no_update, alert, no_update


        if df.shape[0] < 3:
            text = f"São necessários no mínimo 3 observações para aplicar o teste, mas o conjunto de dados contém apenas '{df.shape[0]}' entradas"
            alert = p_func.make_alert(text)
            return no_update, alert, no_update



    return df.to_json(date_format='iso', orient='split'), alert, html.H2(base + list_of_names[0])



@callback(
    Output(component_id="probplot-userdata-table", component_property="children"),
    Input(component_id="probplot-dataset-dropdown", component_property="value"),
    Input(component_id="probplot-dataset-store", component_property="data"),
    Input('probplot-data-upload', 'contents'),
    State('probplot-data-upload', 'filename'),
)
def display_data_table(dropdown_dataset, data, list_of_contents, list_of_names):
    if dropdown_dataset != "Usuário":
        return []
    if list_of_names is None:
        return []
    else:
        df = pd.read_json(data, orient='split')
        table = dash_table.DataTable(
                    columns = make_column_type(df),
                    data = df.to_dict('records'),
                    style_data = style_data,
                    style_table = style_table,
                    style_cell = style_cell,
                    sort_action="native",
                    style_header = style_header
                    )
        return table


@callback(
    Output(component_id="probplot-fig-A", component_property="figure"),
    Output(component_id="probplot-fig-B", component_property="figure"),
    Output(component_id="probplot-fig-C", component_property="figure"),
    Output(component_id="probplot-fig-D", component_property="figure"),
    Output(component_id="probplot-fig-E", component_property="figure"),
    Output(component_id="probplot-fig-F", component_property="figure"),
    Output(component_id="probplot-fig-G", component_property="figure"),
    Output(component_id="probplot-fig-H", component_property="figure"),
    Output(component_id="probplot-fig-I", component_property="figure"),
    Input(component_id="probplot-revail-switch", component_property="checked"),
    Input(component_id="probplot-dataset-store", component_property="data"),
    Input(component_id="probplot-normal-data-store", component_property="data"),
    Input(component_id="probplot-random-list-store", component_property="data")
)
def display_plot(revail, user, normal, ordem):
    df_user = pd.read_json(user, orient='split')
    x_data, y_data, slope, intercept, rpearson = p_func.calculate_quartils(df_user[df_user.columns[0]])
    # gráfico do usuario
    df_user = pd.DataFrame({
        "Quantis Teóricos": x_data,
        "Dados ordenados": y_data
    })
    # print(ordem)
    df_normal = pd.read_json(normal, orient='split')
    figures = list(range(0,9))
    if revail:
        figures[ordem[0]] = p_func.make_probplot(df_user, color="red")
    else:
        figures[ordem[0]] = p_func.make_probplot(df_user, color="blue")
    for i in range(1,9):
        x_data, y_data, slope, intercept, rpearson = p_func.calculate_quartils(df_normal[df_normal.columns[i]])
        df_aux = pd.DataFrame({
            "Quantis Teóricos": x_data,
            "Dados ordenados": y_data
        })
        figures[ordem[i]] = p_func.make_probplot(df_aux, color="blue")

    return figures


@callback(
    Output(component_id="probplot-random-list-store", component_property="data"),
    Input(component_id="probplot-randomization-dropdown", component_property="value")
)
def get_random_value(rand_seed):
    aleatorio = np.arange(9)

    if rand_seed == "None":
        rand_seed = None

    rng = np.random.default_rng(rand_seed)
    rng.shuffle(aleatorio)


    return aleatorio


@callback(
    Output(component_id="probplot-normal-data-store", component_property="data"),
    Input(component_id="probplot-randomization-dropdown", component_property="value"),
    Input(component_id="probplot-dataset-store", component_property="data"),
)
def make_normal_data(rand_seed, dataset):

    if rand_seed is None:
        random_numbers = [None]*8
    else:
        random_numbers = np.arange(0,8)
        np.random.shuffle(random_numbers)
    df = pd.read_json(dataset, orient='split')

    x = df[df.columns[0]].to_numpy()

    series = [pd.Series(np.linspace(x.min(), x.max(), x.size), name="Quartil")]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for i in range(8):
        # if rand_seed != 'None':
        np.random.seed(random_numbers[i])
        series.append(pd.Series(np.random.normal(loc=x.mean(), scale=x.std(ddof=1), size=x.size), name=f"normal_{letters[i]}"))

    # gráfico do usuario
    df = pd.concat(series, axis=1)

    return df.to_json(date_format='iso', orient='split')




#################################################
### Calback para abrir a aba de configurações ###
#################################################
@callback(
    Output("probplot-offcanvas", "is_open"),
    Input("probplot-open-offcanvas", "n_clicks"),
    [State("probplot-offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open



#
