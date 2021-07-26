import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

LSTM_model_close = load_model("model_LSTM_Close.h5")
LSTM_model_roc = load_model("model_LSTM_ROC.h5")
RNN_model_close = load_model("model_RNN_Close.h5")
RNN_model_roc = load_model("model_RNN_ROC.h5")

def get_data_closing(data, model):
    data["Date"] = pd.to_datetime(data.Date, format="%Y-%m-%d")
    data.index = data['Date']

    final_data = data.sort_index(ascending=True, axis=0)
    data_close = pd.DataFrame(index=range(0, len(final_data)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        data_close["Date"][i] = data['Date'][i]
        data_close["Close"][i] = data["Close"][i]

    data_close.index = data_close.Date
    data_close.drop("Date", axis=1, inplace=True)

    dataset_close = data_close.values

    train_close = dataset_close[0:987, :]
    valid_close = dataset_close[987:, :]

    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaled_data_close = scaler_close.fit_transform(dataset_close)

    x_train_close, y_train_close = [], []

    for i in range(60, len(train_close)):
        x_train_close.append(scaled_data_close[i - 60:i, 0])
        y_train_close.append(scaled_data_close[i, 0])

    x_train_close, y_train_close = np.array(x_train_close), np.array(y_train_close)

    x_train_close = np.reshape(x_train_close, (x_train_close.shape[0], x_train_close.shape[1], 1))

    inputs_close = data_close[len(data_close) - len(valid_close) - 60:].values
    inputs_close = inputs_close.reshape(-1, 1)
    inputs_close = scaler_close.transform(inputs_close)

    X_test_close = []
    for i in range(60, inputs_close.shape[0]):
        X_test_close.append(inputs_close[i - 60:i, 0])
    X_test_close = np.array(X_test_close)

    X_test_close = np.reshape(X_test_close, (X_test_close.shape[0], X_test_close.shape[1], 1))

    if model == "LSTM":
        closing_price = LSTM_model_close.predict(X_test_close)
    else:
        closing_price = RNN_model_close.predict(X_test_close)

    closing_price = scaler_close.inverse_transform(closing_price)

    train_close = data_close[:987]
    valid_close = data_close[987:]
    valid_close['Predictions'] = closing_price

    return {"train_close": train_close, "valid_close": valid_close}


def get_data_roc(data, model):
    data["Date"] = pd.to_datetime(data.Date, format="%Y-%m-%d")
    data.index = data['Date']

    final_data = data.sort_index(ascending=True, axis=0)
    data_roc = pd.DataFrame(index=range(0, len(final_data)), columns=['Date', 'ROC'])
    n = 10
    for i in range(n, len(data)):
        data_roc["Date"][i-n] = data['Date'][i]
        data_roc["ROC"][i-n] = (data["Close"][i] - data["Close"][i-n])/data["Close"][i-n]*100

    data_roc.index = data_roc.Date
    data_roc.drop("Date", axis=1, inplace=True)

    dataset_roc = data_roc.values

    train_roc = dataset_roc[0:987, :]
    valid_roc = dataset_roc[987:, :]

    scaler_roc = MinMaxScaler(feature_range=(0, 1))
    scaled_data_roc = scaler_roc.fit_transform(dataset_roc)

    x_train_roc, y_train_roc = [], []

    for i in range(60, len(train_roc)):
        x_train_roc.append(scaled_data_roc[i - 60:i, 0])
        y_train_roc.append(scaled_data_roc[i, 0])

    x_train_roc, y_train_roc = np.array(x_train_roc), np.array(y_train_roc)

    x_train_roc = np.reshape(x_train_roc, (x_train_roc.shape[0], x_train_roc.shape[1], 1))

    inputs_roc = data_roc[len(data_roc) - len(valid_roc) - 60:].values
    inputs_roc = inputs_roc.reshape(-1, 1)
    inputs_roc = scaler_roc.transform(inputs_roc)

    X_test_roc = []
    for i in range(60, inputs_roc.shape[0]):
        X_test_roc.append(inputs_roc[i - 60:i, 0])
    X_test_roc = np.array(X_test_roc)

    X_test_roc = np.reshape(X_test_roc, (X_test_roc.shape[0], X_test_roc.shape[1], 1))
    
    if model == "LSTM":
        roc_price = LSTM_model_roc.predict(X_test_roc)
    else:
        roc_price = RNN_model_roc.predict(X_test_roc)

    roc_price = scaler_roc.inverse_transform(roc_price)

    train_roc = data_roc[:987]
    valid_roc = data_roc[987:]
    valid_roc['Predictions'] = roc_price

    return {"train_roc": train_roc, "valid_roc": valid_roc}


app = dash.Dash()
server = app.server

df_nse = pd.read_csv("./NSE-TATA.csv")
df = pd.read_csv("./stock_data.csv")

app.layout = html.Div([
    html.H1("STOCK PRICE ANALYSIS DASHBOARD", style={"textAlign": "center", "fontSize": 32}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='NSE-TATAGLOBAL STOCK DATA', children=[
            html.Div([
                html.H1('NSE-TATAGLOBAL STOCK DATA',
                        style={'textAlign': 'center'}),
                html.Div(style={'display': 'flex', "margin-left": 80}, children=[
                    html.Label("Filter", style={"fontSize": 24, "margin-right": 10, "font-style": "italic", "font-weight": "bold"}),
                    dcc.Dropdown(id='my-dropdown2',
                                options=[{'label': 'LSTM', 'value': 'LSTM'},
                                        {'label': 'RNN', 'value': 'RNN'},
                                        {'label': 'XGBoost ', 'value': 'XGBoost '}],
                                value='LSTM',
                                style={"width": "40%", "padding-left": 10}),
                ]),
                dcc.Graph(id='actual closing NSE'),
                dcc.Graph(id='predict closing NSE'),
                dcc.Graph(id='actual roc NSE'),
                dcc.Graph(id='predict roc NSE')
            ])
        ]),
        dcc.Tab(label="STOCK DATA", children=[
            html.Div([
                html.H1("STOCK DATA",
                        style={'textAlign': 'center'}),
                html.Div(style={'display': 'flex', "margin-left": 80}, children=[
                    html.Label("Filter", style={"fontSize": 24, "margin-right": 10, "font-style": "italic", "font-weight": "bold"}),
                    dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"width": "100%"}),
                    dcc.Dropdown(id='my-dropdown1',
                                options=[{'label': 'LSTM', 'value': 'LSTM'},
                                        {'label': 'RNN', 'value': 'RNN'},
                                        {'label': 'XGBoost ', 'value': 'XGBoost '}],
                                value='LSTM',
                                style={"width": "40%", "padding-left": 10}),
                ]),
                dcc.Graph(id='actual closing'),
                dcc.Graph(id='predict closing'),
                dcc.Graph(id='actual roc'),
                dcc.Graph(id='predict roc')
            ], className="container"),
        ])
    ])
])


@app.callback(Output('actual closing NSE', 'figure'),
              Output('predict closing NSE', 'figure'),
              Output('actual roc NSE', 'figure'),
              Output('predict roc NSE', 'figure'),
              Input('my-dropdown2', 'value'))
def update_graph(selected_dropdown):  
    result1 = get_data_closing(df_nse, selected_dropdown)
    result2 = get_data_roc(df_nse, selected_dropdown)
    figure1 = {
                "data": [
                    go.Scatter(
                        x=result1["train_close"].index,
                        y=result1["valid_close"]["Close"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Actual closing price</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Closing Rate'}
                )
            }
    figure2 = {
                "data": [
                    go.Scatter(
                        x=result1["valid_close"].index,
                        y=result1["valid_close"]["Predictions"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Predicted closing price</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Closing Rate'}
                )
            }
    figure3 = {
                "data": [
                    go.Scatter(
                        x=result2["train_roc"].index,
                        y=result2["valid_roc"]["ROC"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Actual price of change</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price Of Change Rate'}
                )
            }
    figure4 = {
                "data": [
                    go.Scatter(
                        x=result2["valid_roc"].index,
                        y=result2["valid_roc"]["Predictions"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Predicted price of change</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price Of Change Rate'}
                )
            }

    return figure1, figure2, figure3, figure4


@app.callback(Output('actual closing', 'figure'),
              Output('predict closing', 'figure'),
              Output('actual roc', 'figure'),
              Output('predict roc', 'figure'),
              Input('my-dropdown', 'value'), 
              Input('my-dropdown1', 'value'))
def update_graph(selected_dropdown1, selected_dropdown2):
    filter_df = pd.DataFrame()
    for stock in selected_dropdown1:
        filter_df = filter_df.append(df.loc[df['Stock'] == stock])
    
    result1 = get_data_closing(filter_df, selected_dropdown2)
    result2 = get_data_roc(filter_df, selected_dropdown2)
    figure1 = {
                "data": [
                    go.Scatter(
                        x=result1["train_close"].index,
                        y=result1["valid_close"]["Close"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Actual closing price</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Closing Rate'}
                )
            }
    figure2 = {
                "data": [
                    go.Scatter(
                        x=result1["valid_close"].index,
                        y=result1["valid_close"]["Predictions"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Predicted closing price</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Closing Rate'}
                )
            }
    figure3 = {
                "data": [
                    go.Scatter(
                        x=result2["train_roc"].index,
                        y=result2["valid_roc"]["ROC"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Actual price of change</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price Of Change Rate'}
                )
            }
    figure4 = {
                "data": [
                    go.Scatter(
                        x=result2["valid_roc"].index,
                        y=result2["valid_roc"]["Predictions"],
                        mode='markers'
                    )
                ],
                "layout": go.Layout(
                    title='<b>Predicted price of change</b>',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price Of Change Rate'}
                )
            }

    return figure1, figure2, figure3, figure4


if __name__ == '__main__':
    app.run_server(debug=True)