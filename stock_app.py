import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

# NSE-TATA dataset
df_nse = pd.read_csv("./NSE-TATA.csv")

df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)

# Predict NSE dataset by Closing Price
data_close = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

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

model_close = load_model("model_LSTM_Close.h5")

inputs_close = data_close[len(data_close) - len(valid_close) - 60:].values
inputs_close = inputs_close.reshape(-1, 1)
inputs_close = scaler_close.transform(inputs_close)

X_test_close = []
for i in range(60, inputs_close.shape[0]):
    X_test_close.append(inputs_close[i - 60:i, 0])
X_test_close = np.array(X_test_close)

X_test_close = np.reshape(X_test_close, (X_test_close.shape[0], X_test_close.shape[1], 1))
closing_price = model_close.predict(X_test_close)
closing_price = scaler_close.inverse_transform(closing_price)

train_close = data_close[:987]
valid_close = data_close[987:]
valid_close['Predictions'] = closing_price

# Predict NSE dataset by Price of Change
data_roc = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'ROC'])
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

model_roc = load_model("model_LSTM_ROC.h5")

inputs_roc = data_roc[len(data_roc) - len(valid_roc) - 60:].values
inputs_roc = inputs_roc.reshape(-1, 1)
inputs_roc = scaler_roc.transform(inputs_roc)

X_test_roc = []
for i in range(60, inputs_roc.shape[0]):
    X_test_roc.append(inputs_roc[i - 60:i, 0])
X_test_roc = np.array(X_test_roc)

X_test_roc = np.reshape(X_test_roc, (X_test_roc.shape[0], X_test_roc.shape[1], 1))
roc_price = model_roc.predict(X_test_roc)
roc_price = scaler_roc.inverse_transform(roc_price)

train_roc = data_roc[:987]
valid_roc = data_roc[987:]
valid_roc['Predictions'] = roc_price


df = pd.read_csv("./stock_data.csv")

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train_close.index,
                                y=valid_close["Close"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data closing price",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_close.index,
                                y=valid_close["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted price of change", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data price of change",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_roc.index,
                                y=valid_roc["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Price Of Change Rate'}
                        )
                    }

                )
            ])

        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Facebook Stocks High vs Lows",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Facebook Market Volume", style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])

    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["High"],
                       mode='lines', opacity=0.7,
                       name=f'High {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["Low"],
                       mode='lines', opacity=0.6,
                       name=f'Low {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x=df[df["Stock"] == stock]["Date"],
                       y=df[df["Stock"] == stock]["Volume"],
                       mode='lines', opacity=0.7,
                       name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Transactions Volume"})}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)