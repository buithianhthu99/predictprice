#Import the necessary data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

df=pd.read_csv("NSE-TATA.csv")
df.head()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

n = 10
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)-n),columns=['Date', 'ROC'])
for i in range(n, len(data)):
    new_dataset["Date"][i-n]=data["Date"][i]
    new_dataset["ROC"][i-n]=(data["Close"][i] - data["Close"][i-n])/data["Close"][i-n]*100
new_dataset.head()

new_dataset["Date"]=pd.to_datetime(new_dataset.Date,format="%Y-%m-%d")
new_dataset.index=new_dataset['Date']

plt.figure(figsize=(16,8))
plt.plot(new_dataset["ROC"],label='Price range of change history')

scaler = MinMaxScaler(feature_range=(0, 1))
new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)
final_dataset = new_dataset.values

train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

#Initializing our recurrent neural network
rnn = Sequential()

#Adding our first LSTM layer
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_train_data.shape[1], 1)))

#Perform some dropout regularization
rnn.add(Dropout(0.2))

#Adding three more LSTM layers with dropout regularization
for i in [True, True, False]:
    rnn.add(LSTM(units = 45, return_sequences = i))
    rnn.add(Dropout(0.2))

#Adding our output layer
rnn.add(Dense(units = 1))

#Compiling the recurrent neural network
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the recurrent neural network
rnn.fit(x_train_data, y_train_data, epochs = 100, batch_size = 32)

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_ROC=rnn.predict(X_test)
predicted_ROC=scaler.inverse_transform(predicted_ROC)

rnn.save("model_RNN_ROC.h5")