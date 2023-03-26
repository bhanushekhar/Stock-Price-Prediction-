import math
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Tkagg')
import time as t
import numpy as np
import pandas as pd
from datetime import time,date
import pandas_datareader as web
import streamlit as st
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')
st.title('Stock Trend Prediction')
list_stocks=[ ['AMZN','(AMAZON)'],["CCL","(Carnival Corporation)"],
 ["AMD","(Advanced Micro Devices)"],[ "NVDA","(NVIDIA Corporation )" ],["AAPL","(Apple Inc)"],
 ["TSLA","(Tesla)"],[ "F","(FORD)"],["META","(Meta Platforms)"],
 ["CSCO","(CISCO)"],["GOOG","(GOOGLE)"],["NFLX","(NETFLIX)"],["TWTR","(TWITTER)"]]
user_in=st.selectbox("select from below",list_stocks)
# user_in=st.text_input('Enter Stock Ticker','AAPL')
#df=web.DataReader(user_in[0], data_source='yahoo',start='2010-01-01',end='2022-11-19')
# ORIGINAL CLOSING PRICE 

start_time = st.slider(
    "When do you start?",
    value=(date(2010, 1, 1),date(2022, 11, 21)))
t.sleep(2)
#adding sleep so that we could get time to select stock and date
df=web.DataReader(user_in[0], data_source='yahoo',start=start_time[0],end=start_time[1])
st.write('Data from ',start_time[0],'to',start_time[1])
st.write(df.describe())
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('closing price USD($)')
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)
#############################
# #CALCULATING MEAN AVG FOR 100D AND 200D
mean_avg100=df.Close.rolling(100).mean()
mean_avg200=df.Close.rolling(200).mean()

st.subheader('Closing Price vs Time chart with 100MA')
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('closing price USD($)')
plt.plot(mean_avg100,'b',label="last_100D_AVG")
plt.plot(df.Close,'r',label="ORIGINAL")
plt.legend()
st.pyplot(fig)



st.subheader('Closing Price vs Time chart with 200MA')
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('closing price USD($)')
plt.plot(mean_avg200,'g',label="last_200D_AVG")
plt.plot(df.Close,'red',label="ORIGINAL")
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with both')
fig=plt.figure(figsize=(12,6))
plt.xlabel('Date',fontsize=18)
plt.ylabel('closing price USD($)')
plt.plot(mean_avg100,'b',label="last_100D_AVG")
plt.plot(mean_avg200,'g',label="last_200D_AVG")
plt.plot(df.Close,'r',label="ORIGINAL")
plt.legend()
st.pyplot(fig)


data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*0.8)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

training_data=scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]
for i in range(100,len(training_data)):
  x_train.append(training_data[i-100:i,0])#0-59
  y_train.append(training_data[i,0])#61
x_train,y_train=np.array(x_train),np.array(y_train)
#############################3
# # building the lstm model
model=Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)



# model=Sequential()
# model.add(LSTM(units=50,activation='relu',return_sequences=True,
# input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.2))

# model.add(LSTM(units=60,activation='relu',return_sequences=True,
# input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.3))

# model.add(LSTM(units=80,activation='relu',return_sequences=True,
# input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.4))

# model.add(LSTM(units=120,activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(1))

# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(x_train,y_train,epochs=1)

testing_data=scaled_data[training_data_len-100:,:]
x_test=[]
y_test=dataset[training_data_len: ,:]
for i in range(100,len(testing_data)):
  x_test.append(testing_data[i-100:i,0])

x_test=np.array(x_test)


predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
st.subheader('ROOT MEAN SQUARE FOR OUR MODEL IS')
st.write(rmse)

# f1 = round(f1_score(y_test, predictions, average='weighted'), 3)
# A = round((accuracy_score(y_test, predictions)*100),2)

# st.write(f1,A)

# visualizing
########
fig=plt.figure(figsize=(16,8))
plt.title('LSTM Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('closing price USD($)')
plt.plot(y_test,'b',label='original price')
plt.plot(predictions,'r',label='predicted price')
plt.legend()
st.pyplot(fig)

train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions

fig=plt.figure(figsize=(16,8))
plt.title('LSTM Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('closing price USD($)')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','val','predictions'],loc='lower right')
st.pyplot(fig)

new_df=df.filter(['Close'])
last_100d=new_df[-100:].values
scaled_100=scaler.transform(last_100d)
x_test=[]
x_test.append(scaled_100)
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
pred_price=model.predict(x_test)
pred_price=scaler.inverse_transform(pred_price)
st.write(pred_price)
