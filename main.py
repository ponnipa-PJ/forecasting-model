# Import packages
from cmath import log
import os
from flask import Flask, request, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
import time
import math

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'

SECRET_KEY = os.urandom(24)

# Display plots based on user's selection of model

@app.route('/index')

@app.route('/')

@app.route('/result', methods=['POST','GET'])
def home():
    fcast = ""
    if request.method == 'POST':
        
        method = request.form['forecast']
            
            # Forecasting using saved ARMA model
        if method == "lstm":
            # target = os.path.join(app_root, 'static/')
            f = open('filename.txt', 'r')
            name = f.read()
            
            apple_training_complete = pd.read_csv(name)
            col_label = apple_training_complete.columns[len(apple_training_complete.columns.values)-1]
            apple_training_processed = apple_training_complete.iloc[:,2:3].values
            scaler = MinMaxScaler(feature_range = (0, 1))
            apple_training_scaled = scaler.fit_transform(apple_training_processed)
            features_set = []
            labels = []
            for i in range(60, apple_training_scaled.shape[0]):
                features_set.append(apple_training_scaled[i-60:i, 0])
                labels.append(apple_training_scaled[i, 0])
                
            features_set, labels = np.array(features_set), np.array(labels)
            features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
                
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))

            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))

            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units = 1))
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            model.fit(features_set, labels, epochs = 20, batch_size = 32, verbose=0)
                
            apple_testing_complete = pd.read_csv(name)
            apple_testing_processed = apple_testing_complete.iloc[:,2:3].values
            apple_total = pd.concat((apple_training_complete[col_label], apple_testing_complete[col_label]), axis=0)
            test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
            test_inputs = test_inputs.reshape(-1,1)
            test_inputs = scaler.transform(test_inputs)
            test_features = []
            for i in range(60, test_inputs.shape[0]):
                test_features.append(test_inputs[i-60:i, 0])
            test_features = np.array(test_features)
            test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
            predictions = model.predict(test_features)
            predictions = scaler.inverse_transform(predictions)
            
            score = model.evaluate(features_set, labels, batch_size=32, verbose=0)
            MSE = score
            RMSE = math.sqrt(MSE)
            
            plt.figure(figsize=(17,6))
            plt.plot(apple_testing_processed, color='blue', label='Actual Price')
            plt.plot(predictions , color='red', label='Predicted Price')
            plt.title('Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            new_arma_plot = "lstm_plot_" + str(time.time()) + ".png"
            plt.savefig('static/' + new_arma_plot)
                
            return render_template('index.html', forecast='LSTM', rmse=RMSE, mse=MSE, fcast='static/' + new_arma_plot)

            # Forecasting using saved ARIMA model
        elif method =="arima":
            new_arima_plot = "arima_plot_1591687174.918447" + ".png"
            return render_template('index.html', forecast='ARIMA', fcast='static/' + new_arima_plot)

            # Forecasting using saved Exponential Smoothing model
        elif method == 'exp':            
            new_exp_plot = "exp_plot_1591687185.701119" + ".png"
            return render_template('index.html', forecast='Exponential Smoothing', fcast='static/' + new_exp_plot)

            # # Forecasting using saved Prophet model
            # elif method == 'prophet':
            #     new_prophet_plot = "prophet_plot_1591687195.249861" + ".png"
            #     return render_template('index.html', forecast='Prophet', fcast='static/' + new_prophet_plot)

            # # Forecasting using saved AutoARIMA model
            # elif method == 'auto_arima':            
            #     new_auto_arima_plot = "auto_arima_plot_1591687207.990299" + ".png"
            #     return render_template('index.html', forecast='Auto-arima', fcast='static/' + new_auto_arima_plot)
    return render_template('index.html', fcast=fcast)

if __name__ == '__main__':
    app.run(debug=False)
