'''
This file contains the model class which is used to create a model object
author: @jsotoL
created: 24-01-2023
version: 1.0
'''

import itertools
import os
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import pandas as pd
import prophet
from neuralprophet import NeuralProphet
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
#from tensorflow.keras.models import load_model

# Path: Model.py
class Model:
    def __init__(self,growth,change_point=0.001,seasonality="multiplicative"): #init new model
        try:
            self.model = prophet.Prophet(growth=growth,changepoint_prior_scale=change_point,seasonality_mode=seasonality)
            print("Model created")
        except:
            print("Error: growth must be either 'linear' or 'logistic'")

    def fit(self, df):#fit model to data
        self.model.fit(df)
    def add_cap(self, df, cap):#add cap to model
        df['cap'] = cap
        return df
    def predict(self, df):#predict future values
        return self.model.predict(df)
    def convert_to_prophet_format(self, df):#convert dataframe to prophet format
        df = df.reset_index()
        df.columns = ['ds', 'y']
        return df
    def model_plot(self,forecast,name): #save figures
        if not os.path.exists(name):
            os.mkdir(name)
        self.model.plot(forecast).savefig(f'{name}\{name}_Figure.png') #plot model
        self.model.plot_components(forecast).savefig(f"{name}\{name}_components.png") #plot model components
    def future(self, periods,cap):#create future dataframe
        try:
            future = self.model.make_future_dataframe(periods=periods)
            future['cap'] = cap
        except:
            future = self.model.make_future_dataframe(periods=periods)
            future['cap'] = cap[:len(future)]
        return future

    def hyperparameter_tuning(self, df, params):#hyperparameter tuning
        self.model = prophet.Prophet(**params)
        return self.model
    def cross_validation_run(self, df):
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here
        cutoffs = pd.to_datetime(['2017-01-01', '2018-01-01', '2019-01-01','2020-01-01'])  # Use cutoffs with training data
        for params in all_params:
            m = prophet.Prophet(**params).fit(df)  # Fit model with given params
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='400 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['mae'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['mae'] = rmses
        print(tuning_results)

class Model_Neural:
    def __init__(self): #init new model
        self.model = NeuralProphet()
        print("Model created")

class LSTM_model(): #init new model
    def __init__(self):

        self.model = Sequential()#init model
    def add_layer(self, units, activation, input_shape=None):#First layer must have input shape in TRUE
        if input_shape:
            self.model.add(LSTM(units=units, activation=activation, input_shape=input_shape))#add layer with input shape (only for first layer)
        else:
            self.model.add(LSTM(units=units, activation=activation)) #add layer without input shape (only for hidden layers)
    def add_dense_layer(self, units, activation):#add dense layer
        self.model.add(Dense(units=units, activation=activation))
    def add_dropout(self, rate):#rate is the percentage of neurons to drop USE FOR OVERFITTING PROBLEMS
        self.model.add(Dropout(rate))
    def add_batch_normalization(self):#normalizes the data USE FOR OVERFITTING PROBLEMS
        self.model.add(BatchNormalization())
    def compile_model(self, optimizer, loss): #compile model can be adam, rmsprop, sgd, and loss can be mse, mae, mape
        self.model.compile(optimizer=optimizer, loss=loss)
    def fit_model(self, X_train, y_train, epochs, batch_size):#fit model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    def predict(self, X_test):#predict future values
        return self.model.predict(X_test)
    def save_model(self, name):#save model
        self.model.save(F'{name}.h5')

    def summary(self): #summary of model layers
        self.model.summary()
    def delete_layer(self, index): #delete layer by index see summary for index
        self.model.pop(index)

    def evaluate(self, X_test, y_test):#evaluate model
        return self.model.evaluate(X_test, y_test)

    def load_model(self, name):#load model
        self.model = load_model(name)

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers_LSTM', 1, 3)):
        model.add(LSTM(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu', return_sequences=True))
    for z in range(hp.Int('num_layers_Dense', 1, 5)):
        model.add(Dense(units=hp.Int(f'units_{z}', min_value=32, max_value=512, step=32), activation='relu'))
    for y in range(hp.Int('num_layers_Dropout', 1, 2)):
        model.add(Dropout(rate=hp.Float(f'units_{y}', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=1)) #output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_model_Neural(hp):
    model = NeuralProphet()
    model.add_lagged_regressor(name='y', lag=hp.Int('lag', 1, 5))
    model.add_lagged_regressor(name='cap', lag=hp.Int('lag', 1, 5))
    model.add_seasonality(name='weekly', period=7, fourier_order=hp.Int('fourier_order', 3, 10))
    model.add_seasonality(name='monthly', period=30.5, fourier_order=hp.Int('fourier_order', 3, 10))
    model.add_seasonality(name='yearly', period=365.25, fourier_order=hp.Int('fourier_order', 3, 10))
    model.add_country_holidays(country_name='US')
    return model

class normal_Prophet:
    def __init__(self):
        self.model = prophet.Prophet(seasonality_mode='multiplicative')
    def fit(self, df):
        self.model.fit(df)
    def predict(self,periods):
        future = self.model.make_future_dataframe(periods=periods)
        return self.model.predict(future)
    def plot(self, df):
        self.model.plot(df)
    def plot_components(self, df):
        self.model.plot_components(df)
    def save_model(self, name):
        self.model.save(F'{name}.pkl')
    def load_model(self, name):
        self.model = prophet.load(F'{name}.pkl')
    def cross_validation(self, df):
        df_cv = cross_validation(self.model, horizon='400 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        print(df_p)
    def hyperparameter_tuning(self, df, params):
        self.model = prophet.Prophet(**params)
        return self.model
    def cross_validation_run(self, df):
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here
        cutoffs = pd.to_datetime(['2017-01-01', '2018-01-01', '2019-01-01','2020-01-01'])  # Use cutoffs with training data
        for params in all_params:
            m = prophet.Prophet(**params).fit(df)  # Fit model with given params
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='400 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['mae'].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['mae'] = rmses
        print(tuning_results)


