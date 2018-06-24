# -*- coding: utf-8 -*-
"""
Goal: utilities
"""

#%%
# Dependencies
import time
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import json


#%%
# Inspired by https://keras.io/layers/core
class ModelLayer:
    
    def __init__(
            self,
            num_neurons,
            layer_type = 'Dense',
            dropout_rate = None,
            activation = None
        ):
        # Type of layer: Dense, LSTM
        # > Default: Dense
        self.type = layer_type
        
        # Number of neurons
        self.neurons = num_neurons
        
        # Dropout rate after this layer > https://keras.io/layers/core/#dropout
        # > Default: None
        self.dropout_rate = dropout_rate
        
        # Activation function
        # > Default: None (Keras decides default)
        self.activation = activation
        
    def as_json(self):
        result = {
            'num_neurons': self.neurons,
            'layer_type': self.type,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
            }


        return result
    
class ExperimentPlotConfig():
    def __init__(
            self,
            raw_data_visualization = True,
            raw_vs_adapted = True,
            actuals_vs_predicted = True
        ):
        
        self.raw_data_visualization = raw_data_visualization
        self.raw_vs_adapted = raw_vs_adapted
        self.actuals_vs_predicted = actuals_vs_predicted
        
    
class LSTMModelConfig:
    
    def __init__(
            self,
#            input_layer,
            hidden_layers,
#            output_layer,
            window_size = 0,
            data_train_ratio = 0.7,
            data_test_ratio = 0.3,
            data_validation_ratio = 0,
            data_stationary = False,

            compile_loss_function = "mse",
            compile_optimizer = "adam",
            fit_epochs = 1,
            fit_batch_size = 1
        ):
        
        # Window size
        self.window_size = window_size
        
        # Proportion between train/validation/test data
        self.data_train_ratio = round(data_train_ratio,3)
        self.data_test_ratio = round(data_test_ratio,3)
        self.data_validation_ratio = round(1 - self.data_train_ratio - self.data_test_ratio,3)
    
        # Do we want to remove trend from the data? -> https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
        self.data_stationary = data_stationary
    
        # Layers of the model
#        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
#        self.output_layer = output_layer
    
        # Model training configuration
        self.compile_loss_function = compile_loss_function
        self.compile_optimizer = compile_optimizer
        self.fit_epochs = fit_epochs
        self.fit_batch_size = fit_batch_size
        
    def as_json(self):
        result = {
            'window_size': self.window_size,
            'data_train_ratio': self.data_train_ratio,
            'data_test_ratio': self.data_test_ratio,
            'data_validation_ratio': self.data_validation_ratio,
            'data_stationary': self.data_stationary,
            'compile_loss_function': self.compile_loss_function,
            'compile_optimizer': self.compile_optimizer,
            'fit_epochs': self.fit_epochs,
            'fit_batch_size': self.fit_batch_size,
            'hidden_layers': list()
            }
        
        for layer in self.hidden_layers:
            result['hidden_layers'].append(layer.as_json())

        return result
   
#%%
def convert_seconds(seconds):
    exp_dur_h, seconds =  seconds // 3600, seconds % 3600
    exp_dur_m, exp_dur_s = seconds // 60, seconds % 60
    
    exp_dur_h = str(round(exp_dur_h))
    exp_dur_m = str(round(exp_dur_m))
    exp_dur_s = str(round(exp_dur_s, 3))
    
    return exp_dur_h, exp_dur_m, exp_dur_s

def dataset_splitting(model_config, dataset):
    
    i_train = model_config.data_train_ratio
    i_train = len(dataset) * i_train
    i_train = int(round(i_train))
    
    i_validation = model_config.data_train_ratio + model_config.data_validation_ratio
    i_validation = len(dataset) * i_validation
    i_validation = int(round(i_validation))
    
    i_test = 1
    i_test = len(dataset) * i_test
    i_test = int(round(i_test))
    
    #print(model_config.data_train_ratio)
    #print(model_config.data_validation_ratio)
    #print(model_config.data_test_ratio)
    #print('----')
    #print(i_train)
    #print(i_validation)
    #print(i_test)
    #print('----')
    
    train_dataset = dataset[:i_train]
    validation_dataset = dataset[i_train:i_validation]
    test_dataset = dataset[i_validation:]
    
    train_dataset = pd.DataFrame(train_dataset)
    validation_dataset = pd.DataFrame(validation_dataset)
    test_dataset = pd.DataFrame(test_dataset)
    
    return train_dataset, validation_dataset, test_dataset

#%%
    
# Inspired from https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

def scale_2(train, validation, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))        
    scaler = scaler.fit(train)
    
    train_scaled = scaler.transform(train)
#    print('validation', validation.shape, validation, len(validation))
    if (len(validation) > 0):
        validation_scaled = scaler.transform(validation)
    else:
        validation_scaled = validation
#    print('validation_scaled', validation_scaled.shape, validation_scaled)
    test_scaled = scaler.transform(test)
    
    return scaler, train_scaled, validation_scaled, test_scaled
    
def prepare_moving_window(values, window_size):
    values2 = pd.DataFrame(values)
    series_s = values2.copy()
    column_names = []
    for i in range(window_size):
        column_names.append('T-' + str(window_size - i))
        values2 = pd.concat([values2, series_s.shift(-(i+1))], axis = 1)
      
    values2.dropna(axis=0, inplace=True)
    column_names.append('TARGET')
    values2.columns = column_names
    print(values2.shape) # it has lost {window_size} rows from the original shape of train_scaled
    
    return values2
    
def adapt_data_for_LSTM(train_data, test_data):   
    train_X = pd.DataFrame(train_data).iloc[:,:-1].values
    train_y = pd.DataFrame(train_data).iloc[:,-1].values
    test_X = pd.DataFrame(test_data).iloc[:,:-1].values
    test_y = pd.DataFrame(test_data).iloc[:,-1].values

    # Reshape for LSTM: (samples, time steps, features)
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)
    
#    print(train_X.shape)
#    print(train_y.shape)
#    print(test_X.shape)
#    print(test_y.shape)
    
    return train_X, train_y, test_X, test_y
    

   
def compose_model_2(model_config):
    print('> Starting model compilation...')
    
    # Input layer
    model = Sequential()
    print('-- Input layer added')
    
    # Hidden layers
    is_first_LSTM = True
    
    for layer in model_config.hidden_layers:
        print(layer.neurons)
        if (layer.type == 'Dense'):
            model.add(Dense(1))
            print('-- Dense layer added')
        elif (layer.type == 'LSTM'):
            if (is_first_LSTM):
                is_first_LSTM = False

                model.add(LSTM(
                        input_shape = (model_config.window_size, 1),
                        output_dim = layer.neurons, #50
                        return_sequences = True)
                    )
        
            else:
                print('add another')
                model.add(LSTM(layer.neurons))
#                model.add(LSTM(
#                        input_shape = (model_config.window_size,1),
#                        output_dim = layer.neurons, #50
#                        return_sequences = True)
#                    )
            
            print('-- LSTM layer added')
        
    # Output layer
    model.add(Dense(1))
    model.add(Activation("linear"))
    print('-- Output layer added')

    # Compile model
    model.compile(
            loss = model_config.compile_loss_function,
            optimizer = model_config.compile_optimizer,
            metrics = ['accuracy']
            )
    
    print('Model compiled')
    print(model.summary())
    
    return model
    


def fit_model_2(model_config, model, features, target, verbose = 0):
    
    print('> Starting model training...')
    start = time.time()
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    
    history = model.fit(
            features,
            target,
            batch_size = model_config.fit_batch_size,
            epochs = model_config.fit_epochs,
            verbose=verbose,
            validation_split=0.2,
#            callbacks=[early_stop],
#            shuffle=False
            )
    print('Model trained')
    training_time = round(time.time() - start, 3)
    print("-- Training Time : ", training_time, 'seconds')
        
    return model, history, training_time


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
    
    
def results_summary_adapted_from_1(error_scores, error_texts):
    for text in error_texts:
        print(text)
    
    results = DataFrame()
    results['rmse'] = error_scores
    print(results.describe())
    
    fig_error_results, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(20,6))
    results.boxplot(ax=axes[0])
    results.plot(ax=axes[1])

    
def save_results(experiment_id, model_config, model, history, dest_folder, results_i):
    
    # Save experiment summary -> _summary.json
    summary_filename = dest_folder + '_summary.json'
    model_as_json = model_config.as_json()
    model_as_json['experiment_id'] = experiment_id
    model_as_json['training_time'] = results_i['training_time']
    model_as_json['loss'] = results_i['loss']
    
    with open(summary_filename, 'w') as fp:
        json.dump(model_as_json, fp, indent=4)
    
    # Save model -> _model.h5
    model_filename = dest_folder + '_model.h5'
    model.save(model_filename)
    
    # Save results ->_results.json
    results_filename = dest_folder + '_results.json'
    with open(results_filename, 'w') as fp:
        json.dump(history.history, fp, sort_keys=True, indent=4)
    
    # Save result's plots
    plot_filename = dest_folder + '_plot.png'
    
    temp_figure = pyplot.figure(figsize=(20,6))
    
    temp_figure_1 = temp_figure.add_subplot(211)
    temp_figure_1.plot(history.history['acc'])
    temp_figure_1.plot(history.history['val_acc'])
#    temp_figure_1.set_title(f'model accuracy')
    temp_figure_1.set(ylabel='accuracy')
    temp_figure_1.set(xlabel='epoch')
    temp_figure_1.legend(['train', 'validation'], loc='upper left')

    temp_figure_2 = temp_figure.add_subplot(212)
    temp_figure_2.plot(history.history['loss'])
    temp_figure_2.plot(history.history['val_loss'])
#    temp_figure_2.set_title(f'model loss')
    temp_figure_2.set(ylabel='loss')
    temp_figure_2.set(xlabel='epoch')
    temp_figure_2.legend(['train', 'validation'], loc='upper left')
    
    temp_figure.savefig(plot_filename)
    
    pyplot.close()