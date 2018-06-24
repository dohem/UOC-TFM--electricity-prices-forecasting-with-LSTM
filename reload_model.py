# -*- coding: utf-8 -*-
"""
Goal: show results of a selected model

Note: most of this file is a replica from modelLSTM.py
"""


#%%
# Dependencies
import os
import pandas as pd
import numpy as np
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.utils import plot_model
import json


# Other modules for TFM
import utils_modelLSTM as _utils
import tfm_config as _tfm



## Configuration
#######################
_dirs = _tfm.TFMDirectories()
data_filename = 'ready_2007_2018.csv'
show_model = True
plot_raw_data = False
plot_data = True

best_model_dir = _dirs.best_model_dir
#best_model_dir = '.\\models\\best_model_200'



#%% Load model and training's summary
#####################################
best_model_path = os.path.join(best_model_dir,'best_model.h5')
summary_path = os.path.join(best_model_dir,'summary.json')

model = load_model(best_model_path)
summary = json.load(open(summary_path))

hidden_layers = list()
for layer in summary['hidden_layers']:
    layer_model = _utils.ModelLayer(layer_type = layer['layer_type'], num_neurons = layer['num_neurons'])
    hidden_layers.append(layer_model)

model_config = _utils.LSTMModelConfig(
                    window_size = summary['window_size'],
                    data_train_ratio = summary['data_train_ratio'],
                    data_test_ratio = summary['data_test_ratio'],
                    data_validation_ratio = summary['data_validation_ratio'],
                    data_stationary = summary['data_stationary'],
                    hidden_layers = hidden_layers,
                    compile_loss_function = summary['compile_loss_function'],
                    compile_optimizer = summary['compile_optimizer'],
                    fit_epochs = summary['fit_epochs'],
                    fit_batch_size = summary['fit_batch_size']
            )



#%% Show model
###############
if (show_model):
    model_plot_path = os.path.join(best_model_dir,'model_plot.png')
    print(model.summary())
    plot_model(model, to_file=model_plot_path)




#%% Load & prepare data
########################

series = pd.read_csv(os.path.join(_dirs.ready_dir,data_filename))
    
# TODO: mover a data_preparation
# Columnas temporales
series['FixedHour'] = series['Hour'] - 1
series['DayHourStr'] = series.apply(
        lambda row: row["Day"] + str(row["FixedHour"]),
        axis=1
        )
series['DayHour'] = pd.to_datetime(series['DayHourStr'], format='%d/%m/%Y%H')
series = series.filter(['DayHour', 'Energy', 'Price'], axis=1)
series.shape #(8928, 3)

#%% Data preparation & split (train/validaiton/test)
######################

# transform data to be stationary
raw_values = series['Price'].values
if (model_config.data_stationary):
    adapted_values = _utils.difference(raw_values, 1)
else:
    adapted_values = raw_values

# Prepare previous datapoints as input for the given target (i.e. learn from a time window)
window_size = model_config.window_size
adapted_values = _utils.prepare_moving_window(adapted_values, window_size)


# Split available data in train/validation/test
# NOTE: validation_dataset is never used, as validation_slipt option for .fit() method is used instead
train_dataset, validation_dataset, test_dataset = _utils.dataset_splitting(model_config, adapted_values)

# transform the scale of the data
scaler, train_scaled, validation_scaled, test_scaled = _utils.scale_2(train_dataset, validation_dataset, test_dataset)

# for debugging only
train_scaled_old = train_scaled.copy()

# Reshape for LSTM
train_X, train_y, test_X, test_y = _utils.adapt_data_for_LSTM(train_scaled, test_scaled)

#%% Data visualization
######################
    
if (plot_raw_data):
    fig_data_vis = pyplot.figure(figsize=(20,6))
    fig_data_vis_1 = fig_data_vis.add_subplot(211)
    fig_data_vis_1.plot(series['Energy'].values)
    fig_data_vis_1.set(ylabel='Energy (MWh)')
    
    
    fig_data_vis_2 = fig_data_vis.add_subplot(212)
    fig_data_vis_2.plot(series['Price'].values)
    fig_data_vis_2.set(ylabel='Price (€/MWh)')
    
    fig_raw_vs_adapted = pyplot.figure(figsize=(20,6))
    fig_raw_vs_adapted_1 = fig_raw_vs_adapted.add_subplot(211)
    fig_raw_vs_adapted_1.plot(raw_values)
    fig_raw_vs_adapted_1.set(ylabel='raw')
    fig_raw_vs_adapted_2 = fig_raw_vs_adapted.add_subplot(212)
    fig_raw_vs_adapted_2.plot(adapted_values)
    fig_raw_vs_adapted_2.set(ylabel='adapted')


#%%
# Evaluate test data

loss = model.evaluate(test_X, test_y)


#%%
# Show predicted data
preds_scaled = model.predict(test_X)

test_X_reshaped = test_X.reshape(test_X.shape[0],test_X.shape[1])
preds_unscalable = pd.concat([pd.DataFrame(test_X_reshaped), pd.DataFrame(preds_scaled)], axis = 1)

preds = scaler.inverse_transform(preds_unscalable)[:,-1:]

actuals = test_dataset.loc[:,'TARGET'].values


initial_abs = raw_values[train_X.shape[0] + window_size]
preds_abs = list()
preds_abs.append(initial_abs)
for diff in preds:
    next_abs = preds_abs[-1] + diff
    preds_abs.append(next_abs[0])
preds_abs = Series(preds_abs)

actuals_abs = raw_values[(train_X.shape[0] + window_size):]


if (plot_data):
    fig_final_data_vis = pyplot.figure(figsize=(20,6))
    fig_final_data_vis_1 = fig_final_data_vis.add_subplot(211)
    fig_final_data_vis_1.plot(actuals)
    fig_final_data_vis_1.set(ylabel='Actual values')
    pyplot.gca().set_ylim([-80,40])
    
    fig_final_data_vis_2 = fig_final_data_vis.add_subplot(212)
    fig_final_data_vis_2.plot(preds)
    fig_final_data_vis_2.set(ylabel='Predicted values')
    pyplot.gca().set_ylim([-80,40])

    fig_final_abs = pyplot.figure(figsize=(20,3))
    pyplot.plot(actuals_abs, label='Actual values')
    pyplot.plot(preds_abs, label='Predicted values')
    pyplot.legend()


#%%
# Evaluate predicted data

mse_real = mean_squared_error(actuals,preds)
print('Mean square error', mse_real)

