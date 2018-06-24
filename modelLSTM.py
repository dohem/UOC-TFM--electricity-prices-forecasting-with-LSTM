# -*- coding: utf-8 -*-
"""
Goal: manage all the logic about running an experiment
"""
"""
TODO:
    - move things to data_preparation + rerun that
    
References:
    - [0] https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
    - [1] https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
"""



#%%
# Dependencies
import os
import pandas as pd
import time
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# Other modules for TFM
import utils_modelLSTM as _utils
import tfm_config as _tfm




#%%

## Initialize
#######################
#_dirs = _tfm.TFMDirectories()
#
#
## Configuration
#######################
#data_filename = 'ready_2007_2018.csv'
#
#model_approach = 0 # https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
##model_approach = 1 # https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

#hidden_layers = [
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 2),
##        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 4),
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 2)
#        ]
#
#model_config_01 = _utils.LSTMModelConfig(
#        window_size = 50,
#        data_train_ratio = 0.7,
#        data_test_ratio = 0.3,
#        data_validation_ratio = 0,
#        data_stationary = True,
##            input_layer,
#        hidden_layers = hidden_layers,
##            output_layer,
#        compile_loss_function = "mse",
#        compile_optimizer = "adam",
#        fit_epochs = 5,
#        fit_batch_size = 50
#        )
#
#plot_config_01 = _utils.ExperimentPlotConfig(
#        raw_data_visualization = True,
#        raw_vs_adapted = True,
#        actuals_vs_predicted = True
#        )
#
#experiments = [
#        {
#            'repetitions': 1,
#            'model_configuration': model_config_01,
#            'plot_configuration': plot_config_01,
#        }
#    ]

## TODO: automate experiments
##for exp in experiments:
#exp = experiments[0]
#model_config = exp['model_configuration']
#plot_config = exp['plot_configuration']
#experiment_id = 'hola'
#
#output_results = list()






def run_experiment(
        experiment_id,
        experiment
        ):
    
    # Configuration
    _dirs = _tfm.TFMDirectories()
    data_filename = 'ready_2007_2018.csv'
    model_approach = 0
    
    
    model_config = experiment['model_configuration']
    plot_config = experiment['plot_configuration']
    exp = experiment
    
    output_results = list()

    
    #%% Data loading
    ######################
    
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
    
    
    #%% Data visualization
    ######################
    
    if (plot_config.raw_data_visualization):
        fig_data_vis = pyplot.figure(figsize=(20,6))
        fig_data_vis_1 = fig_data_vis.add_subplot(211)
        fig_data_vis_1.plot(series['Energy'].values)
        #subplot_1_1.set_title('Energy')
        fig_data_vis_1.set(ylabel='Energy (MWh)')
        
        
        fig_data_vis_2 = fig_data_vis.add_subplot(212)
        fig_data_vis_2.plot(series['Price'].values)
        #subplot_1_2.set_title('Price')
        fig_data_vis_2.set(ylabel='Price (€/MWh)')
    
    
    
    #%% Data preparation & split (train/validaiton/test)
    ######################
    
    # JUST FOR PRICES
    
    # transform data to be stationary
    raw_values = series['Price'].values
    #print(model_config.data_stationary)
    if (model_config.data_stationary):
        adapted_values = _utils.difference(raw_values, 1)
    else:
        adapted_values = raw_values
        
    if (plot_config.raw_vs_adapted): 
        fig_raw_vs_adapted = pyplot.figure(figsize=(20,6))
        fig_raw_vs_adapted_1 = fig_raw_vs_adapted.add_subplot(211)
        fig_raw_vs_adapted_1.plot(raw_values)
        fig_raw_vs_adapted_1.set(ylabel='raw')
        fig_raw_vs_adapted_2 = fig_raw_vs_adapted.add_subplot(212)
        fig_raw_vs_adapted_2.plot(adapted_values)
        fig_raw_vs_adapted_2.set(ylabel='adapted')
    
    
    # Prepare previous datapoints as input for the given target (i.e. learn from a time window)
    window_size = model_config.window_size
    adapted_values = _utils.prepare_moving_window(adapted_values, window_size)
    
    
    # Split available data in train/validation/test
    # NOTE: validation_dataset is never used, as validation_slipt option for .fit() method is used instead
    train_dataset, validation_dataset, test_dataset = _utils.dataset_splitting(model_config, adapted_values)
    #print(len(train_dataset));
    #print(len(validation_dataset));
    #print(len(test_dataset));
    #print(len(adapted_values))
    
    # transform the scale of the data
    scaler, train_scaled, validation_scaled, test_scaled = _utils.scale_2(train_dataset, validation_dataset, test_dataset)
    
    # for debugging only
    train_scaled_old = train_scaled.copy()
    
    
    #%% Adapt data for model
    
    # TODO: add validation data
    if model_approach == 0:
        
#        # Shuffle and reshape
#        print('SHUFFLE', train_scaled.shape)
#        train_scaled = shuffle(train_scaled)
#        print('SHUFFLE', train_scaled.shape)
    
        # Reshape for LSTM
        train_X, train_y, test_X, test_y = _utils.adapt_data_for_LSTM(train_scaled, test_scaled)
    
      
    #%% Train model
    ######################
    
    start = time.time()
    
    if model_approach == 0:
        
        performance_loss = list()
        for r in range(exp['repetitions']):
            
            # Compose model
            lstm_model = _utils.compose_model_2(model_config)
            
            # Fit model
            lstm_model, history, training_time = _utils.fit_model_2(model_config, lstm_model, train_X, train_y, verbose = 1)
            
            experiment_id_rep = experiment_id + f'___rep_{r+1}-de-{exp["repetitions"]}'
            dest_folder = os.path.join(_dirs.models_dir, experiment_id_rep)
            
        
            # Evaluate the model
            loss_i = lstm_model.evaluate(test_X, test_y)
            performance_loss.append(loss_i)
            
            
            # Doing a prediction on all the test data at once
#            predictions = lstm_model.predict(test_X)
                  
            results_i = {
                    'experiment_id_rep': experiment_id_rep,
                    'training_time': training_time,
                    'loss': loss_i[0],
                    }
            output_results.append(results_i)
                
            _utils.save_results(experiment_id_rep, model_config, lstm_model, history, dest_folder, results_i)
            
        
        for r in range(exp['repetitions']):
            exp_num = r + 1
            loss = str(performance_loss[r])
            print(f'Exp. {str(exp_num)} - Loss: {loss}')
    
    
    
    
    # Report total duration
    exp_duration = time.time() - start
    exp_dur_h, exp_dur_m, exp_dur_s = _utils.convert_seconds(exp_duration)
    print(f'-- Total experiments time : {exp_dur_h}h {exp_dur_m}m {exp_dur_s}s ({exp["repetitions"]} experiments,  {str(round(exp_duration, 3))}s)')
    
    
    

    #%% Summarize results
    ######################
        
    # Results summary adapted from [1]
    #_utils.results_summary_adapted_from_1(error_scores, error_texts)



    return output_results















#%%
# EXTRA CELL
######################

