# -*- coding: utf-8 -*-
"""
Goal: manage training sessions
"""


#%%
# Dependencies
import datetime
import csv
import os

# Other modules for TFM
import utils_modelLSTM as _utils
import modelLSTM as _modelLSTM
import tfm_config as _tfm



#%% Configuration of the session
_dirs = _tfm.TFMDirectories()


#%% Define experiments

# Fixed values
experiment_repetitions = 3

data_train_ratio = 0.7
data_test_ratio = 0.3
data_validation_ratio = 0
data_stationary = True
compile_loss_function = "mse"
compile_optimizer = "adam"

# Ranges
#window_size_range = range(10, 51, 10)
window_size_range = [200]
#window_size_range = [50, 100, 150, 200, 250]
#window_size_range = [150, 200, 250]

#fit_epochs_range = range(1, 101, 20) # más épocas?
#fit_epochs_range = [5]
fit_epochs_range = [1000]

#fit_batch_size = [50, 512]
#fit_batch_size = [1, 20, 50, 512]
#fit_batch_size = [64, 128, 256, 512, 1024]
fit_batch_size = [1024]
#fit_batch_size = [256, 512, 1024]





# Architectures
# 2+2 -> 75 params
# 4+4 -> 245 params
# 8+8-> 873 params
# 8+4-> 533 params
# 12+12 -> 1885 params
# 12+4 -> 949 params
#hidden_layers_01 = [
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 2),
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 2)
#        ]

#hidden_layers_01 = [
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 4),
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 4)
#        ]

#hidden_layers_01 = [
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 8),
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 8)
#        ]
#
hidden_layers_01 = [
        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 8),
        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 4)
        ]
#
#hidden_layers_01 = [
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 12),
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 12)
#        ]

#hidden_layers_01 = [
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 12),
#        _utils.ModelLayer(layer_type = 'LSTM', num_neurons = 4)
#        ]


# Model configurations
models = list()
for window_size_i in window_size_range:
    for fit_epochs_i in fit_epochs_range:
        for fit_batch_i in fit_batch_size:

            model_i = _utils.LSTMModelConfig(
                    window_size = window_size_i,
                    data_train_ratio = data_train_ratio,
                    data_test_ratio = data_test_ratio,
                    data_validation_ratio = data_validation_ratio,
                    data_stationary = data_stationary,
                    hidden_layers = hidden_layers_01,
                    compile_loss_function = compile_loss_function,
                    compile_optimizer = compile_optimizer,
                    fit_epochs = fit_epochs_i,
                    fit_batch_size = fit_batch_i
            )
            models.append(model_i)


# Plot configurations
plot_config_01 = _utils.ExperimentPlotConfig(
        raw_data_visualization = False,
        raw_vs_adapted = False,
        actuals_vs_predicted = False
        )



experiments = list()
for model in models:
    experiment = {
                'repetitions': experiment_repetitions,
                'model_configuration': model,
                'plot_configuration': plot_config_01,
            }
    
    experiments.append(experiment)
    

#%% Train the model, saving the results
 
# TODO: print key values of experiment configuration
    
results = list()

safe_datetime = str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')) #.replace(':','.')
    
for experiment in experiments:
    # assign experiment id
    training_session_id = safe_datetime + ' - Training'
    experiment_id = safe_datetime \
                    + '_' \
                    + '_Rep' + str(experiment['repetitions']) \
                    + '_Win' + str(experiment['model_configuration'].window_size) \
                    + '_Epo' + str(experiment['model_configuration'].fit_epochs) \
                    + '_Bat' + str(experiment['model_configuration'].fit_batch_size)
                    
    try:
        # execute training
        output_results = _modelLSTM.run_experiment(experiment_id, experiment)
        
        # Output: duration, loss, mse -> guardarlo junto con trainint session_id
        for res in output_results:
            results.append(res)
            
    except:
        print(f'Experiment {experiment_id} FAILED')
        
    

#print('resutls', results)

dest_folder = os.path.join(_dirs.models_dir, training_session_id)
f = open(dest_folder + ' - output.csv', 'w', newline='')
with f:
    fnames = ['experiment_id_rep', 'training_time', 'loss']
    writer = csv.DictWriter(f, fieldnames=fnames)    
    writer.writeheader()
    
    for res in results:
       writer.writerow(res) 

