# -*- coding: utf-8 -*-
"""
Goal: data preparation
"""

# Dependencies
import os
import csv
import pandas as pd

# Other modules for TFM
import tfm_config as _tfm

# Initialize variables
######################
years_df = {}
_dirs = _tfm.TFMDirectories()

#energy_df = pd.DataFrame() # All energy registries, per hour
#prices_df = pd.DataFrame() # All prices registries, per hour
data_df = pd.DataFrame()

# Configuration
######################
console_debugging = False
years = list(range(2007,2018+1))  #2007:2018
ready_path = _dirs.ready_dir
ready_filename = 'ready_2007_2018.csv'
extracted_components = {
    'energy': { 'name': 'Energía final', 'units': 'MWh', 'colIndex': 2, 'colName': 'Energía final MWh'  },
    'prices': { 'name': 'Precio en el Mercado Diario', 'units': '€/MWh', 'colIndex': 3, 'colName': 'Mercado diario/MWh'  },
}

# Definitions
#############
def detect_skippable_rows(filepath):
    result = 0
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            result += 1
            if '01/' in row[0]:
                break
    result -= 1
    return result
    

# Read files
######################
for year in years:
    if console_debugging:
        print('reading', year)
    path = _dirs.raw_selected_dir['csv']
    name = f'{year}01_selected.csv'
    
    skiprows = detect_skippable_rows(os.path.join(path,name))
    years_df[str(year)] = pd.read_csv(
            os.path.join(path,name), encoding='latin1', delimiter=';',
            skiprows=skiprows, header=None
    )
    if console_debugging:
        print(years_df[str(year)].describe())
        

# Concatenate data
##################
for df in years_df.values():   
    temp_df = df.iloc[:,:4]
    temp_df.columns = ['Day', 'Hour', 'Energy', 'Price']
    
    data_df = pd.concat([data_df, temp_df])
    data_df.columns = ['Day', 'Hour', 'Energy', 'Price']
    

if console_debugging:
    print(data_df)
    print(data_df.head())
    print(data_df.shape)
    
    
# Save as csv
#############
data_df.to_csv(os.path.join(ready_path,ready_filename))
print('Saved file')
