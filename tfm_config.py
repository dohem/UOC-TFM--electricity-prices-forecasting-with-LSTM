# -*- coding: utf-8 -*-
"""
Goal: centralize main configurations of the project

Contents on this file:
    - Class TFMDirectories -> info about used directories
    - Class TFMDataExtraction -> info about data extraction
"""

# Dependencies
#import os

# All paths are relative to the main_working_directory
class TFMDirectories:
    # Main working directory
    user_directory = 'C:\\Users\\agabarro'
    main_working_directory = '.\\Dropbox\\Carpeta del máster\\TFM\\Develop\\2 - Code'
    main_working_directory_absolute = 'C:\\Users\\agabarro\\Dropbox\\Carpeta del máster\\TFM\\Develop\\2 - Code'
    
    # Dataset directories
    raw_original_dir = '..\\1 - Datasets\\1 - Raw\\1 - Original'
    raw_selected_dir = {
       'csv': '..\\1 - Datasets\\1 - Raw\\2 - Selected\\csv',
       'xls': '..\\1 - Datasets\\1 - Raw\\2 - Selected\\xls',
       'htm': '..\\1 - Datasets\\1 - Raw\\2 - Selected\\htm'
    }
    ready_dir = '..\\1 - Datasets\\2 - Ready'
    
    models_dir = '.\\models'
    
    best_model_dir = '.\\models\\best_model'
    
#    def move_to_main_working_directory(self):
#        if (os.getcwd() != self.main_working_directory_absolute):
#                os.chdir(self.main_working_directory)
#                print('cambiado!')

class TFMDataExtraction:
    _dirs = TFMDirectories()
    
    provisional_years = [2017, 2018]
    per = 'HORAS' # Periodo escogido entre los datos disponibles
    aggr = 'TOD'  # Agregación por Unidad de Adquisicón entre los datos disponibles
    
    special_files = [
        _dirs.raw_original_dir + '\\2007 Agosto del Provisional\\PFMHORAS_TOD_20070801_20070831',
        _dirs.raw_original_dir + '\\2010 Agosto del Provisional\\PFMHORAS_TOD_20100801_20100831'
    ]
    
