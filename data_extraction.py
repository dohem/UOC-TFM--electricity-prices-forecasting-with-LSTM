# -*- coding: utf-8 -*-
"""
Goal: data extraction from raw downloaded files.

Raw downloaded files available at 1 - Datasets\1 - Raw\1 - Original, covering from 2006 to 2018 (Provisional)
"""



# Dependencies
import os
import sys
from shutil import copyfile
import zipfile
from io import BytesIO
import json

# Other modules for TFM
import tfm_config as _tfm



# Variables initialization
###########################
_dirs = _tfm.TFMDirectories()
_extraction = _tfm.TFMDataExtraction()
log = {}


# Configuration
###############
per = _extraction.per
aggr = _extraction.aggr
special_files = _extraction.special_files
console_debugging = False
years = list(range(2007,2018+1))  #2007:2018
provisional_years = [2017, 2018]


# Setting directories
######################
# TODO: esto sólo funciona si ya estás en el directorio adecuado, de lo contrario no se importa _tfm
if (os.getcwd() == _dirs.user_directory):
    os.chdir(_dirs.main_working_directory)
main_dir = os.getcwd()
raw_original_dir = _dirs.raw_original_dir
raw_selected_dir = _dirs.raw_selected_dir


# Definitions
###############

def extractDataFile(source_file, name, extension, y, new_name):
    if console_debugging:
        print('START - extractDataFile')
    
    ext = extension if extension else 'csv'
    path = raw_selected_dir[ext]
    if extension:
        filename = name + '.' + extension
    else:
        filename = name
    newfilename = new_name + '.' + ext
    if console_debugging:
        print('filename', filename)
        
    source_file.extract(filename, path)
    log[str(y)][ext] += 1
    
    if new_name:
        old_path = path + '\\' + filename
        new_path = path + '\\' + newfilename
        
        if console_debugging:
            print('rename', old_path, new_path)
            
        os.rename(old_path, new_path)
        
    if console_debugging:
        print(ext.upper() + ' - '+filename + ' to ' + newfilename)


# Loop over each file and extract the desired ones, saving them
###############################################################
#years = [] # to avoid hard work
for year in years:
    log[str(year)] = {'csv': 0, 'xls':0 , 'htm': 0}
    year_str = str(year)
    filename = ('Definitivo' if year not in provisional_years else 'Provisional') + '_' + year_str + '.zip'
    print('========')
    print('L0 - '+filename)
    print('========')
    file = zipfile.ZipFile(raw_original_dir + '\\' + filename)
    
    for month_filename in file.namelist():
        print('--------')
        print('L1 - '+month_filename)
        print('--------')
        month_str = month_filename[-6:-4]
        
        try:
            month_file_data = BytesIO(file.read(month_filename))
            month_file = zipfile.ZipFile(month_file_data)
            
            bucket_namelist = month_file.namelist()
            core_filename = f'PFM{per}_{aggr}' #_{year_str}{month_str}
            if console_debugging:
                print(bucket_namelist)
                print(core_filename)
            bucket_filename = [x for x in bucket_namelist if core_filename in x][0]
            
            bucket_file_data = BytesIO(month_file.read(bucket_filename))
            bucket_file = zipfile.ZipFile(bucket_file_data)
            
            bucket_files = bucket_file.namelist()
            bucket_range = [x for x in bucket_files if '.' not in x][0][-17:]
            final_filename = f'PFM{per}_{aggr}_{bucket_range}' #PFMHORAS_TOD_20180101_20180131
            if console_debugging:
                print(final_filename)
            
            for ext in [None, 'xls', 'htm']: #null is for csv
                newfn = bucket_range[:6] + '_selected'
                if console_debugging:
                    print('extrayendo...new file name', newfn)
                try:
                    extractDataFile(bucket_file, final_filename, ext, year, newfn)
                except:
                    print("ERROR extracting file", final_filename, ext)
                    print("Unexpected error:", sys.exc_info()[0])

            
        except:
            print("Error with", month_filename)
            print("Unexpected error:", sys.exc_info()[0])
        
    
        
print('Extracted ' + ', '.join([str(x) for x in years]))
print(json.dumps(log, indent=4, sort_keys=True))

# Special exportations
##################################

for file in special_files:
    for ext in ['csv', 'xls', 'htm']: #null is for csv
        if console_debugging:
            print('START - Special exportation', file)
            
        new_name = file[-17:-11] + '_selected'     
        if ext != 'csv':
            src = file + '.' + ext
        else:
            src = file
        newfilename = new_name + '.' + ext
        if console_debugging:
            print('newfilename', newfilename)
            
        copyfile(
            src,
            raw_selected_dir[ext] + '\\' + newfilename
        )
        
        print('Special exportation', file, newfilename)



