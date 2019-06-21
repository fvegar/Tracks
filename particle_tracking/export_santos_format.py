# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:58:08 2019

@author: malopez
"""
import os
import json
from zipfile import ZipFile
import pandas as pd


folder = 'D:/pruebas 10-06-2019/'
experiment_ids = ['1143d64b2fa1cea0c5f81f264f357a70',]

for id in experiment_ids:

    data = pd.read_csv(os.path.join(folder, str(id)+'_pos_vel_ppp.dat'), sep='\t')
    # Santos doesn't need these two columns
    if 'vx' in data.columns:
        data = data.drop(['vx','vy'], 1)
    # Reorder in Santos format
    data = data[['x','y','frame','track']]
    
    # Create zip file to store all the little 'one-trajectory' data files
    santos_zipped_data = ZipFile(os.path.join(folder, str(id)+'_santos_format.zip'), 'w')
    
    
    track_lenghts_list = []
    for part in set(data.track):
        file = 'datosParticulaFile'+str(part)+'.dat'
        filePath = os.path.join(folder, file)
        print(filePath)
        
        sub_data = data[data.track == part]
        sub_data.to_csv(filePath, sep='\t', header=False, index=False)
        # Include in zip file
        santos_zipped_data.write(filePath, arcname=file)
        
        lenght = len(sub_data)
        track_lenghts_list.append([part,lenght])
    
    l = pd.DataFrame(track_lenghts_list, columns=['trajectory','lenght'])
    l = l.sort_values('lenght', ascending=False)
    l = l.reset_index(drop=True)
    
    tracks_and_lenghts = str(track_lenghts_list).replace('[','{').replace(']','}')
    
    # Save matrix and experiment information to .zip
    with open(os.path.join(folder, str(id)+'_track_lenghts_matrix.txt'), 'w') as f:
        json.dump(tracks_and_lenghts, f, indent=0)
    santos_zipped_data.write(os.path.join(folder, str(id)+'_track_lenghts_matrix.txt'),
                             arcname=str(id)+'_track_lenghts_matrix.txt')
    santos_zipped_data.write(os.path.join(folder, str(id)+'_experiment_info.txt'),
                             arcname=str(id)+'_experiment_info.txt')
    # Closing zip file
    santos_zipped_data.close()
    
    print(tracks_and_lenghts)