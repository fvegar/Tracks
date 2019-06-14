# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:00:36 2019

@author: malopez
"""

import os
import json
import cv2
import pandas as pd
from utils import select_tracks_by_lenght, play_video_with_labels, maskImage, createCircularMask
from graphics import plot_trajectories_by_lenght


def mod_play_video_with_labels(videoPath, trajectories, list_of_particles_to_track='all', mean_radius=39, mask_ROI=False):
    video = cv2.VideoCapture(videoPath)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('C:/Users/malopez/Desktop/prueba150_m.avi',fourcc, 30.0, (1280,800))

    f=1
    while(video.isOpened()):
        # Leemos el frame actual y lo asignamos a la variable frame
        frameExists, frame = video.read()
        
        # We extract the data corresponding to that frame
        sub_data = trajectories[trajectories['frame']==f]
        
        if list_of_particles_to_track=='all':
            for index, row in sub_data.iterrows():
                # Draw the outer circle [(x,y), radius, rgb]
                cv2.circle(frame, (int(row['x']),int(row['y'])), mean_radius, (127, 0, 255), 2)
                # Annotate track label
                cv2.putText(frame, str(int(row['track'])), (int(row['x']),int(row['y'])), 
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255,150,250), 2)
        else:
            for index, row in sub_data.iterrows():
                if (row['track'] in list_of_particles_to_track):
                    # Draw the outer circle [(x,y), radius, rgb]
                    cv2.circle(frame, (int(row['x']),int(row['y'])), mean_radius, (127, 0, 255), 2)
                    # Annotate track label
                    cv2.putText(frame, str(int(row['track'])), (int(row['x']),int(row['y'])), 
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255,150,250), 2)


        if mask_ROI==True:
            frame = maskImage(frame, createCircularMask(800,1280, center=[650,400], radius=390))
        # Mostramos en pantalla el video (esperando 3ms entre frame y frame) 
        # hasta que llega al final o se pulsa la tecla q
        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # Incrementing the count of frames
        f+=1
        
    # Cerramos el stream de video y las ventanas abiertas
    video.release()
    out.release()
    cv2.destroyAllWindows()

Ns_to_play = [150,]
if __name__ == "__main__":
    min_track_lenght = 1
    max_track_lenght = 200
    folder = 'D:/serieDensidad 21-05-2019/'
    files = []
    files += [file for file in os.listdir(folder) if file.endswith('_roi_trajectories.pkl')]
    
    for file in files:      
        # Read information:
        experiment_id = file.split('_')[0]
        info_file = os.path.join(folder, experiment_id+'_experiment_info.txt')
        with open(info_file) as f:
            jsonstr = json.load(f)
            
        info = pd.io.json.json_normalize(jsonstr) 
        N = info.N[0]
        video_path = info.original_file[0]
        
        if N in Ns_to_play:
            traj = pd.read_pickle(os.path.join(folder, file), compression='xz')
            
            print(str(info.packing_fraction[0]))
            plot_trajectories_by_lenght(traj, min_lenght=1, max_lenght=250)
            plot_trajectories_by_lenght(traj, min_lenght=250, max_lenght=10000)
            plot_trajectories_by_lenght(traj, min_lenght=10000, max_lenght=25000)
            
            tags = select_tracks_by_lenght(traj, min_lenght=250, max_lenght=10000)
            mod_play_video_with_labels(video_path, traj, list_of_particles_to_track=tags, mask_ROI=False)
