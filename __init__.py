# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:04:49 2019

@author: malopez
"""
import os
import json
import hashlib
from zipfile import ZipFile
import numpy as np
import pandas as pd
import trackpy as tp
import pims
from detect_blobs import detectCirclesVideo
from calculateVelocity import findVelocities, deleteShortTrajectories, alternative_delete_short_trajectories
from utils import createCircularROI, reorder_rename_dataFrame, reset_track_indexes
from graphics import densityPlot


if __name__ == "__main__":
  
    folder = 'D:/videos 03-05-2019/'
    files = ['pelotas_luzLejana_CamaraCercana_exp1500_fps250_p45_n170',]
    extension = '.cine'
    
    for file in files:
        path = os.path.join(folder, file) + extension
        
        # --- EXPERIMENTAL DETAILS ---
        vid = pims.Cine(path)
        
        original_file = path
        fps = vid.frame_rate
        shape = vid.frame_shape
        date = str(vid.frame_time_stamps[0][0])
        exposure = vid.all_exposures[0] #in seconds
        n_frames = vid.image_count
        recording_time = n_frames/fps
        N = 170 # Number of particles
        power = 45 # Power of the fan
        lights = 'luzLejana'
        camera_distance = 0.535 #in meters
        pixel_ratio = 1950 #in px/meter
        particle_diameter_px = 78
        particle_diameter_m = 0.04
        particle_shape = 'sphere'
        system_diameter = 0.725 #in meters
        packing_fraction = N*(particle_diameter_m/system_diameter)**2
        ROI_center = [650,400] #in pixels
        ROI_radius = 390        
        # Hashing function to asign an unique id to each experiment
        # date+time should be specific enough to tell them apart
        hash_object = hashlib.md5(date.encode())
        experiment_id = str(hash_object.hexdigest())

        associated_code = os.path.join(folder, str(experiment_id)+'_code.zip')
        
        # I create a dictionary to store all this properties in a .txt file
        experiment_properties_dict = {}
        for i in ('experiment_id', 'original_file', 'date', 'shape', 'fps', 'exposure',
                  'n_frames', 'recording_time', 'camera_distance', 'pixel_ratio',
                  'particle_diameter_px', 'N', 'particle_shape', 'particle_diameter_m',
                  'system_diameter', 'packing_fraction', 'lights', 'power', 'associated_code',
                  'ROI_center', 'ROI_radius'):
            experiment_properties_dict[i] = locals()[i]
        # Save to a file
        with open(os.path.join(folder, str(experiment_id)+'_experiment_info.txt'), 'w') as f:
            json.dump(experiment_properties_dict, f, indent=0)
        # Finally we want tho freeze all the code used to track and process data
        # into a single .zip file
        codefiles = ['__init__.py',
                     'detect_blobs.py',
                     'calculateVelocity.py',
                     'utils.py',
                     'graphics.py',
                     'export_santos_format.py',
                     'watershedContours.py']
        with ZipFile(associated_code, 'w') as myzip:
            for f in codefiles:   
                myzip.write(os.path.join('D:/particleTracking/',f), arcname=f)
        
        
        # --- PARTICLE TRACKING ---
        # DEFAULT DETECTION PARAMETERS
        opening_kernel = 5 # Size of the kernel for getting rid of spurious features. Careful, large values affect static measuring error
        thresh = 20 # Threshold for binarization, should increase with exposure
        
        # General calibration, accounting for exposure
        if ('exposicion300' in file):
            thresh = 20
        elif ('exposicion1000' in file):
            thresh = 30
        elif ('exposicion1500' in file):
            thresh = 30
        elif ('exposicion2500' in file):
            thresh = 45
        
        # Specific calibration
        if ('CamaraCercana' not in file):
            opening_kernel = 20
        if ('CamaraCercana' not in file) and ('exposicion300' in file):
            thresh = 18
            opening_kernel = 25
        if ('Foco' in file):
            thresh += 5
            opening_kernel = 25


        # --CIRCLE DETECTION--
        circles = detectCirclesVideo(path, thresh=thresh, display_intermediate_steps=False, 
                                     opening_kernel=opening_kernel)
        # TRAJECTORY LINKING
        traj = tp.link_df(circles, 5, memory=0)
        traj = reorder_rename_dataFrame(traj) # Always run after trackpy
        traj = reset_track_indexes(traj) # Always run after trackpy or calculate_vels, this fills voids
        
        # VELOCITY DERIVATION
        vels = findVelocities(traj)
        
        #SAVING RAW DATA
        circles.to_pickle(os.path.join(folder, str(experiment_id)+'_raw_data.pkl'), compression='xz')
        traj.to_pickle(os.path.join(folder, str(experiment_id)+'_raw_trajectories.pkl'), compression='xz')
        vels.to_pickle(os.path.join(folder, str(experiment_id)+'_raw_velocities.pkl'), compression='xz')
        
        
        # CREATION OF REGION OF INTEREST
        roi_data = createCircularROI(circles, ROI_center, ROI_radius)
        roi_traj = tp.link_df(roi_data, 5, memory=0)
        roi_traj = reorder_rename_dataFrame(roi_traj) # Always run after trackpy
        # DELETING SHORT TRAJECTORIES
        roi_traj = alternative_delete_short_trajectories(roi_traj, minimumFrames=10)
        roi_traj = reset_track_indexes(roi_traj) # Always run after trackpy or calculate_vels, this fills voids
        # DERIVE VELOCITIES
        roi_vels = findVelocities(roi_traj)
        roivels = reset_track_indexes(roi_vels) # Always run after trackpy or calculate_vels, this fills voids
        # roi_vels = deleteShortTrajectories(roi_vels, minimumFrames=10)
        # SAVING DATA
        roi_data.to_pickle(os.path.join(folder, str(experiment_id)+'_roi_data.pkl'), compression='xz')
        roi_traj.to_pickle(os.path.join(folder, str(experiment_id)+'_roi_trajectories.pkl'), compression='xz')
        roi_vels.to_pickle(os.path.join(folder, str(experiment_id)+'_roi_velocities.pkl'), compression='xz')
        # SAVING DATA 'SANTOS' FORMAT
        roi_vels.to_csv(os.path.join(folder, str(experiment_id)+'_pos_vel_ppp.dat'), sep='\t', header=True, index=False)
        
        
        # --ACTUAL DATA ANALISYS--
        
        # from utils import play_video_with_labels
        # play_video_with_labels(path, traj, list_of_particles_to_track='all')
        
        # READING DATA
        data = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_raw_data.pkl'), compression='xz')
        roi_traj = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_roi_trajectories.pkl'), compression='xz')
        # roi_vels already has deleted traj with less than 100 datapoints
        roi_vels = pd.read_pickle(os.path.join(folder, str(experiment_id)+'_roi_velocities.pkl'), compression='xz')
        
        
        print('Density plot including all data:')
        densityPlot(data, nbins=80)
        print('Density plot inside ROI:')
        densityPlot(roi_traj, nbins=80)
        print('Density plot inside ROI without short traj:')
        densityPlot(roi_vels, nbins=80)