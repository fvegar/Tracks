# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:04:49 2019

import os
from functools import partial
from multiprocessing import Pool
import json
import glob
import hashlib
from zipfile import ZipFile
import trackpy as tp
import pims
from detect_blobs import detectCirclesVideo
from calculateVelocity import findVelocities, alternative_delete_short_trajectories, alternative_calculate_velocities
from utils import createCircularROI, reorder_rename_dataFrame, reset_track_indexes, present_in_folder, createRectangularROI


# =============================================================================
# folder = 'D:/serieDensidad 24-07-2019/'
# =============================================================================
# Diametro bola 78 px, diametro discos giratorios 79 px (78.5)
# ROI Bolas, camara cercana: [650, 400], R=390

def detect_particles_and_save_data(folder, file):
    # --- EXPERIMENTAL DETAILS ---
    vid = pims.Cine(file)

    original_file = file
    fps = vid.frame_rate
    shape = vid.frame_shape
    date = str(vid.frame_time_stamps[0][0])
    exposure = int(1000000*vid.all_exposures[0]) #in microseconds
    n_frames = vid.image_count
    recording_time = n_frames/fps
# =============================================================================
#         N = int(file.split('_')[-1].split('.')[0].split('n')[-1]) # Metodo guarrero y temporal
# =============================================================================
    N = int(file.split('_')[-3].split('N')[-1]) # Metodo guarrero y temporal
    power = int(file.split('_')[-2].split('p')[-1]) # Metodo guarrero y temporal
    if power>=100:
        power /= 10

    lights = 'luzLejana'
    camera_distance = 0.924 #in meters (bolas, cercana 0.535)
    pixel_ratio = 1950 #in px/meter, CAMBIARLO
    particle_diameter_px = 79
    particle_diameter_m = 0.0725
    particle_shape = 'rotating disk'
    system_diameter = 0.725 #in meters
    packing_fraction = N*(particle_diameter_m/system_diameter)**2
    ROI_center = [649, 392] #in pixels
    ROI_radius = 408
    # Hashing function to asign an unique id to each experiment
    # date+time should be specific enough to tell them apart
    hash_object = hashlib.md5(date.encode())
    experiment_id = str(hash_object.hexdigest())


    # SI YA HA SIDO PROCESADO NO TRABAJAR EN ESE ARCHIVO
    if present_in_folder(experiment_id, folder) == True:
        print('Experiment', experiment_id, 'already processed')
        return None # Exit function
    if os.path.getsize(file) != 31976589832:
        print('Corrupted file, skipping')
        return None
# =============================================================================
#         if folder[:-1]+'\\'+ str(experiment_id)+'_code.zip' in glob.glob(folder + '*.zip'):
#             print('already processed')
#             continue
# =============================================================================


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
# =============================================================================
#     # Finally we want tho freeze all the code used to track and process data
#     # into a single .zip file
#     codefiles = ['__init__.py',
#                  'detect_blobs.py',
#                  'calculateVelocity.py',
#                  'utils.py',
#                  'graphics.py',
#                  'export_santos_format.py',
#                  'stats.py',
#                  'analysis.py']
#     with ZipFile(associated_code, 'w') as myzip:
#         for f in codefiles:
#             myzip.write(os.path.join('D:/particleTracking/', f), arcname=f)
# =============================================================================


    # --- PARTICLE TRACKING ---
    # DEFAULT DETECTION PARAMETERS
    opening_kernel = 5 # Size of the kernel for getting rid of spurious features. Careful, large values affect static measuring error
    thresh = 20 # Threshold for binarization, should increase with exposure

    # General calibration, accounting for exposure
    if exposure == 300:
        thresh = 20
    elif exposure == 1000:
        thresh = 30
    elif exposure == 1500:
        thresh = 30
    elif exposure == 2500:
        thresh = 45

    # Specific calibration
    if 'CamaraCercana' not in file:
        opening_kernel = 20
    if ('CamaraCercana' not in file) and ('exposicion300' in file):
        thresh = 18
        opening_kernel = 25
    if 'Foco' in file:
        thresh += 5
        opening_kernel = 25


    # --CIRCLE DETECTION--
    print(file)
    circles = detectCirclesVideo(file, thresh=thresh, display_intermediate_steps=False,
                                 opening_kernel=opening_kernel)
    # A veces se captan partículas inexistentes muy cerca de los bordes del sistema. (por brillos o reflejos)
    # Por ello hay que eliminar todo aquello cuyo centro este a menos de n pixeles del borde,
    # donde n es el radio de la partícula menos un par de pixeles. Este proceso no tiene que ver con el de la ROI, en este caso 15
    circles = createCircularROI(circles, ROI_center, ROI_radius-15)

    # TRAJECTORY LINKING
    traj = tp.link_df(circles, 5, memory=0)
    traj = reorder_rename_dataFrame(traj) # Always run after trackpy

    # VELOCITY DERIVATION
    vels = findVelocities(traj)
    vels = reset_track_indexes(vels) # Always run after deleting traj or calculate_vels, this fills voids

    #SAVING RAW DATA
    circles.to_pickle(os.path.join(folder, str(experiment_id)+'_raw_data.pkl'), compression='xz')
    traj.to_pickle(os.path.join(folder, str(experiment_id)+'_raw_trajectories.pkl'), compression='xz')
    vels.to_pickle(os.path.join(folder, str(experiment_id)+'_raw_velocities.pkl'), compression='xz')



    # CREATION OF REGION OF INTEREST (circular)
    roi_data = createCircularROI(circles, ROI_center, ROI_radius)
    roi_traj = tp.link_df(roi_data, 5, memory=0)
    roi_traj = reorder_rename_dataFrame(roi_traj) # Always run after trackpy
    roi_traj = reset_track_indexes(roi_traj) # Always run after deleting traj or calculate_vels, this fills voids
    # DERIVE VELOCITIES
    roi_vels = alternative_calculate_velocities(roi_traj, n=1, use_gradient=False)
    # DELETING SHORT TRAJECTORIES
    roi_vels = alternative_delete_short_trajectories(roi_vels, minimumFrames=10)
    roi_vels = reset_track_indexes(roi_vels) # Always run after deleting traj or calculate_vels, this fills voids
    # roi_vels = deleteShortTrajectories(roi_vels, minimumFrames=10)
    # SAVING DATA
    roi_data.to_pickle(os.path.join(folder, str(experiment_id)+'_roi_data.pkl'), compression='xz')
    roi_traj.to_pickle(os.path.join(folder, str(experiment_id)+'_roi_trajectories.pkl'), compression='xz')
    roi_vels.to_pickle(os.path.join(folder, str(experiment_id)+'_roi_velocities.pkl'), compression='xz')
    # SAVING DATA 'SANTOS' FORMAT
    roi_vels.to_csv(os.path.join(folder, str(experiment_id)+'_pos_vel_ppp.dat'), sep='\t', header=True, index=False)




    # CREATION OF REGION OF INTEREST (rectangular)
    roi_data = createRectangularROI(circles, [250,50], 750, 600)
    roi_traj = tp.link_df(roi_data, 5, memory=0)
    roi_traj = reorder_rename_dataFrame(roi_traj) # Always run after trackpy
    roi_traj = reset_track_indexes(roi_traj) # Always run after deleting traj or calculate_vels, this fills voids
    # DERIVE VELOCITIES
    roi_vels = alternative_calculate_velocities(roi_traj, n=1, use_gradient=False)
    # DELETING SHORT TRAJECTORIES
    roi_vels = alternative_delete_short_trajectories(roi_vels, minimumFrames=10)
    roi_vels = reset_track_indexes(roi_vels) # Always run after deleting traj or calculate_vels, this fills voids
    # roi_vels = deleteShortTrajectories(roi_vels, minimumFrames=10)
    # SAVING DATA
    roi_data.to_pickle(os.path.join(folder, str(experiment_id)+'_rect_roi_data.pkl'), compression='xz')
    roi_traj.to_pickle(os.path.join(folder, str(experiment_id)+'_rect_roi_trajectories.pkl'), compression='xz')
    roi_vels.to_pickle(os.path.join(folder, str(experiment_id)+'_rect_roi_velocities.pkl'), compression='xz')



if __name__ == "__main__":

    folder = 'D:/'

    files = glob.glob(folder + '*Aspas*.cine') # List with all .cine files

    func = partial(detect_particles_and_save_data, folder) # Partial function that only accept a file

    pool = Pool(processes=2)
    pool.map(func, files)
    pool.close()
    pool.join()

    #for file in files:
