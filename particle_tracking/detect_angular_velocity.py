# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:56:44 2020

@author: malopez
"""
import cv2
import pims
import glob
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.stats import mode
from utils import printp, createCircularMask, maskImage, angle_from_2D_points


def get_radial_brightness_peaks(video_path, row, min_r=15, max_r=20):
# ¿Usar tambien mínimos para aumentar la estadística?
    indice = int(row[0])
    frame_number = int(row[1])
    x = row[2]
    y = row[3]
    video = pims.Cine(video_path)
    frame = video.get_frame(frame_number-1)
    # Select only the portion of the frame corresponfing to current particle (x,y)
    # And mask so that only an annulus is visible (corresponding to the 'aspas')
    outer_mask = createCircularMask(800, 1280, center=[x,y], radius=max_r)
    inner_mask = createCircularMask(800, 1280, center=[x,y], radius=min_r)
    frame = maskImage(frame, outer_mask)
    frame = maskImage(frame, ~inner_mask)

    df = pd.DataFrame(frame)
    df['y'] = df.index
    df = pd.melt(df, id_vars=[('y')])
    df.rename(columns = {'variable':'x', 'value':'brightness'}, inplace = True)
    df = df[df.brightness!=0]
    df['brightness'] *= (255/frame.max())

    x_rel_to_center = df['x'] - x
    y_rel_to_center = df['y'] - y
    df['angles'] = angle_from_2D_points(x_rel_to_center.astype(int).values, y_rel_to_center.astype(int).values)
    df.sort_values('angles', inplace=True)

    angulos = df['angles'].values
    brillo = savgol_filter(df['brightness'], window_length=21, polyorder=3)

    picos_indice, picos_altura = find_peaks(brillo, width=10, prominence=5)
    picos = angulos[picos_indice]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=370)
    plt.plot(angulos, brillo)
    plt.scatter(angulos[picos_indice], brillo[picos_indice], c='r')

    fig, ax = plt.subplots(figsize=(7, 5), dpi=370)
    plt.imshow(frame[int(y)-40:int(y)+40, int(x)-40:int(x)+40])
    return (indice, picos[1:12]) # Devuelvo solo los picos centrales, a veces los extremos no se detectan bien



if __name__ == "__main__":

    GRADOS_POR_ASPA = 360/14

    folder = 'D:/'
    files = glob.glob(folder + '*Aspas*.cine') # List with all .cine files
    video_path = files[0]

    experiment_id = '87387719783ab1ba0d1d2008cd1f2ac5'
    data_file = f'D:/{experiment_id}_raw_trajectories.pkl'
    df = pd.read_pickle(data_file, compression='xz')
    df['indice'] = df.index

    # Función parcial, ahora solo acepta como entrada una lista, de la forma [indice, n_frame, x, y]
    partial_get_peaks = partial(get_radial_brightness_peaks, video_path, min_r=15, max_r=20)

    rows = df[['indice','frame','x','y']].values#[0:10000]
    N = len(rows)


    # Calculamos los angulos de los picos de brillo alrededor de las partículas.
    # Usamos múltiples procesadores
    N_CORES = mp.cpu_count()
    print(f'Computing angular brightness peaks using {N_CORES} cores \n')
    with mp.Pool(processes=12) as pool:
        dict_indices_picos = dict(list(tqdm(pool.imap(partial_get_peaks, rows), total=N)))

    df['picos'] = df['indice'].map(dict_indices_picos)


    #df = df.iloc[:10000]
    new_df = []
    for t in set(df['track']):
        sub = df[df['track']==t]
        desplazamientos_picos = np.diff(sub['picos'])

        angular_velocities = [np.mean(desp) for desp in desplazamientos_picos]
        # A veces al calcular la diferencia entre picos nos saltamos uno, ej: [-20.9, 4.8 , 4.9, 4.5, 5.3, 5.7, -20.6, 5.2, 5.0, 5.6 , -21.4 , 6.2, 4.6 , 5.7, 5.1, -20.5, 5.8, 5.8, 5.2, 5.3, -20.6, 5.7]
        # Como en estos casos nos estamos saltando un aspa hay que calcular el módulo con el angulo por aspa.
        # El ejemplo anterior pasaría a ser: [4.8, 4.8, 5. , 4.6, 5.4, 5.7, 5.1, 5.2, 5.1, 5.6, 4.3, 6.2, 4.6, 5.8, 5.1, 5.1, 5.8, 5.9, 5.3, 5.3, 5. , 5.7]
        signo, _ = mode(np.sign(angular_velocities))
        angular_velocities = np.mod(angular_velocities, 360/14)
        angular_velocities *= signo
        sub['angular_velocity'] = [0] + list(angular_velocities)
        new_df.append(sub)

    new_df = pd.concat(new_df, axis=0)[['frame','track','x','y','angular_velocity']]
    new_df.to_pickle(f'D:/{experiment_id}_raw_trajectories_angular.pkl', compression='xz')

    plt.hist(new_df.angular_velocity, bins=100)

    stds = []
    fig, ax = plt.subplots(figsize=(7, 5), dpi=370)
    for t in set(new_df.track):
        s = new_df[new_df.track==t]
        ax.plot(s.angular_velocity.values, lw=1, alpha=0.5)
        stds.append(np.std(s.angular_velocity.values))


# =============================================================================
#     plt.plot(stds)
# =============================================================================

