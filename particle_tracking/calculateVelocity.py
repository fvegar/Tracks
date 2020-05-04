# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:05:37 2019

@author: malopez
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, buttord, butter, filtfilt
from utils import printp, reset_track_indexes

def findVelocities(trajectories):
    """ This function admits a 'trajectories' dataframe as an input
        (usually from trackpy) and returns another dataframe where
        instant velocities are included """

    n_tracks = len(set(trajectories.track))
    col_names = ['frame', 'track', 'x', 'y', 'vx', 'vy']
    # Creating an empty dataframe to store results
    data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
    
    for item in set(trajectories.track):      
        sub = trajectories[trajectories.track==item]
        
        if sub.shape[0]<=2:
            #Para obviar los casos en los que solo hay una o dos filas de datos
            pass
        else:    
            printp('Deriving velocities for track: '+ str(item) + '/'+ str(n_tracks))
            dvx = pd.DataFrame(np.gradient(sub.x), columns=['vx',])
            dvy = pd.DataFrame(np.gradient(sub.y), columns=['vy',])
        
            new_df = pd.concat((sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True), 
                                sub.x.reset_index(drop=True), sub.y.reset_index(drop=True), dvx, dvy), 
                                axis=1, names=col_names, sort=False)
            data = pd.concat((data, new_df), axis=0)
        
    # This is to get rid of the first 'np.zeros' row and to reset indexes
    data = data.reset_index(drop=True)
    data = data.drop(0)
    data = data.reset_index(drop=True)
        
    return data



def alternative_calculate_velocities(trajectories, n=1, use_gradient=False):

    n_tracks = len(set(trajectories.track))

    # Con gradient solo se puede usar n=1
    if use_gradient==True:
        out = []
        for t in set(trajectories.track):
            sub = trajectories[trajectories.track == t]
            try:
                vxvy = np.gradient(sub[['x','y']].values, axis=0)
                sub['vx'] = vxvy[:,0]
                sub['vy'] = vxvy[:,1]
        
                out.append(sub)
                printp('Deriving velocities for track: '+ str(t+1) + '/'+ str(n_tracks))
            except:
                pass

    else:
        if n==1:
            out = []
            for t in set(trajectories.track):
                sub = trajectories[trajectories.track == t]
    
                vxvy = np.diff(sub[['x','y']].values, axis=0)
                sub = sub[:-1]
                sub['vx'] = vxvy[:,0]
                sub['vy'] = vxvy[:,1]
        
                out.append(sub)
                printp('Deriving velocities for track: '+ str(t+1) + '/'+ str(n_tracks))
    
        # Calculo de desplazamientos con ventana movil
        else:
            out = []
            for t in set(trajectories.track):
                sub = trajectories[trajectories.track == t]
        
                try:
                    vxvy = sub[['x','y']][n:].values - sub[['x','y']][:-n].values
                    sub = sub[:-n]
                    sub['vx'] = vxvy[:,0] / n
                    sub['vy'] = vxvy[:,1] / n
            
                    out.append(sub)
                    printp('Deriving velocities for track: '+ str(t+1) + '/'+ str(n_tracks))
                except:
                    pass

    out = pd.concat(out)
    out = out.reset_index(drop=True)
    out = reset_track_indexes(out)

    return out




def butter_lowpass(fr, step, arr, fps=250):
    
    #N, Wn = signal.buttord(1./(1.*fps/step),1./fps, 1/step, fps*0.5)
    N, Wn = buttord(fr/step, fr, 1/step, fps*0.5 ,0.5/fps)
    b, a = butter(N, Wn,'low')
    y = filtfilt(b, a, arr)
    return y


    
def smoothVelocities(velocities, window_length=25, poly_order=3, kind='savgol', butter_size=0.8):
    """ This function takes a pandas dataframe with the velocities of one or
        more particles and smooths velocities applying an Savgol filter """
    # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way

    if kind=='savgol':
        col_names = ['frame', 'track', 'x', 'y', 'vx', 'vy']
        # Creating an empty dataframe to store results
        data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
        
        for item in set(velocities.track):       
            sub = velocities[velocities.track==item]
            
            if sub.shape[0] <= window_length+1:
                #Para obviar los casos en los que la trayectoria dura menos que la ventana de suavizado
                pass
            else:
                printp('Smoothing velocities for track: '+ str(item))
                # Savgol filter
                vx = pd.DataFrame(savgol_filter(sub.vx, window_length, poly_order), columns=['vx',])
                vy = pd.DataFrame(savgol_filter(sub.vy, window_length, poly_order), columns=['vy',])
            
                new_df = pd.concat((sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True), 
                                    sub.x.reset_index(drop=True), sub.y.reset_index(drop=True), vx, vy), 
                                    axis=1, names=col_names, sort=False)
                data = pd.concat((data, new_df), axis=0)
            
        # This is to get rid of the first 'np.zeros' row and to reset indexes
        data = data.reset_index(drop=True)
        data = data.drop(0)
        data = data.reset_index(drop=True)

    elif kind=='butter':
        col_names = ['frame', 'track', 'x', 'y', 'vx', 'vy']
        # Creating an empty dataframe to store results
        data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
        
        for item in set(velocities.track):       
            sub = velocities[velocities.track==item]
            
            if sub.shape[0] <= window_length+1:
                #Para obviar los casos en los que la trayectoria dura menos que la ventana de suavizado
                pass
            else:
                printp('Smoothing velocities for track: '+ str(item))
                # Savgol filter
                try:
                    vx = pd.DataFrame(butter_lowpass(butter_size, window_length, sub.vx), columns=['vx',])
                    vy = pd.DataFrame(butter_lowpass(butter_size, window_length, sub.vy), columns=['vy',])
                
                    new_df = pd.concat((sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True), 
                                        sub.x.reset_index(drop=True), sub.y.reset_index(drop=True), vx, vy), 
                                        axis=1, names=col_names, sort=False)
                    data = pd.concat((data, new_df), axis=0)
                except:
                    print('Track too short for Butterworth filter ', sub.shape[0])

        # This is to get rid of the first 'np.zeros' row and to reset indexes
        data = data.reset_index(drop=True)
        data = data.drop(0)
        data = data.reset_index(drop=True)

    return data


def smoothPositions(pos_data, window_length=25, poly_order=3, kind='savgol', butter_size=0.8):
    """ This function takes a pandas dataframe with the velocities of one or
        more particles and smooths positions applying an Savgol filter """
    # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    if kind=='savgol':
        col_names = ['frame', 'track', 'x', 'y', 'vx', 'vy']
        # Creating an empty dataframe to store results
        data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
        
        for item in set(pos_data.track):
            sub = pos_data[pos_data.track==item]
            
            if sub.shape[0] <= window_length+1:
                #Para obviar los casos en los que la trayectoria dura menos que la ventana de suavizado
                pass
            else:
                printp('Smoothing positions for track: '+ str(item))
                # Savgol filter
                x = pd.DataFrame(savgol_filter(sub.x, window_length, poly_order), columns=['x',])
                y = pd.DataFrame(savgol_filter(sub.y, window_length, poly_order), columns=['y',])
            
                new_df = pd.concat((x, y, sub.vx.reset_index(drop=True), sub.vy.reset_index(drop=True),
                                    sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True)),
                                    axis=1, names=col_names, sort=False)
                data = pd.concat((data, new_df), axis=0, sort=True)
            
        # This is to get rid of the first 'np.zeros' row and to reset indexes
        data = data.reset_index(drop=True)
        data = data.drop(0)
        data = data.reset_index(drop=True)

    elif kind=='butter':
        col_names = ['frame', 'track', 'x', 'y', 'vx', 'vy']
        # Creating an empty dataframe to store results
        data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
        
        for item in set(pos_data.track):
            sub = pos_data[pos_data.track==item]
            
            if sub.shape[0] <= window_length+1:
                #Para obviar los casos en los que la trayectoria dura menos que la ventana de suavizado
                pass
            else:
                printp('Smoothing positions for track: '+ str(item))
                # Savgol filter
                x = pd.DataFrame(butter_lowpass(butter_size, window_length, sub.x), columns=['x',])
                y = pd.DataFrame(butter_lowpass(butter_size, window_length, sub.y), columns=['y',])
            
                new_df = pd.concat((x, y, sub.vx.reset_index(drop=True), sub.vy.reset_index(drop=True),
                                    sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True)),
                                    axis=1, names=col_names, sort=False)
                data = pd.concat((data, new_df), axis=0)
            
        # This is to get rid of the first 'np.zeros' row and to reset indexes
        data = data.reset_index(drop=True)
        data = data.drop(0)
        data = data.reset_index(drop=True)

    return data



def deleteShortTrajectories(velocities, minimumFrames=150):
    """ This function takes a pandas dataframe with the velocities of one or
        more particles and deletes all trajectories 
        shorter than 'minimumFrames' """
    # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    
    col_names = ['frame', 'track', 'x', 'y', 'vx', 'vy']
    # Creating an empty dataframe to store results
    data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
    
    for item in set(velocities.track):
        sub = velocities[velocities.track==item]
        
        if sub.shape[0] <= minimumFrames:
            printp('Deleting velocities for track: '+str(item)+ ' --> Size: ' + str(sub.shape[0]))
            pass
        else:
            new_df = pd.concat((sub.vx.reset_index(drop=True), sub.vy.reset_index(drop=True), 
                                sub.x.reset_index(drop=True), sub.y.reset_index(drop=True),
                                sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True)),
                                axis=1, names=col_names, sort=False)
            data = pd.concat((data, new_df), axis=0)
        
    # This is to get rid of the first 'np.zeros' row and to reset indexes
    data = data.reset_index(drop=True)
    data = data.drop(0)
    data = data.reset_index(drop=True)
    
    return data


def alternative_delete_short_trajectories(data, minimumFrames=10):
    """ This function takes a pandas dataframe with the velocities of one or
        more particles and deletes all trajectories 
        shorter than 'minimumFrames' """
    # First, find what tracks have less datapoints that 'minimumFrames'
    tracks_to_delete = []
    for item in set(data.track):
        sub = data[data.track==item]
        n_elements = len(sub)
        if n_elements <= minimumFrames:
            printp('Deleting velocities for track: ' + str(item) + ' --> Size: ' + str(n_elements))
            tracks_to_delete.append(item)
    
    # Actual cleaning of short trajectories, first locate indexes   
    indexNames = data[data.track.isin(tracks_to_delete)].index
    #https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    #https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
    data = data.drop(indexNames)
    data = data.reset_index(drop=True)
    
    return data


def difference(array, n=1):
    """ What np.diff should be for orders higher than 1 """
    array = np.array(array)
    return array[n:] - array[:-n]


def calculate_jumps(data, interval=1, moving_window=True):
    """ returns a neww dataframe with columns 'dx', 'dy', representing
        the jump during a jump of lenght 'interval' (n_frames) """
        
    col_names = ['frame', 'track', 'x', 'y', 'dx', 'dy']
    n_tracks = len(set(data.track))
    # Creating an empty dataframe to store results
# =============================================================================
#     out = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)
# =============================================================================
    out = []

    for item in set(data.track):      
        sub = data[data.track==item]
        
        if sub.shape[0]<=interval+1:
            #Para obviar los casos en los que solo hay pocos datos
            pass
        else:
            printp('Calculating jumps for track: '+ str(item) + '/'+ str(n_tracks))
            if moving_window==False:
                dx = pd.DataFrame(sub.x.iloc[interval::interval].values - sub.x.iloc[0::interval].values[:-1], columns=['dx',])
                dy = pd.DataFrame(sub.y.iloc[interval::interval].values - sub.y.iloc[0::interval].values[:-1], columns=['dy',])
                sub = sub.iloc[0::interval].reset_index(drop=True)

            elif moving_window==True: # Este es el comportamiento por defecto
                dx = pd.DataFrame(difference(sub.x.values, n=interval), columns=['dx',])
                dy = pd.DataFrame(difference(sub.y.values, n=interval), columns=['dy',])


            new_df = pd.concat((sub.frame.reset_index(drop=True), sub.track.reset_index(drop=True), 
                                sub.x.reset_index(drop=True), sub.y.reset_index(drop=True), dx, dy), 
                                axis=1, names=col_names, sort=False)
# =============================================================================
#             out = pd.concat((out, new_df), axis=0)
# =============================================================================
            out.append(new_df)
    out = pd.concat(out, axis=0)

    # This is to get rid of the first 'np.zeros' row and to reset indexes, also, delete nans that appear at the end (in [-interval:])
    data = out.reset_index(drop=True)
    data = data.drop(0)
    data = data.dropna() #26-09-2019
    data = data.reset_index(drop=True)
        
    return data







def calculate_accelerations(trajectories, n=1, use_gradient=False):

    n_tracks = len(set(trajectories.track))

    # Con gradient solo se puede usar n=1
    if use_gradient==True:
        out = []
        for t in set(trajectories.track):
            sub = trajectories[trajectories.track == t]

            vxvy = np.gradient(sub[['vx','vy']].values, axis=0)
            sub['ax'] = vxvy[:,0]
            sub['ay'] = vxvy[:,1]
    
            out.append(sub)
            printp('Deriving a for track: '+ str(t+1) + '/'+ str(n_tracks))

    else:
        if n==1:
            out = []
            for t in set(trajectories.track):
                sub = trajectories[trajectories.track == t]
    
                vxvy = np.diff(sub[['vx','vy']].values, axis=0)
                sub = sub[:-1]
                sub['ax'] = vxvy[:,0]
                sub['ay'] = vxvy[:,1]
        
                out.append(sub)
                printp('Deriving a for track: '+ str(t+1) + '/'+ str(n_tracks))
    
        # Calculo de desplazamientos con ventana movil
        else:
            out = []
            for t in set(trajectories.track):
                sub = trajectories[trajectories.track == t]
        
                try:
                    vxvy = sub[['vx','vy']][n:].values - sub[['vx','vy']][:-n].values
                    sub = sub[:-n]
                    sub['ax'] = vxvy[:,0] / n
                    sub['ay'] = vxvy[:,1] / n
            
                    out.append(sub)
                    printp('Deriving a for track: '+ str(t+1) + '/'+ str(n_tracks))
                except:
                    pass

    out = pd.concat(out)
    out = out.reset_index(drop=True)
    out = reset_track_indexes(out)

    return out