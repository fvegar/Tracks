import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, buttord, butter, filtfilt
from utils import printp, reset_track_indexes

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
                data = pd.concat((data, new_df), axis=0)
            
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
    
    
    
 def butter_lowpass(fr, step, arr, fps=250):
    
    #N, Wn = signal.buttord(1./(1.*fps/step),1./fps, 1/step, fps*0.5)
    N, Wn = buttord(fr/step, fr, 1/step, fps*0.5 ,0.5/fps)
    b, a = butter(N, Wn,'low')
    y = filtfilt(b, a, arr)
    return y
