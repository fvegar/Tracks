# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:08:26 2018

@author: malopez
"""
import numpy as np
import pandas as pd
import pims
import cv2
import matplotlib.pyplot as plt

def detectContourRadius(contours):
    """ Returns an array of all the radiuses detected in a given frame """
    radius = []
    for c in contours:
        area = cv2.contourArea(c)
        equivalent_radius = np.sqrt(area/np.pi)
        radius.append(equivalent_radius)
    # Now we get those circles' positions and store them in an appropiate format
    radius = np.array(radius)
    return radius


def alternative_findMeanRadius(videoPath, initialFrame=0, lastFrame='max', thresh=20, opening_kernel=5):
    """ Finds mean radius from contour analysis """
    all_radiuses = np.array([])
    
    if lastFrame=='max':
        # Find number of frames in the video
        v = pims.Cine(videoPath)
        lastFrame = v.len()-1
    
    video = cv2.VideoCapture(videoPath)    
    n = 1 # Simple acumulador, para llevar la cuenta de por cual frame voy
    while(video.isOpened()):
        # Leemos el frame actual y lo asignamos a la variable frame
        frameExists, frame = video.read()
        
        if n<initialFrame+1:
            n+=1
            pass
        elif n>lastFrame+1:
            break
        else:
            # Detect circles for current frame
            bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binarized = cv2.threshold(bw, thresh, 255.0, cv2.THRESH_BINARY)
            opened = morphOperation(binarized, operation='opening', times=1, kernel_size=opening_kernel)
            closed = morphOperation(opened, operation='closing', times=1, kernel_size=10)
            
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            radiuses = detectContourRadius(contours)
            all_radiuses = np.concatenate((all_radiuses, radiuses))
            n+=1
    
        printProgressBar(n, lastFrame+2-initialFrame, prefix='Detecting radiuses:', suffix='frames searched')
    
    count = all_radiuses.shape[0]
    mean_radius = np.mean(all_radiuses)
    std_radius = np.std(all_radiuses)
    
    return count, mean_radius, std_radius


def findMeanRadius(videoPath, n_frames=5):
    """ Hough circles function is used to estimate mean radius of particles """
    video = cv2.VideoCapture(videoPath)
        
    allCircles = []   
    n=1
    while(video.isOpened()):
        # Leemos el frame actual y lo asignamos a la variable frame
        frameExists, frame = video.read()

        open_img = morphOperation(frame, operation='opening', times=1, kernel_type='small')                    
        frame_gray = cv2.cvtColor(open_img, cv2.COLOR_BGR2GRAY)
        # Detectamos los circulos dentro de una clausula try
        try:
            circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT,2,40,param1=30,
                                       param2=30,minRadius=20,maxRadius=35)
            allCircles.append(circles[0])
        except:
            pass        
        n+=1
        if n>n_frames:
            break
        
    video.release()
    
    result = pd.DataFrame(allCircles[0])
    for element in allCircles[1:]:
        result = pd.concat((result, pd.DataFrame(element)), axis=0)
    result = result.reset_index(drop=True)
    
    print('Mean Radius calculation:',result.shape[0]/n_frames, 'particles per frame.', n_frames, 'frames')
    return result[2].mean()      
        
        
def createCircularROI(df, center, radius):
    """ Select circles that are within a given radius of a central point
        (represented by a tuple) """
    # Find squared distance of each detected position to the center
    r2 = (df.x - center[0])**2 + (df.y - center[1])**2
    # Select only those rows that fall inside the ROI
    inside = df.loc[r2 <= radius**2]
    # Reset indexes and return the new dataFrame
    inside = inside.reset_index(drop=True)
    return inside


def createRectangularROI(df, origin, width, height):
    """ Select circles that are in a rectangular area (origin point
        (represented by a tuple) """
    x0 = origin[0]
    y0 = origin[1]
    # Select only those rows that fall inside the ROI
    inside = df[df.x >= x0]
    inside = inside[inside.y >= y0]
    inside = inside[inside.x <= (x0+width)]
    inside = inside[inside.y <= (y0+height)]
    # Reset indexes and return the new dataFrame
    inside = inside.reset_index(drop=True)
    return inside
   
    
def plotPosvsTime(data, timePerFrame=1):
        y = data
        dt = np.linspace(0,y.size, y.size)*timePerFrame
        
        fig, ax = plt.subplots(figsize=(10,8), dpi=250)
        ax.set_xlim([0,dt.max()])
    
        ax.set_xlabel('t (s)')
        ax.set_ylabel('x (m)')
        plt.plot(dt, y)
     
        
        
def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def maskImage(img, mask):
    masked_img = img.copy()
    masked_img[~mask] = 0
    
    return masked_img


def morphOperation(img, operation='opening', times=1, kernel_size=3):
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html 
    
    # I will default to a circular 7x7 kernel
# =============================================================================
#     kernel = np.array([[0, 0, 1, 1, 1, 0, 0],
#                        [0, 1, 1, 1, 1, 1, 0],
#                        [1, 1, 1, 1, 1, 1, 1],
#                        [1, 1, 1, 1, 1, 1, 1],
#                        [1, 1, 1, 1, 1, 1, 1],
#                        [0, 1, 1, 1, 1, 1, 0],
#                        [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
# =============================================================================
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # We define a dictionary with the different morphological operations
    morphs = {'erosion' : cv2.erode(img, kernel, iterations = times), 
              'dilation' : cv2.dilate(img, kernel,iterations = times), 
              'opening' : cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = times), 
              'closing' : cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations = times),
              'gradient': cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel),
              'top_hat': cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel),
              'black_hat': cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel),
              'laplacian': cv2.Laplacian(img, cv2.CV_64F)}
    
    final_img = morphs[operation]
    return final_img


def threshImage(img, minBrightness, option='binary'):
    if option=='tozero':
        ret, thresh = cv2.threshold(img, minBrightness, 255, cv2.THRESH_TOZERO)
    else:
        ret, thresh = cv2.threshold(img, minBrightness, 255, cv2.THRESH_BINARY)
        
    return thresh


def showImage(img, fps=0, name='Image'):
    if fps==0:
        t=200000
    else:
        t=int(1/fps)
    
    cv2.imshow(name, img)
    if cv2.waitKey(t) or 0xFF == ord('q'):
        cv2.destroyAllWindows()


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        

def to_IS_units(data, pixel_ratio):
    pass


def toNaturalUnits(circles, particle_diameter, vels_present=False):
    """ 
    Converts data in a typical dataframe [frame, track, x, y] to natural units.
    This is, it changes the units of x and y to be expressed in terms of the
    particle's diameter.
    
    Parameters
    ----------
    data : pandas DataFrame
        Standard dataframe obtained from our tracking proccess        
    particle_diameter : float
        Diameter of the particles we are currently studying.
        particle_diameter units should match those of the DataFrame
    vels_present : bool
        It tells us if columns vx and vy are pre
    
    Returns
    -------
    array
        A numpy array of equal shape as the source.
    """
    pass


def reset_track_indexes(data):
    """ This function takes a dataframe in which some trajectory indexes
        are missing (maybe due to having deleted short trajectories) and
        resets indexes so that we can loop over the tracks with 'range' """
    # 'real_number_of_tracks' should be <=  than 'current_last_particle_index'
    real_number_of_tracks = len(set(data['track']))
    # current_last_particle_index = data['track'].max()

    original_indexes = np.array(list(set(data['track'])))
    original_indexes = np.sort(original_indexes) #BE CAREFUUUUL
    fixed_indexes = np.arange(0, real_number_of_tracks, step=1)
    
    # With these two lists we create a dictionary and map old values to new ones
    replacement_dict = dict(zip(original_indexes, fixed_indexes))
    tracks_column = data['track']
    data['track'] = tracks_column.map(replacement_dict)
    
    print('Reseting indexes so that the tracks list is continous')
    
    return data
# =============================================================================
#     tracks_column = data['track'].astype('int32')
#     tracks_column.replace(to_replace=original_indexes, value=fixed_indexes, inplace=True)
#     data['track'] = tracks_column
# =============================================================================
    

def select_tracks_by_lenght(data, min_lenght=0, max_lenght=25000):
    """ Given a dataframe and a range of lenghts, returns an array with the
        indexes of tracks with a lenght within those limits """
    track_lenghts_list = []
    for part in set(data.track):
        sub_data = data[data.track == part]
        lenght = len(sub_data)
        track_lenghts_list.append([part,lenght])
    
    l = pd.DataFrame(track_lenghts_list, columns=['trajectory','lenght'])
    l = l.sort_values('lenght', ascending=False)
    l = l.reset_index(drop=True)
    
    selection = l[l.lenght <= max_lenght]
    selection = selection[selection.lenght >= min_lenght]
    
    return selection.trajectory.values


def play_video_with_labels(videoPath, trajectories, list_of_particles_to_track='all', mean_radius=39, mask_ROI=False):
    video = cv2.VideoCapture(videoPath)

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
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # Incrementing the count of frames
        f+=1
        
    # Cerramos el stream de video y las ventanas abiertas
    video.release()
    cv2.destroyAllWindows()
   
    
def reorder_rename_dataFrame(data):
    """ This function is only to rename column 'particle' with 'track'
        then reorder the columns and finally delete the column 'size'
        if present 
        It should always be called after running trackpy ( because that is 
        the one that generates column name 'particle' """
    # Delete
    if 'size' in data.columns:
        data = data.drop('size', 1)
    # Rename
    if 'particle' in data.columns:
        data = data.rename(columns={'particle': 'track'})
    else:
        print('WARNING: No column named -particle- in this dataframe')
    # Reorder
    if 'vx' in data.columns:
        data = data[['frame','track','x','y','vx','vy']]
    else:
        data = data[['frame','track','x','y']]
    
    data = data.reset_index(drop=True)
    print('Fran is now a happy man')    
    return data
        


def printp(string):
    """ Modification of print function to do everything on one line """
    import sys
    sys.stdout.write('\r'+ str(string))
    sys.stdout.flush()
    
    
def distanceToCenter(x, y, x_center, y_center):
    """ Simple function, given a pair of coordinates x,y. It returns its
        distances to a central point """
    return np.sqrt((x-x_center)**2 + (y-y_center)**2)
