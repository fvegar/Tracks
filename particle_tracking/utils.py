# -*- coding: utf-8 -*-
"""
Collection of utility functions, some of them are only used internally and others may
be broken.

@author: malopez
"""
import glob
import numpy as np
import pandas as pd
import pims
import cv2
import matplotlib.pyplot as plt
# =============================================================================
# from scipy.signal import find_peaks
# =============================================================================


class Peak:
    # 
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return float("inf") if self.died is None else seq[self.born] - seq[self.died]

def get_persistent_homology(seq):
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)


def find_ideal_thresh_value(videoPath):

    video = cv2.VideoCapture(videoPath)

    # I read the first 20 frames of the video to build an histogram
    frames = []
    n = 0
    while(video.isOpened()):
        # Leemos el frame actual y lo asignamos a la variable frame
        frameExists, frame = video.read()
        frames.append(frame)
        n+=1
        if n > 19:
            break

    # From this frame i transform it into a long 1D array
    frames = np.array(frames).ravel()
    hist, bins = np.histogram(frames, bins=100)

# =============================================================================
#     peaks = find_peaks(hist, width=1.75)[0]
#     downs = find_peaks(-hist, width=1.75)[0]
# =============================================================================
    peaks = get_persistent_homology(hist)

    # The ideal thresh value will be right after (+8 bins) the first maxima
    #ideal_thresh = bins[peaks[0]+4]
    ideal_thresh = bins[peaks[0].born+11]

    # The following code is used to check if this method works
# =============================================================================
#     for file in files:
#       thresh = find_ideal_thresh_value(file)
#       print(thresh)
#       cap = cv2.VideoCapture(file)
#       rval, frame = cap.read()
#       _, binarized = cv2.threshold(frame, thresh, 255.0, cv2.THRESH_BINARY)
#       showImage(binarized, name='Binary')
# =============================================================================
    return ideal_thresh


def angle_from_2D_points(array_x, array_y):
    """ Given two 1D arrays, corresponding to 'x' and 'y' coordinates of 2D points,
        this function returns the angle of those points directions

    Parameters
    ----------
    array_x : array
        1D array of 'x' positions, ints or floats.        
    array_y : array
        1D array of 'y' positions, ints or floats.   
    
    Returns
    -------
    array
        A 1D numpy array with points angles expressed in degrees.
    """
    rad_angles = -np.arctan2(array_y, array_x) % (2 * np.pi)
    deg_angles = np.rad2deg(rad_angles)
    return deg_angles


def createCircularROI(pos_data, center, radius):
    """ Select points within a given radius of a central point. Circular Region of Interest defined
        by its center and radius.

    Parameters
    ----------
    pos_data : pandas Ddataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'x', 'y']   
    center : tuple or list
        Pair of coordinates representing the central point of the Region of Interest.
        It must have the same units as 'x', 'y' in pos_data
    radius : float
        Value of the ROI radius
    
    Returns
    -------
    pandas Dataframe
        Dataframe with the same columns as the input but selecting only the rows
        for which (x,y) is located inside the ROI
    """
    # Find squared distance of each detected position to the center
    r2 = (pos_data['x'] - center[0])**2 + (pos_data['y'] - center[1])**2
    # Select only those rows that fall inside the ROI
    inside = pos_data.loc[r2 <= radius**2]
    # Reset indexes and return the new dataFrame
    inside = inside.reset_index(drop=True)
    return inside


def createRectangularROI(pos_data, origin, width, height):
    """ Select points that are in a rectangular area. Rectangular Region of Interest defined
        by its lower left corner, height and width.

    Parameters
    ----------
    pos_data : pandas Ddataframe
        It must be have the usual structure with at least the folowing columns:
        ['frame', 'track', 'x', 'y']   
    origin : tuple or list
        Pair of coordinates representing the origin of the ROI (lower left corner).
        It must have the same units as 'x', 'y' in pos_data
    width : float
        Length of the box in the x coordinate
    height : float
        Length of the box in the y coordinate
    
    Returns
    -------
    pandas Dataframe
        Dataframe with the same columns as the input but selecting only the rows
        for which (x,y) is located inside the ROI
    """
    x0 = origin[0]
    y0 = origin[1]
    # Select only those rows that fall inside the ROI
    inside = pos_data[pos_data['x'] >= x0]
    inside = inside[inside['y'] >= y0]
    inside = inside[inside['x'] <= (x0+width)]
    inside = inside[inside['y'] <= (y0+height)]
    # Reset indexes and return the new dataFrame
    inside = inside.reset_index(drop=True)
    return inside


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


def plotPosvsTime(data, timePerFrame=1):
        y = data
        dt = np.linspace(0,y.size, y.size)*timePerFrame
        
        fig, ax = plt.subplots(figsize=(10,8), dpi=250)
        ax.set_xlim([0,dt.max()])
    
        ax.set_xlabel('t (s)')
        ax.set_ylabel('x (m)')
        plt.plot(dt, y)
     
        
        
def createCircularMask(h, w, center=None, radius=None):
    """ Creates an OpenCV circular mask

    Parameters
    ----------
    h : int
        Height of the image for which the mask is going to be used  
    w : int
        Width of the image for which the mask is going to be used 
    center : tuple or list, optional
        Pair of coordinates for the mask's central point. If not specified uses: [w/2, h/2]
    radius : float, optional
        Value of the mask radius. If not specified uses max possible value
    
    Returns
    -------
    mask : array
        Mask for using with the image (array of True/False values)
    """

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def maskImage(img, mask):
    """ Masks an input image

    Parameters
    ----------
    img : array
        Input image
    mask : array
        True/False array with same shape as input image

    Returns
    -------
    masked_img : array
        output masked image

    """
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
        resets indexes so that we can loop over the tracks with 'range' 

    Parameters
    ----------
    data : pandas Dataframe
        Standard dataframe obtained from our tracking proccess. 
        At least with columns ['frame', 'track', 'x', 'y']

    Returns
    -------
    data : pandas Dataframe
        Return dataframe with same shape and info but with changed 'track' indexes
    """
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


def present_in_folder(experiment_id, folder):
    """ Returns True if there are files containing 'experiment_id' in a given folder """
    files_in_folder = glob.glob(folder + '*')
    files_containing_exp_id = glob.glob(folder+experiment_id+'*')

    present = False
    for file in files_containing_exp_id:
        if file in files_in_folder:
            present = True

    return present


