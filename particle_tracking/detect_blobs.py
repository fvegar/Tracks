# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:24:32 2018

@author: malopez
"""
import pandas as pd
import pims
import cv2
import numpy as np
from utils import morphOperation, showImage, printProgressBar, detectContourRadius, alternative_findMeanRadius, createCircularMask, maskImage
# =============================================================================
# from watershedContours import watershedContours
# =============================================================================


def showImageWithCircles(img, circles):
    
    for index, c in circles.iterrows():   
        cv2.circle(img, (int(c.x), int(c.y)), 5, (255, 255, 255), -1)
        cv2.putText(img, "centroid", (int(c.x) - 25, int(c.y) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    showImage(img)
# =============================================================================
#     cv2.imshow('Video', img)
#     if cv2.waitKey(1) or 0xFF == ord('q'):
#         pass
# =============================================================================
    
def eraseText(img, k1=3, k2=5, k3=3, k4=5, display_intermediate_steps=False):
    """ Tries to delete text combining thresholds and other morphological operations
        k1, k2, k3 and k4 are parameters that define the kernel size for
        the diferent morph ops """
    
    # ---------------- Fran's Code ---------------------
    # Grayscale and Gaussian Blur
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.GaussianBlur(bw, (5,5), cv2.BORDER_DEFAULT)
    if display_intermediate_steps==True:
        showImage(bw)
    
    # apply grad morph so that I get thick borders. use a circle kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)
    if display_intermediate_steps==True:
        showImage(grad)
    
    # binarize intelligently, by combining Otus's and simple binarization
    _, binary_bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if display_intermediate_steps==True:
        showImage(binary_bw)
        
    # apply open morph over a straight line kernel, 
    #so that I redraw in white regions inside thick borders, but only those containing straight lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
    connected = cv2.morphologyEx(binary_bw, cv2.MORPH_CLOSE, kernel)
    if display_intermediate_steps==True:
        showImage(connected)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3, k3))
    erosion = cv2.erode(binary_bw, kernel ,iterations = 2)
    if display_intermediate_steps==True:
        showImage(erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k4, k4))
    dilation =  cv2.dilate(erosion,  kernel)
    
    if display_intermediate_steps==True:
        showImage(dilation)
    difference = cv2.absdiff(connected,dilation)
    if display_intermediate_steps==True:
        showImage(difference)
    # add whitened thick borders to b/w image so that I erase only text/symbols with straigth lines
    added = cv2.add(difference, bw)
    return added


def findCentroids(contours):
    centroids = []
    for c in contours:
       # calculate moments for each contour
       M = cv2.moments(c)     
       # calculate x,y coordinate of center
       if M["m00"] != 0:
           cX = M["m10"] / M["m00"]
           cY = M["m01"] / M["m00"]
           centroids.append([cX, cY])
       else:
           cX, cY = 0, 0    
    # Now we get those circles' positions and store them in an appropiate format
    centroids = np.array(centroids)
    return centroids


def detectCircles_watershed(img, frame_number=0, display_intermediate_steps=False, meanRadius=30, thresh=20, opening_kernel=5):
    """ Produces a biased position when particles are not uniform (letters)
        but it is way more robust to changes in ilumination """

    contours = watershedContours(img)
    # Detect the centroid of each contour and store it on a list
    centroids = findCentroids(contours)
    N = centroids.shape[0]

    # We first prepare the structure to store detected circles in a format that
    # trackpy can easily work with (for the linking)   
    A = pd.DataFrame(np.zeros((N, 2), dtype=np.float64), columns=('x', 'y'))
    B = pd.DataFrame(np.full((N, 1), frame_number, dtype=np.int64), columns=('frame',))
    C = pd.DataFrame(np.full((N, 1), meanRadius, dtype=np.float64), columns=('size',))
    circles_tp = pd.concat((A, C, B), axis=1)
    # Fill that structure:
    circles_tp.x, circles_tp.y = centroids[:,0], centroids[:,1]
    
    if display_intermediate_steps==True:
        showImageWithCircles(img, circles_tp)
    
    return circles_tp


def alternative_detectCirclesImage(img, frame_number=0, display_intermediate_steps=False, meanRadius=30, thresh=20, opening_kernel=5):
    """ Produces a biased position when particles are not uniform (letters)
        but it is way more robust to changes in ilumination """
    # FOR CAMERA VERY HIGH, FIRST CREATE CIRCULAR MASK
    mask = createCircularMask(800, 1280, center=[649,392], radius=408)
    img = maskImage(img, mask)



    if display_intermediate_steps==True:
        showImage(img, name='Original')
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(bw, thresh, 255.0, cv2.THRESH_BINARY)
    if display_intermediate_steps==True:
        showImage(binarized, name='Binary')
        print('Thresh: '+str(thresh))
    opened = morphOperation(binarized, operation='opening', times=1, kernel_size=opening_kernel)
    if display_intermediate_steps==True:
        showImage(opened, name='Opened')
        print('Opening Kernel size: '+str(opening_kernel))
    closed = morphOperation(opened, operation='closing', times=1, kernel_size=6)
    if display_intermediate_steps==True:
        showImage(closed, name='Closed')
    eroded = morphOperation(closed, operation='erosion', times=1, kernel_size=opening_kernel+38)
    if display_intermediate_steps==True:
        showImage(eroded, name='Erode')
# =============================================================================
#     eroded = morphOperation(opened, operation='gradient', times=1, kernel_size=12)
#     if display_intermediate_steps==True:
#         showImage(eroded, name='Erode')
#         showImage(closed-eroded, name='Erode')
# =============================================================================


    # Find contours in the binary image:
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Detect the centroid of each contour and store it on a list
    centroids = findCentroids(contours)
    N = centroids.shape[0]

    # We first prepare the structure to store detected circles in a format that
    # trackpy can easily work with (for the linking)   
    A = pd.DataFrame(np.zeros((N, 2), dtype=np.float64), columns=('x', 'y'))
    B = pd.DataFrame(np.full((N, 1), frame_number, dtype=np.int64), columns=('frame',))
    C = pd.DataFrame(np.full((N, 1), meanRadius, dtype=np.float64), columns=('size',))
    circles_tp = pd.concat((A, C, B), axis=1)
    # Fill that structure:
    circles_tp.x, circles_tp.y = centroids[:,0], centroids[:,1]
    
    if display_intermediate_steps==True:
        showImageWithCircles(img, circles_tp)
    
    return circles_tp


def detectCirclesImage(img, frame_number=0, display_intermediate_steps=False, meanRadius=30):

    if display_intermediate_steps==True:
        showImage(img, name='Original')
    
    added = eraseText(img, k1=3, k2=5, k3=3, k4=5, display_intermediate_steps=display_intermediate_steps)
    
    if display_intermediate_steps==True:
        showImage(added)
        
    # Now I binarize the image to get only b/w, the erosion so blobs are not
    # connected and can later be detected as separate contours
    _, binarized = cv2.threshold(added, 20.0, 255.0, cv2.THRESH_BINARY)
    if display_intermediate_steps==True:
        showImage(binarized)
    # The kernel must be of a size similar (better higher) to that of the feature we want to detect
# =============================================================================
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
#     eroded = cv2.erode(binarized[1], kernel , iterations=1)
# =============================================================================
    eroded = morphOperation(binarized, operation='erosion', times=1, kernel_size=36)
    if display_intermediate_steps==True:
        showImage(eroded)  
        
    # Find contours in the binary image:
    contours, hierarchy = cv2.findContours(eroded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Detect the centroid of each contour and store it on a list
    centroids = []
    for c in contours:
       # calculate moments for each contour
       M = cv2.moments(c)     
       # calculate x,y coordinate of center
       if M["m00"] != 0:
           cX = M["m10"] / M["m00"]
           cY = M["m01"] / M["m00"]
           centroids.append([cX, cY])
       else:
           cX, cY = 0, 0
    
    # Now we get those circles' positions and store them in an appropiate format
    centroids = np.array(centroids)
    N = centroids.shape[0]

    # We first prepare the structure to store detected circles in a format that
    # trackpy can easily work with (for the linking)   
    A = pd.DataFrame(np.zeros((N, 2), dtype=np.float64), columns=('x', 'y'))
    B = pd.DataFrame(np.full((N, 1), frame_number, dtype=np.int64), columns=('frame',))
    C = pd.DataFrame(np.full((N, 1), meanRadius, dtype=np.float64), columns=('size',))
    circles_tp = pd.concat((A, C, B), axis=1)
    # Fill that structure:
    circles_tp.x, circles_tp.y = centroids[:,0], centroids[:,1]
    
    if display_intermediate_steps==True:
        showImageWithCircles(img, circles_tp)
    
    return circles_tp


def detectCirclesVideo(videoPath, initialFrame=0, lastFrame='max', thresh=20, display_intermediate_steps=False, opening_kernel=5):
    if lastFrame=='max':
        # Find number of frames in the video
        v = pims.Cine(videoPath)
        lastFrame = v.len()-1
    #TODO: falta rellenar la columna size (con el radio medio detectado en los primeros 10 frames por ejemplo)
    # We first create an empty dataframe to store the circles in the correct format
    A = pd.DataFrame(np.zeros((1, 2), dtype=np.float64), index=('-1',), columns=('x', 'y'))
    B = pd.DataFrame(np.full((1, 1), 0, dtype=np.int64), index=('-1',), columns=('frame',))
    C = pd.DataFrame(np.full((1, 1), 0, dtype=np.float64), index=('-1',), columns=('size',))
    circles_tp = pd.concat((A, C, B), axis=1)
# =============================================================================
#     try:
#         meanRadius = findMeanRadius(videoPath, n_frames=10)
#     except:
#         meanRadius = 30    
# =============================================================================
    meanRadius = 29 
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
            # Detect circles for current frame and append them to the general dataframe
            new_circles = alternative_detectCirclesImage(frame, frame_number=n, meanRadius=meanRadius, 
                                             display_intermediate_steps=display_intermediate_steps, thresh=thresh)
            
# =============================================================================
#             new_circles = detectCircles_watershed(frame, frame_number=n, meanRadius=meanRadius, 
#                                                   display_intermediate_steps=display_intermediate_steps, thresh=thresh)
# =============================================================================
            
            circles_tp = pd.concat((circles_tp, new_circles), axis=0)
            n+=1
    
        printProgressBar(n, lastFrame+2-initialFrame, prefix='Detecting particles:', suffix='frames searched')
    # Cerramos el stream de video
    video.release()
    # We delete the first row of circles_tp, since it was only used for 
    # initialization and is no longer needed.
    circles_tp = circles_tp.drop('-1')
    #TODO: Reniciar indexes
    circles_tp = circles_tp.reset_index(drop=True)
    
    return circles_tp
