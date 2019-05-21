# USAGE
# python watershed.py --image images/coins_01.png

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2
from utils import showImage, morphOperation


def watershedContours(image):
# =============================================================================
#     # load the image and perform pyramid mean shift filtering
#     # to aid the thresholding step
#     shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
# =============================================================================

    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(bw, 20, 255.0, cv2.THRESH_BINARY)
    opened = morphOperation(binarized, operation='opening', times=1, kernel_size=5)
# =============================================================================
#     showImage(opened, name="Output")
# =============================================================================
    closed = morphOperation(opened, operation='closing', times=1, kernel_size=5)
    eroded =  eroded = morphOperation(closed, operation='erosion', times=1, kernel_size=35)
# =============================================================================
#     showImage(closed, name="Output")
# =============================================================================

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(eroded)
    localMax = peak_local_max(D, indices=False, min_distance=30, exclude_border=False)
    

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=closed)

    # loop over the unique labels returned by the Watershed
    # algorithm
    contours = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
    
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(bw.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # detect contours in the mask and grab the largest one
        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        contours.append(c)

    contours = np.array(contours)

    return contours
        