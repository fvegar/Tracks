
import cv2 as cv2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#%matplotlib inline

#read original image
img = cv2.imread('snapshot6.png')
# make image smaller
# =============================================================================
# rgb = cv2.pyrDown(img)
# =============================================================================
# reduced image in b/w
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (5,5),0)
# apply grad morph so that I get thick borders. use a circle kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
grad = cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, kernel)

# binarize intelligently, by combining Otus's and simple binarization
_, binary_bw = cv2.threshold(grad, 0.0, 22.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# apply open morph over a straight line kernel, 
#so that I redraw in white regions inside thick borders, but only those containing straight lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
connected = cv2.morphologyEx(binary_bw, cv2.MORPH_CLOSE, kernel)

connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))

# add whitened thick borders to b/w image so that I erase only text/symbols with straigth lines
added=cv2.add(connected,bw)

circles = cv2.HoughCircles(added, cv2.HOUGH_GRADIENT,2,45,param1=60,param2=50,minRadius=24,maxRadius=33)

for i in circles[0]:
    pass
    # Draw the outer circle [(x,y), radius, rgb]
# =============================================================================
#     cv2.circle(added, (i[0],i[1]),i[2], (127, 0, 255), 2)
# =============================================================================
    # Draw the center of the circle
# =============================================================================
#     cv2.circle(added, (i[0],i[1]), 2, (127, 0, 255), 3) 
# =============================================================================

#display and compare unprocessed b/w image and final processed image
cv2.imshow('add', connected)
cv2.waitKey(0)
cv2.imshow('small', added)
cv2.waitKey(0)