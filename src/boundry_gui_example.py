# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 21:06:16 2018

@author: Matthew Tarchick
"""

import os
os.chdir("C://pyscripts//drag_rect//")
import cv2
import sys
import numpy as np
# Set recursion limit
sys.setrecursionlimit(10 ** 9)

import selectinwindow

# Define the drag object
rectI = selectinwindow.dragRect

# Initialize the  drag object
wName = "select region"
image = cv2.imread("Figure_1.png")
imageWidth = image.shape[0]
imageHeight = image.shape[1]
selectinwindow.init(rectI, image, wName, imageWidth, imageHeight)

cv2.namedWindow(rectI.wname)
cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)

# keep looping until rectangle finalized
while True:
    # display the image
    cv2.imshow(wName, rectI.image)
    key = cv2.waitKey(1) & 0xFF

    # if returnflag is True, break from the loop
    if rectI.returnflag == True:
        break

print("Dragged rectangle coordinates")
print(str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
      str(rectI.outRect.w) + ',' + str(rectI.outRect.h))

# close all open windows
cv2.destroyAllWindows()