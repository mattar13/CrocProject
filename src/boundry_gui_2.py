import numpy as np
import cv2
import os
import math
from scipy import ndimage

def theta(x, y):
    deg = math.degrees(math.atan2(x[1]-x[0],y[1]-y[0]))
    return 90 - deg


os.chdir("C://pyscripts//drag_rect//")
btn_down = False

def get_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])
    
    return points, data['im']

def mouse_handler(event, x, y, flags, data):
    global btn_down

    if event == cv2.EVENT_LBUTTONUP and btn_down:
        #if you release the button, finish the line
        btn_down = False
        data['lines'][0].append((x, y)) #append the seconf point
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255),5)
        cv2.line(data['im'], data['lines'][0][0], data['lines'][0][1], (0,0,255), 2)
        cv2.imshow("Image", data['im'])

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #thi is just for a ine visualization
        image = data['im'].copy()
        cv2.line(image, data['lines'][0][0], (x, y), (0,0,0), 1)
        cv2.imshow("Image", image)

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['lines']) < 1:
        btn_down = True
        data['lines'].insert(0,[(x, y)]) #prepend the point
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
        cv2.imshow("Image", data['im'])


img = cv2.imread('Figure_1.png')
rows,cols, ch = img.shape

pts, final_image = get_points(img)
cv2.imshow('Image', final_image)
xs = pts[0,:,0]
ys = pts[0,:,1]
dist_x = abs(xs[0] - xs[1])
dist_y = abs(ys[0] - ys[1])
angle = theta(xs, ys)

M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow("Warped Image", dst)

cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
print(pts)
print(angle)