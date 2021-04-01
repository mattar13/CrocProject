import numpy as np
import cv2
import os
import math
import sys
from scipy import ndimage
# Set recursion limit
sys.setrecursionlimit(10 ** 9)
import sys
import selectinwindow


def theta(x, y):
    #print(x[1], x[0])
    #print(y[1], y[0])
    
    atans = math.atan2(int(x[1])-int(x[0]),int(y[1])-int(y[0]))
    deg = math.degrees(atans)
    return 90 - deg

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


def manual_format(img, scale = 2, stop_sign = False):
    global btn_down
    btn_down = False
    orows, ocols, __ = img.shape
    print(int(ocols/scale))
    r_img = cv2.resize(img, (int(ocols/scale), int(orows/scale)))
    repeat = True
    rows, cols, ch = img.shape
    print(img.shape)
    wName = "Select region"
    rectI = selectinwindow.dragRect
    
    while repeat:

        if stop_sign == False:
            pts, rot_image = get_points(r_img)
            xs = pts[0,:,0]
            ys = pts[0,:,1]
            
            thet = theta(xs, ys)
            
            M = cv2.getRotationMatrix2D((cols/2, rows/2), thet, 1)
            dst = cv2.warpAffine(rot_image, M, (cols, rows))
            cv2.destroyWindow("Image")
            
            selectinwindow.init(rectI, dst, wName, rows, cols)   
            cv2.namedWindow(rectI.wname)
            cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
            
            while True:
                # display the image
                cv2.imshow(wName, rectI.image)
                key = cv2.waitKey(0) & 0xFF
                # if returnflag is True, break from the loop
                if rectI.returnflag == True:
                    cv2.destroyAllWindows()
                    break
            x = rectI.outRect.x * scale
            y = rectI.outRect.y * scale
            w = rectI.outRect.w * scale
            h = rectI.outRect.h * scale
            print("all done")
            escape = input("Are you satisfied with these image boundaries?").capitalize()
            if escape == "Y":
                repeat = False
                cv2.destroyAllWindows()
                return x,y,w,h,thet
            elif escape == "ESC":
                exit()
        else:
            
            pts, rot_image = get_points(r_img)
            xs = pts[0,:,0]
            ys = pts[0,:,1]
            
            thet = theta(xs, ys)

            M = cv2.getRotationMatrix2D((cols/2, rows/2), thet, 1)
            dst = cv2.warpAffine(rot_image, M, (cols, rows))
            cv2.destroyWindow("Image")
            
            selectinwindow.init(rectI, dst, wName, rows, cols)   
            cv2.namedWindow(rectI.wname)
            cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
            
            while True:
                # display the image
                cv2.imshow(wName, rectI.image)
                key = cv2.waitKey(0) & 0xFF
                # if returnflag is True, break from the loop
                if rectI.returnflag == True:
                    cv2.destroyAllWindows()
                    break
            x = rectI.outRect.x * scale
            y = rectI.outRect.y * scale
            w = rectI.outRect.w * scale
            h = rectI.outRect.h * scale
            print("all done")
            escape = input("Are you satisfied with these image boundaries?").capitalize()
            if escape == "Y":
                repeat = False
                cv2.destroyAllWindows()
                return x,y,w,h
            
        
    #return x, y, ix, iy, thet



if __name__ == "__main__":
#if False:
    os.chdir("C://pyscripts//drag_rect//")
    btn_down = False
    img = cv2.imread('Figure_1.png')
    res = manual_format(img)

    