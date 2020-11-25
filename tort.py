import cv2
import numpy as np
from utils import *
from fractals import *

img = cv2.imread('branched.png',0)

canny = cv2.Canny(img,0,255)
contours,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if len(cnt)>=25:
        output = cv2.drawContours(np.zeros(img.shape,dtype = np.uint8),[cnt],-1,(255,255,255),1)

        x,y,h,w = cv2.boundingRect(cnt)

        roi = output[y:y+w, x:x+h]
        print(len(roi[roi==255]))

        r,c = roi.shape
        back = np.zeros((r+2,c+2),dtype=np.uint8)
        back[1:r+1,1:c+1] = roi

        iroi = back.copy()
        froi = back.copy()
        inflections = getinflections(roi.copy())
        
        fdimension = fractal_dimension(froi.copy())
        fdimension = fdimension*(-1)

        print(fdimension)

        # out = cv2.cvtColor(roi.copy(),cv2.COLOR_GRAY2BGR)
        
        # for i,j in inflections:
        #     out[j,i] = [0,0,255]

        # cv2.imshow('output',out)
        # cv2.waitKey(0)

cv2.imshow('canny',canny)


cv2.waitKey(0)
cv2.destroyAllWindows()