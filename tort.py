import cv2
import numpy as np
from numpy.core.fromnumeric import mean
from utils import *
from fractals import *
from inflections import *


def arclength(order,img):
    h = distance(order[0],order[-1])

    arclength = 0
    n = len(order)
    for i in range(n-1):
        arclength+= distance(order[i],order[i+1])
    
    t = arclength/h

    return t

def tortuosity(img):
    k = 0

    contours,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    torts = []
    arc_torts = []
    for cnt in contours:
        output = cv2.drawContours(np.zeros(img.shape,dtype = np.uint8),[cnt],-1,(255,255,255),1)
        count = np.count_nonzero(output)
        if count>=15:            

            x,y,h,w = cv2.boundingRect(cnt)

            roi = output[y:y+w, x:x+h]

            r,c = roi.shape
            back = np.zeros((r+2,c+2),dtype=np.uint8)
            back[1:r+1,1:c+1] = roi

            iroi = back.copy()
            # cv2.imshow('iroi',iroi)
            # cv2.waitKey(0)
            order = order_points(iroi)

            

            inflections = getinflections(iroi,order)
            angles = get_angles(inflections)
            torts.append(mean(angles))

            arcbased = arclength(order,iroi)
            arc_torts.append(arcbased)

            

            if(count>=75 and k<5):
                cv2.imwrite('roi/'+'roi'+str(k)+'.png',back)
                out = cv2.cvtColor(back.copy(),cv2.COLOR_GRAY2BGR)
                for i,j in inflections:
                    out[i,j] = [0,0,255]
                cv2.imwrite('inf/inf'+str(k)+'.png',out)
                k+=1

    print("Inflection Tortuosity : ")
    print(mean(torts))
    tortuos = (1-mean(torts)/180)
    print(tortuos)

    print("Arc Tortuosity : ")
    tortuos = mean(arc_torts)
    print(tortuos)

img = cv2.imread('branched.png',0)
tortuosity(img)