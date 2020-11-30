from types import FrameType
import cv2
import numpy as np
from utils import *
np.set_printoptions(threshold=np.inf)

def padding(img,size):

    hh,ww = size,size
    ht,wd = img.shape
    xx = max(ww - wd,0)
    yy = max(hh - ht,0)

    yy = yy // 2
    xx = xx // 2

    color = 0

    new_im = np.zeros((size,size),dtype = np.uint8)

    new_im[yy:yy+ht , xx:xx+wd] = img
    # new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #                                 value=color)
    
    # x = cv2.resize(new_im,(size,size))

    return new_im

def regression_slope(x,y):
    
    n = len(x)

    xy = np.sum(x*y)
    xsq = np.sum(x**2)
    
    slope = (n*xy) - (np.sum(x)*np.sum(y))
    slope = slope/ ((xsq*n) - ((np.sum(x))**2))

    return slope

def slope(x,y):

    x0,y0 = x[0],y[0]
    x1,y1 = x[-1],y[-1]

    slope = (y1-y0)/(x1-x0)

    return slope


def fractal_dimension(img):

    r,c = img.shape

    temp = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)

    grids = []

    tolerance = min(r,c)
    g = 1
    while(g<tolerance):
        grids.append(g)
        g = g*2

    grids = grids[::-1]

    img[img>0] = 1
    
    box_counts = []

    for gsize in grids:

        gtemp = temp.copy()
        count = 0

        rowranges = list(range(0,r,gsize))
        colranges = list(range(0,c,gsize))

        if rowranges[-1]!= r:
            rowranges.append(r)
        
        if colranges[-1]!=c:
            colranges.append(c)

        for i in range(len(rowranges)-1):
            for j in range(len(colranges)-1):
                
                x,xh = rowranges[i],rowranges[i+1]
                y,yh = colranges[j],colranges[j+1]

                roi = img[x:xh,y:yh]

                # gtemp = cv2.rectangle(gtemp,(y,x),(yh,xh),(0,0,255),1)

                if 1 in roi:
                    count+=1

                # if gsize==8:
                #     print(roi)
                #     print(1 in roi)

        # cv2.imshow('grid',gtemp)
        # cv2.waitKey(0)

        box_counts.append(count)

    # print(grids,box_counts)

    fdimension = regression_slope(np.log(grids),np.log(box_counts))

    return fdimension
    

# image = cv2.imread('skel.png')
# print(image)
# fd = fractal_dimension(image)

# print(fd)