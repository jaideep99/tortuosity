from math import inf
import cv2
import numpy as np
import os
import math
from threading import Thread

def border(image,size):
  cur = image.shape[:2]
  dw = size - cur[1]
  dh = size - cur[0]

  top, bottom = dh // 2, dh - (dh // 2)
  left, right = dw // 2, dw - (dw // 2)

  color = [0, 0, 0]
  new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
  
  x = cv2.resize(new_im,(size,size))

  return x

def denoise(thresh,thresh_area):
    edges = cv2.Canny(thresh,0,255)

    ret,threshs = cv2.threshold(edges,200,255,0)
    cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    outer = thresh.copy()


    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) 
    rect_areas = []
    for c in cnts:   
        (x, y, w, h) = cv2.boundingRect(c)
        rect_areas.append(w * h)
    avg_area = np.mean(rect_areas)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cnt_area = w * h
        if cnt_area<thresh_area:
            outer[y:y + h, x:x + w] = 0

    return outer

def edgesmoothing(thresh):
    blur = cv2.pyrUp(thresh)

    for i in range(15):
        blur = cv2.medianBlur(blur,5)

    blur = cv2.pyrDown(blur)
    ret,ths = cv2.threshold(blur,30,255,cv2.THRESH_BINARY)

    return ths

def skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv2.threshold(img,0,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    
    return skel

points = []
def branchpoints(img,thresh):

    global points

    r,c = img.shape
    
    # output = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    img[img==255] = 1

    ranges = [[1,148],[148,295],[295,442],[442,591]]

    threads = []
    i = 0
    for x,y in ranges:
        t = Thread(target=branchparallel,args=(img,x,y,))
        t.start()
        threads.append(t)
    
    for i in range(len(threads)):
        threads[i].join()

    return points


    # kernels = [np.array([[1,0,1],[0,1,0],[0,1,0]],dtype=np.float32),
    #             np.array([[0,1,0],[1,1,1],[0,0,0]],dtype=np.float32),
    #             np.array([[0,1,0],[0,1,1],[1,0,0]],dtype=np.float32),
    #             np.array([[1,0,1],[0,1,0],[0,0,1]],dtype=np.float32)]

    # kernels = get_rotations(kernels)
    # points = []
    # for i in range(1,r-1):
    #     for j in range(1,c-1):

    #         roi = img[i-1:i+2, j-1 : j+2]

    #         flag = 0

    #         for k in kernels:

    #             p = np.sum(k)

    #             r = np.sum(np.multiply(roi,k))

    #             if(r==p):
    #                 flag=1
    #                 break

    #         if flag==1:
    #             points.appen([j,i])
    #             # cv2.circle(output,(j,i),2,[0,0,255],-1)

    # return points

def branchparallel(img,s,e):

    r,c = img.shape
    img[img==255] = 1


    kernels = [np.array([[1,0,1],[0,1,0],[0,1,0]],dtype=np.float32),
                np.array([[0,1,0],[1,1,1],[0,0,0]],dtype=np.float32),
                np.array([[0,1,0],[0,1,1],[1,0,0]],dtype=np.float32),
                np.array([[1,0,1],[0,1,0],[0,0,1]],dtype=np.float32)]

    kernels = get_rotations(kernels)
    for i in range(s,e):
        for j in range(1,c-1):

            roi = img[i-1:i+2, j-1 : j+2]

            flag = 0

            for k in kernels:

                p = np.sum(k)

                r = np.sum(np.multiply(roi,k))

                if(r==p):
                    flag=1
                    break

            if flag==1:
                points.append([j,i])

    print('thread ended')



def get_rotations(kerns = None):

    kernels = []

    for x in kerns:

        kernels.append(x)
        
        x = np.rot90(x)

        kernels.append(x)

        x = np.rot90(x)

        kernels.append(x)
        
        x = np.rot90(x)

        kernels.append(x)

    return kernels

def setclahe(image,c,t):
    clahe = cv2.createCLAHE(clipLimit=c,tileGridSize=(t,t))
    res = clahe.apply(image)
    return res

def distance(a,b):
    res = (b[0]-a[0])**2 + (b[1]-a[1])**2
    res = math.sqrt(res)

    return res



def get_inflections(img):
    
    kernels = [np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32),
                np.array([[0,1,0],[0,1,0],[0,1,0]],dtype=np.float32),
                np.array([[0,0,1],[0,1,0],[1,0,0]],dtype=np.float32),
                np.array([[0,0,0],[1,1,1],[0,0,0]],dtype=np.float32)]

    r,c = img.shape
    img[img==255] = 1


    points = []
    for i in range(1,r-1):
        for j in range(1,c-1):

            if(img[i,j]==1):

                roi = img[i-1:i+2, j-1 : j+2]

                flag = 0

                for k in kernels:

                    p = np.sum(k)

                    r = np.sum(np.multiply(roi,k))

                    print(roi)
                    print(k)

                    if(r==p):
                    
                        flag=1
                        break

                if flag==0:
                    points.append([i,j])

    return points        

def get_angle(a,b,c):

    ab = distance(a,b)
    bc = distance(b,c)

    angle = (a[0]*b[0]) + (a[1]*b[1])
    angle = angle/(ab*bc)

    angle = math.acos(angle)

    return angle    


