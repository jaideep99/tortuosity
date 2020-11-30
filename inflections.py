import cv2
import numpy as np
import math
from utils import *

def dfs(mat,i,j):

    visit[i][j] = 1

    if (j!=0 and visit[i][j-1]==0 and mat[i][j-1]!=0):
        return (i,j-1)

    if (j+1<c and visit[i][j+1]==0 and mat[i][j+1]!=0):
        return (i,j+1)

    if (i-1>=0 and visit[i-1][j]==0 and mat[i-1][j]!=0):
        return (i-1,j)
    
    if (i+1<r and  visit[i+1][j]==0 and mat[i+1][j]!=0):
        return (i+1,j)

    if (i-1>=0 and j-1>=0 and visit[i-1][j-1]==0 and mat[i-1][j-1]!=0):
        return (i-1,j-1)
        

    if (i-1>=0 and j+1<c and visit[i-1][j+1]==0 and mat[i-1][j+1]!=0):
        return (i-1,j+1)

    if (i+1<r and j-1>=0 and visit[i+1][j-1]==0 and mat[i+1][j-1]!=0):
        return (i+1,j-1)
    if (i+1<r and j+1<c and visit[i+1][j+1]==0 and mat[i+1][j+1]!=0):
        return (i+1,j+1)

    return (-1,-1)



def start_point(img,r,c):
    im = img.copy()

    im[im==255] = 1
    dummy = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
    for i in range(1,r-1):
        for j in range(1,c-1):
            if im[i][j]==1:
                roi = im[i-1:i+2, j-1:j+2]
                p = np.sum(roi)
                if(p==2):
                    dummy[i][j] = [0,0,255]
                    return (i,j)

def order_points(img):
    global r,c,visit
    r,c = img.shape
    visit = np.zeros((r,c))
    x,y = start_point(img,r,c)

    flag = True
    points = [[x,y]]

    while(flag):
        x,y = dfs(img,x,y)
        if x!=-1 and y!=-1:
            points.append([x,y])
        else:
            flag=False

    return points

def getinflections(img,points):
    kernels = [np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32),
                np.array([[0,1,0],[0,1,0],[0,1,0]],dtype=np.float32),
                np.array([[0,0,1],[0,1,0],[1,0,0]],dtype=np.float32),
                np.array([[0,0,0],[1,1,1],[0,0,0]],dtype=np.float32)]

    pts = []
    img[img==255] = 1

    for i,j in points:
        roi = img[i-1:i+2, j-1:j+2]
        flag = 0
        for k in kernels:
            p = np.sum(k)
            r = np.sum(np.multiply(roi,k))
            if(r==p):
                flag=1
                break

        if flag==0:
            pts.append([i,j])

    return pts


def compute_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def get_angles(inflects):
    if(len(inflects)<3):
        return [180]

    n = len(inflects)
    angles = []
    for i in range(n-2):
        angles.append(compute_angle(inflects[i],inflects[i+1],inflects[i+2]))

    return angles

    