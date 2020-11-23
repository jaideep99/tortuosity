import cv2
import math
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

ddir = "C:\\Users\\jaide\\OneDrive\\Desktop\\Tortuosity\\datasets\\training\\images"
mask_dir = "C:\\Users\\jaide\\OneDrive\\Desktop\\Tortuosity\\datasets\\training\\mask"
manual_dir = "C:\\Users\\jaide\\OneDrive\\Desktop\\Tortuosity\\datasets\\training\\1st_manual"

tdir = "C:\\Users\\jaide\\OneDrive\\Desktop\\Tortuosity\\datasets\\test\\images"
tmdir = "C:\\Users\\jaide\\OneDrive\\Desktop\\Tortuosity\\datasets\\test\\mask"

train,masks,out = os.listdir(ddir),os.listdir(mask_dir),os.listdir(manual_dir)

test,tmask = os.listdir(tdir),os.listdir(tmdir)

def green(img):
    b,g,r = cv2.split(img)
    return g

def resized(img,scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)

    res = cv2.resize(img,(width,height))
    return res


def setclahe(image,c,t):
    clahe = cv2.createCLAHE(clipLimit=c,tileGridSize=(t,t))
    res = clahe.apply(image)
    return res

def kirsch_filter(gray,mask):
    if gray.ndim > 2:
        raise Exception("illegal argument: input must be a single channel image (gray)")
    kernelG1 = np.array([[ 5,  5,  5],
                         [-3,  0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG2 = np.array([[ 5,  5, -3],
                         [ 5,  0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[ 5, -3, -3],
                         [ 5,  0, -3],
                         [ 5, -3, -3]], dtype=np.float32)
    kernelG4 = np.array([[-3, -3, -3],
                         [ 5,  0, -3],
                         [ 5,  5, -3]], dtype=np.float32)
    kernelG5 = np.array([[-3, -3, -3],
                         [-3,  0, -3],
                         [ 5,  5,  5]], dtype=np.float32)
    kernelG6 = np.array([[-3, -3, -3],
                         [-3,  0,  5],
                         [-3,  5,  5]], dtype=np.float32)
    kernelG7 = np.array([[-3, -3,  5],
                         [-3,  0,  5],
                         [-3, -3,  5]], dtype=np.float32)
    kernelG8 = np.array([[-3,  5,  5],
                         [-3,  0,  5],
                         [-3, -3, -3]], dtype=np.float32)

    g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    magn = cv2.max(
        g1, cv2.max(
            g2, cv2.max(
                g3, cv2.max(
                    g4, cv2.max(
                        g5, cv2.max(
                            g6, cv2.max(
                                g7, g8
                            )
                        )
                    )
                )
            )
        )
    )

    magn = cv2.bitwise_and(magn,magn,mask=mask)
    return magn

def gradient(gray,mask):

    scales = [2,4,6,8,10,12,14,16]

    gamma = []

    exist = 0

    for s in scales:
        
        L = Lderivative(gray,s)

        locGamma = L/s

        if(exist == 0):
            gamma = locGamma
            exist = 1
        else:
            gamma = cv2.max(gamma,locGamma)

    gradabs = np.uint8(gamma)
    gradabs = cv2.bitwise_and(gradabs,gradabs,mask=mask)

    return gradabs


def Lderivative(image,s):
    ddepth = cv2.CV_64F
    gradx = cv2.Sobel(image,ddepth,dx=1,dy=0,scale=s)
    grady = cv2.Sobel(image,ddepth,dx=0,dy=1,scale=s)

    delL = cv2.sqrt(cv2.pow(gradx,2) + cv2.pow(grady,2))

    return delL

def eigenmax(image,mask):
    dervs = hessian_matrix(image, sigma=1.0, order='rc')
    emax,emin = hessian_matrix_eigvals(dervs)

    emax = cv2.normalize(emax,emax,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    emax = cv2.bitwise_and(emax,emax,mask=mask)

    emin = cv2.normalize(emin,emin,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    emin = cv2.bitwise_and(emin,emin,mask=mask)

    return emax,emin

def distance(a,b):

    dist = math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return int(dist)

def featuremaker(fundus,mask,manual=None,manin=True):

    gfundus = green(fundus)

    # homo_filter = HomomorphicFilter(a=0.75,b=1.25)
    # img_filtered = homo_filter.filter(I=gfundus, filter_params=[30,2])

    # gfundus = img_filtered

    clahe = setclahe(gfundus,3.0,16)
    clahe = cv2.bitwise_and(clahe,clahe,mask=mask)
    kirschs = kirsch_filter(clahe,mask)

    gradabs = gradient(clahe,mask)
    emax = eigenmax(clahe,mask)

    r,c = clahe.shape
    cr,cc = r//2,c//2

    imgdata = []

    for x in range(r):
        for y in range(c):

            featurexy = dict()

            featurexy['intensity'] = clahe[x][y]
            featurexy['gradient'] = gradabs[x][y]
            featurexy['eigen'] = emax[x][y]
            featurexy['kirsch'] = kirschs[x][y]

            if manin == True:
                if manual[x][y]==0:
                    featurexy['target']  = 0
                else:
                    featurexy['target'] = 1

            imgdata.append(featurexy)

    return imgdata

def imloader(index):
    mask = imageio.imread(mask_dir+ "\\"+masks[index])
    fundus = cv2.imread(ddir +  "\\"+train[index])
    manual = imageio.imread(manual_dir +  "\\"+out[index])

    return fundus,mask,manual

def testimloader(index):
    mask = imageio.imread(tmdir+ "\\"+tmask[index])
    fundus = cv2.imread(tdir +  "\\"+test[index])

    return fundus,mask

def getleft():
    a = np.zeros((3,3),dtype = np.uint8)
    np.fill_diagonal(a,1)
    return a

def getright():
    a = np.zeros((3,3),dtype = np.uint8)
    np.fill_diagonal(a,1)
    a = np.flip(a,axis = 1)
    return a

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

def featureprinter(fundus,mask):
    gfundus = green(fundus)

    homo_filter = HomomorphicFilter(a=0.75,b=1.25)
    img_filtered = homo_filter.filter(I=gfundus, filter_params=[30,2])

    gfundus = img_filtered

    clahe = setclahe(gfundus,3.0,16)
    clahe = cv2.bitwise_and(clahe,clahe,mask=mask)
    kirschs = kirsch_filter(clahe,mask)

    gradabs = gradient(clahe,mask)
    emax = eigenmax(clahe,mask)

    cv2.imshow('homo',gfundus)
    cv2.imshow('clahe',clahe)
    cv2.imshow('kirsch',kirschs)
    cv2.imshow('grads',gradabs)
    cv2.imshow('eigen',emax)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv2.threshold(img,60,255,0)
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

def miniblur(gray,thr):
    blur = cv2.GaussianBlur(gray,(13,13),0)
    mini = cv2.min(blur,gray)
    return mini

    # thresh = cv2.threshold(mini,thr , 255, cv2.THRESH_BINARY)[1]

    # return thresh

def prep(pts,endpts,shp):

    im = np.zeros(shp,dtype = np.uint8)
    for j,i in pts:
        im[i][j] = 255

    im = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)

    return im