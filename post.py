import cv2
import numpy as np
import os
import imageio
from utils import *
from skimage import morphology,img_as_bool
from tort import *
from fractals import *

img = cv2.imread('output/pred_22_training.png',cv2.IMREAD_GRAYSCALE)
mask = imageio.imread('datasets/train/mask/22_training_mask.gif')
mask = border(mask,592)

img = cv2.bitwise_and(img,img,mask=mask)

ret,thresh = cv2.threshold(img,64,255,cv2.THRESH_BINARY)


imbool = img_as_bool(thresh)
skel = morphology.skeletonize(imbool)
skel.dtype = np.uint8
skel = skel*255

print('branching..')
bpoints = branchpoints(skel.copy(),thresh)

segmented = skel.copy()
for i,j in bpoints:
    segmented[j,i] = 0

tortuosity(segmented)
fdimension = -1* fractal_dimension(skel.copy())
print('\n\n'+str(fdimension))

cv2.imshow('img',img)
cv2.imshow('thresh',thresh)
cv2.imshow('skel',skel)
cv2.waitKey(0)