import cv2
import numpy as np
import os
import imageio
from utils import *
from skimage import morphology,img_as_bool


img = cv2.imread('output/pred_21_training.png',cv2.IMREAD_GRAYSCALE)
mask = imageio.imread('datasets/train/mask/21_training_mask.gif')
mask = border(mask,592)

img = cv2.bitwise_and(img,img,mask=mask)


ret,thresh = cv2.threshold(img,64,255,cv2.THRESH_BINARY)


imbool = img_as_bool(thresh)
skel = morphology.skeletonize(imbool)
skel.dtype = np.uint8
skel = skel*255

canny = cv2.Canny(thresh,0,255)

hyb = cv2.bitwise_or(canny,skel)

print('branching..')
bpoints = branchpoints(skel.copy(),thresh)

output = cv2.cvtColor(skel.copy(),cv2.COLOR_GRAY2BGR)
# output = skel.copy()
# for i,j in bpoints:
#     output[j,i] = 0

for i,j in bpoints:
    cv2.circle(output,(i,j),1,[0,0,255],-1)

cv2.imwrite('skel.png',skel)

cv2.imshow('img',img)
cv2.imshow('thresh',thresh)
cv2.imshow('skel2',skel)
cv2.imshow('hyb',hyb)
cv2.imshow('bpoints',output)
cv2.waitKey(0)