import cv2
import numpy as np

img = cv2.imread('branched.png',0)

canny = cv2.Canny(img,0,255)
contours,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # output = cv2.drawContours(cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR),[cnt],-1,(0,0,255),1)
    output = cv2.drawContours(cv2.cvtColor(np.zeros(img.shape,dtype = np.uint8),cv2.COLOR_GRAY2BGR),[cnt],-1,(255,255,255),1)
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # Box = np.int0(box) # Get rectangular corner points
    # area = cv2.contourArea(box)
    # width = rect[1][0]
    # height = rect[1][1]
    # cv2.polylines(output, np.int32([box]), True, (0, 255, 0), 1)
    cv2.imshow('output',output)
    cv2.waitKey(0)

cv2.imshow('canny',canny)


cv2.waitKey(0)
cv2.destroyAllWindows()