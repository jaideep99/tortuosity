import cv2
import numpy as np
from numpy.polynomial import Polynomial
import os
from scipy.interpolate import interp1d
from inflections import *
import matplotlib.pyplot as plt

f = os.listdir('roi')

for file in f:

    img = cv2.imread('roi/'+file,0)
    r,c = img.shape
    order = order_points(img)
    if(order[0][1]!=1):
        order = order[::-1]
    order = np.array(order)
    print(order.shape)
    y = r - order[:,0]
    x = c- order[:,1]
    # print(x,y)
    cv2.imshow('img',img)
    xnew = np.linspace(min(x), max(x), num=41, endpoint=True)
    # p = np.poly1d(np.polyfit(x, y, 30))
    # p = interp1d(x, y)
    p = Polynomial.fit(x,y,10)
    # plt.plot(x, y, 'o')
    plt.plot(x, y, 'o', xnew, p(xnew), '-')
    plt.show()
    cv2.waitKey(0)