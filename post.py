import cv2
import numpy as np
import os
import imageio
from utils import *
from skimage import morphology,img_as_bool
from tort import *
from fractals import *
import xlsxwriter

odir = 'test_output/'
mdir = 'datasets/test/mask/'
vfiles = os.listdir(odir)
mfiles = os.listdir(mdir)



workbook = xlsxwriter.Workbook('test_results.xlsx')
worksheet = workbook.add_worksheet()
worksheet.set_default_row(20)

worksheet.write('A1','Fundus Image')
worksheet.write('B1','Mean Angle')
worksheet.write('C1','Angle Tortuosity')
worksheet.write('D1','Arc-Length Tortuosity')
worksheet.write('E1','Fractal Dimension')

def write_book(n,impath,theta,tort,arc_tort,fd):
    worksheet.insert_image('A'+str(n),impath,{'x_scale': 0.4, 'y_scale': 0.4})
    worksheet.write('B'+str(n), str(theta))
    worksheet.write('C'+str(n), str(tort))
    worksheet.write('D'+str(n), str(arc_tort))
    worksheet.write('E'+str(n), str(fd))


p = 1
index = 2
for v,m in zip(vfiles,mfiles):
    print('Image : ',str(p))
    impath = odir+ 'pred_'+str(p)+'_test.png'
    maskpath = mdir+str(p)+'_test_mask.gif'

    img = cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
    mask = imageio.imread(maskpath)
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

    print('computing tortuosity..')
    angle, tort, arc_tort = tortuosity(segmented)
    
    print('computing fractal dimension')
    fdimension = -1* fractal_dimension(skel.copy())

    print('writing to workbook..')


    write_book(index,impath,angle,tort,arc_tort,fdimension)

    p+=1
    index+=1

workbook.close()
