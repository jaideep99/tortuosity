import cv2
from SA_UNet import *
import numpy as np
import os
from utils import *

data_loc = 'datasets/test/images/'

files = os.listdir(data_loc)
data = []
size=592

i=0
for xname in files:
  print(i)

  image = cv2.imread(data_loc + xname)
  x = border(image,592)

  data.append(x)
  i+=1

data = np.array(data)
data = data.astype('float32') / 255.
data = np.reshape(data, (len(data), size, size, 3))

model = SA_UNet(input_size=(size,size,3),start_neurons=16,lr=0.001,keep_prob=0.82,block_size=7)
model.load_weights('model/model.h5')

pred = model.predict(data)

for i in range(len(pred)):
    name = files[i]
    p = pred[i]
    p = cv2.normalize(p,p,0,255,cv2.NORM_MINMAX)

    path = 'test_output/'+'pred_'+name[:-4]+'.png'
    print(path)

    cv2.imwrite(path,p)
