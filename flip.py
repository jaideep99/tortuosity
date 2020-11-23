import cv2
import os
# Please modify the path
path="datasets/training/images/"
lpath = "datasets/training/label/"
save="data/train/image/"
savelabels = "data/train/label/"
for name,lname in zip(os.listdir(path),os.listdir(lpath)):
    image = cv2.imread(path+name)
    label = cv2.imread(lpath+lname)

    cv2.imwrite(save+"o"+name, image)
    cv2.imwrite(savelabels+"o"+lname, label)

    # Flipped Horizontally
    h_flip = cv2.flip(image, 1)
    h_flip_label = cv2.flip(label, 1)
    cv2.imwrite(save+"h"+name, h_flip)
    cv2.imwrite(savelabels+"h"+lname, h_flip_label)

    v_flip = cv2.flip(image, 0)
    v_flip_label = cv2.flip(label, 0)
    cv2.imwrite(save+"v"+name, v_flip)
    cv2.imwrite(savelabels+"v"+lname, v_flip_label)


    hv_flip = cv2.flip(image, -1)
    hv_flip_label = cv2.flip(label, -1)
    cv2.imwrite(save+"hv"+name, hv_flip)
    cv2.imwrite(savelabels+"hv"+lname, hv_flip_label)
