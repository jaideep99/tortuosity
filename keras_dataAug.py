
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
import cv2
# logger = logging.getLogger(__name__)
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:


    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):

        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)


    @staticmethod
    def randomCrop(image, label):

        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label

    @staticmethod
    def randomColor(image, label):

        random_factor = np.random.randint(0, 31) / 10.  
        color_image = ImageEnhance.Color(image).enhance(random_factor)  
        random_factor = np.random.randint(10, 21) / 10.  
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
        random_factor = np.random.randint(10, 21) / 10.  
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
        random_factor = np.random.randint(0, 31) / 10.  
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):

        def gaussianNoisy(im, mean=0.2, sigma=0.3):

            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.array(image)
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def saveImage(image, path):
        # open_cv_image = np.array(image)
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 
        # cv2.imwrite(open_cv_image,path)
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception:
        print(str(Exception))



def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=3):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    
    funcshort = {"randomRotation": 'rr',
               "randomCrop": 'rcr',
               "randomColor": 'rc',
               "randomGaussian": 'rg'
               }

    for _i in range(0, times, 1):
        new_image, new_label = funcMap[func_name](image, label)
        sname = funcshort[func_name]
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, sname + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, sname + str(_i) + label_file_name))


opsList = {"randomRotation", "randomColor", "randomGaussian"}


def threadOPS(img_path, new_img_path, label_path, new_label_path):

    img_names = os.listdir(img_path)
    label_names = os.listdir(label_path)

    n = len(img_names)

    for i in range(n):
        img_name = img_names[i]

        label_name = label_names[i]


        tmp_img_path = os.path.join(img_path, img_name)
        tmp_label_path = os.path.join(label_path, label_name)

        print(tmp_img_path)
        image = DataAugmentation.openImage(tmp_img_path)

        label = DataAugmentation.openImage(tmp_label_path)

        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            imageOps(ops_name,image, label, new_img_path, new_label_path, img_name,label_name)

if __name__ == '__main__':
    threadOPS("datasets\\training\\images",
              "data\\train\\image",
              "datasets\\training\\label",
              "data\\train\\label")
