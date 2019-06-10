import numpy as np
import matplotlib.pyplot as plt

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize


def list_images_with_error(directory):
    idx,bad_inx=0,0
    images_list = []
    for file in listdir(directory):
        image_file_name = join(directory, file)
        try:
            image = imread(image_file_name, mode="RGB")
            width,height,channel=image.shape
            images_list.append(image_file_name)
            idx+=1
            print("idx",idx)

        except Exception:
            print(Exception)
            # bad_inx+=1
            # print("bad_inx:",bad_inx)
            print(image_file_name)

    return images_list


def list_images(directory):
    idx,bad_inx=0,0
    images_list = []
    for file in listdir(directory)[49:55]:
        image_file_name = join(directory, file)
        image = imread(image_file_name, mode="RGB")
        width,height,channel=image.shape
        images_list.append(image_file_name)
        idx+=1
        print("idx",idx)
    return images_list


DIR="E:\\dataSet_of_ML\\big_image"

image_list=list_images(DIR)