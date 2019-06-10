from get_data import list_images, get_resized_images
import numpy as np

TRAINING_CONTENT_DIR = "E:\\dataSet_of_ML\\MS_coco"
TRAINING_STYLE_DIR = "E:\\dataSet_of_ML\\WikiArt"

content_img_path = list_images(TRAINING_STYLE_DIR)
batch_size = 100
np.random.shuffle(content_img_path)
i = 0
# 蒙特卡洛模拟找bug图法
for i in range(len(content_img_path) // batch_size):
    index = np.random.randint(0, len(content_img_path) - batch_size)
    sub_list = content_img_path[index:index + batch_size]
    sub_imgs = get_resized_images(sub_list)
