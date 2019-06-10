import numpy as np

from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize


def list_images(directory):
    images = []
    for file in listdir(directory):
        image_file_name = join(directory, file)
        try:
            # wikiArt文件一般都较大,需要读取
            image = imread(image_file_name, mode="RGB")
            width, height, channel = image.shape
            images.append(image_file_name)
        except Exception:
            print(Exception)
            print(image_file_name)
    return images


def get_resized_images(paths):
    # 调整图片大小，见论文：
    # During training, we first resize
    # the smallest dimension of both images to 512 while preserving
    # the aspect ratio, then randomly crop regions of size
    # 256 * 256
    images = []
    idx = 0
    for image_path in paths:
        idx += 1
        try:
            image = imread(image_path, mode="RGB")
            height, width, _ = image.shape
            if idx % 10 == 0:
                print(idx)
                print(image.shape)
            if height < width:
                new_height = 512
                new_width = int(width / height * 512)
            else:
                new_width = 512
                new_height = int(height / width * 512)
            image = imresize(image, [new_height, new_width], interp='nearest')
            start_h = np.random.choice(new_height - 256 + 1)
            start_w = np.random.choice(new_width - 256 + 1)
            image = image[start_h:(start_h + 256), start_w:(start_w + 256), :]
            images.append(image)
        except Exception:
            print(image_path)

    images = np.stack(images, axis=0)
    return images


def get_orginal_image(paths):
    print(paths)
    images = []
    for image_path in paths:
        print(image_path)
        image = imread(image_path, mode="RGB")
        images.append(image)

    images = np.stack(images, axis=0)

    return images


def get_one_orginal_image(path):
    images=[]
    image = imread(path, mode="RGB")
    images.append(image)
    images=np.array(images)
    print(images.shape)
    return images


def save_images(datas, contents_path, styles_path, save_dir, suffix=None):
    if not exists(save_dir):
        mkdir(save_dir)

    if suffix is None:
        suffix = ''

    data_idx = 0
    for content_path in contents_path:
        for style_path in styles_path:
            data = datas[data_idx]
            data_idx += 1

            content_path_name, content_ext = splitext(content_path)
            style_path_name, style_ext = splitext(style_path)

            content_name = content_path_name.split(sep)[-1]
            style_name = style_path_name.split(sep)[-1]

            save_path = join(save_dir, '%s-%s%s%s' %
                             (content_name, style_name, suffix, content_ext))

            imsave(save_path, data)
