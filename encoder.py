import numpy as np
import tensorflow as tf

# 论文原文对于encoder的阐述：
# We adopt a simple encoder-decoder architecture,
# in which the encoder f is fixed to the first few
# layers(up to relu4_1) of a pre-trained VGG-19.
# 其余均与vgg-19相同

# ImageNet数据集图片像素均值
mean_pixel = np.array([103.939, 116.779, 123.68])
# .npz文件是以字典格式保存的vgg网络权重参数
weights_path = "models_data/vgg19_normalised.npz"

# 定义了encoder的结构
encoder_layers = (
    "conv1_1", "relu1_1",
    "conv1_2", "relu1_2",
    "pool1",
    "conv2_1", "relu2_1",
    "conv2_2", "relu2_2",
    "pool2",
    "conv3_1", "relu3_1",
    "conv3_2", "relu3_2",
    "conv3_3", "relu3_3",
    "conv3_4", "relu3_4",
    "pool3",
    "conv4_1", "relu4_1"
)


# 使用ImageNet训练的网络时需要对图片进行标准化
def img_preprocess(image):
    return image - mean_pixel


# 还原图片
def img_deprocess(image):
    return image + mean_pixel


# 卷积函数
def conv2d(img, kernel, bias):
    img = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    img = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='VALID')
    img = tf.nn.bias_add(img, bias)
    return img


# 池化函数
def pool2d(img):
    img = tf.nn.max_pool(img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return img


# 从.npy文件中读取数据
def get_layer_weights_from_npy():
    layer_weights = []
    weights = np.load(weights_path)
    index = 0
    # 只需读取conv层参数,pool和relu都为函数
    for layer in encoder_layers:
        kind = layer[:4]
        if kind == "conv":
            kernel = np.transpose(weights['arr_%d' % index], [2, 3, 1, 0]).astype(np.float32)
            bias = weights['arr_%d' % (index + 1)].astype(np.float32)
            index += 2
            W = tf.Variable(kernel, trainable=False)
            b = tf.Variable(bias, trainable=False)
            layer_weights.append((W, b))
            # layer_weights.append((kernel, bias))
    return np.array(layer_weights)


# 通过读取的权重数组创建网络
def encode(img):
    layer_weights = get_layer_weights_from_npy()
    layer_dict = {}
    index = 0
    for layer in encoder_layers:
        kind = layer[:4]
        if kind == "conv":
            kernel, bias = layer_weights[index]
            index += 1
            img = conv2d(img, kernel, bias)
        elif kind == "relu":
            img = tf.nn.relu(img)
        elif kind == "pool":
            img = pool2d(img)

        layer_dict[layer] = img

    if len(layer_dict) != len(encoder_layers):
        print("ERROR by encode!")
    # 返回一个layer_dict以计算style损失
    return layer_dict[encoder_layers[-1]], layer_dict
