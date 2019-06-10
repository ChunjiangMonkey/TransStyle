import tensorflow as tf

from encoder import encode, img_preprocess, img_deprocess
from decoder import decode
from adaptive_instance_normalization import adaIn


def style_trans(content, style):
    # 使图片前向通过一次网络
    content, style = img_preprocess(content), img_preprocess(style)
    encoded_c, _ = encode(content)
    encoded_s, _ = encode(style)
    features = adaIn(encoded_c, encoded_s)
    generated_img = decode(features)
    generated_img = img_deprocess(generated_img)
    generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)
    return generated_img


def get_features(content, style):
    #获得特征
    content, style = img_preprocess(content), img_preprocess(style)
    encoded_c, _ = encode(content)
    encoded_s, _ = encode(style)
    return adaIn(encoded_c, encoded_s)


def through_encoder(img):
    #处理后通过encoder
    img = img_preprocess(img)
    return encode(img)
