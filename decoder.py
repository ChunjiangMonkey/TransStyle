import tensorflow as tf


# encoder每层维度如下：
# (64, 3, 3, 3)
# (64,)
# (64, 64, 3, 3)
# (64,)
# (128, 64, 3, 3)
# (128,)
# (128, 128, 3, 3)
# (128,)
# (256, 128, 3, 3)
# (256,)
# (256, 256, 3, 3)
# (256,)
# (256, 256, 3, 3)
# (256,)
# (256, 256, 3, 3)
# (256,)
# (512, 256, 3, 3)
# (512,)
# (512, 512, 3, 3)
# 根据论文：The decoder mostly mirrors the encoder, with all pooling
# layers replaced by nearest up-sampling to reduce checkerboard
# effects. 可以推出decoder网络的结构

def get_weights(input_filters, output_filters, kernel_size, scope):
    # 若不加入scope，TensorFlow创建同名变量时会报错
    with tf.variable_scope(scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=shape,
                                 name='kernel')
        bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=[output_filters],
                               name='bias')
        return (kernel, bias)


def get_layer_weights():
    layer_weights = []
    layer_weights.append(get_weights(512, 256, 3, scope='conv4_1'))
    layer_weights.append(get_weights(256, 256, 3, scope='conv3_4'))
    layer_weights.append(get_weights(256, 256, 3, scope='conv3_3'))
    layer_weights.append(get_weights(256, 256, 3, scope='conv3_2'))
    layer_weights.append(get_weights(256, 128, 3, scope='conv3_1'))
    layer_weights.append(get_weights(128, 128, 3, scope='conv2_2'))
    layer_weights.append(get_weights(128, 64, 3, scope='conv2_1'))
    layer_weights.append(get_weights(64, 64, 3, scope='conv1_2'))
    layer_weights.append(get_weights(64, 3, 3, scope='conv1_1'))
    return layer_weights


def conv2d(x, kernel, bias):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out


def upsample(x, scale=2):
    height = tf.shape(x)[1] * scale
    width = tf.shape(x)[2] * scale
    # 使用最近上采样
    # 论文：with all pooling layers replaced by nearest up-sampling to reduce checkerboard effects
    output = tf.image.resize_images(x, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output


def decode(image):
    upsample_id = (0, 4, 6)
    layer_weights = get_layer_weights()
    out = image
    for i in range(len(layer_weights) - 1):
        kernel, bias = layer_weights[i]
        out = tf.nn.relu(conv2d(out, kernel, bias))
        if i in upsample_id:
            out = upsample(out)
    # 最后一层不使用relu
    kernel, bias = layer_weights[len(layer_weights) - 1]
    out = conv2d(out, kernel, bias)
    return out
