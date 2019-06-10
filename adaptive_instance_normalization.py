import tensorflow as tf

# 定义式(3)、式(6)中的 ε
EPSILON = 1e-5


def adaIn(x, y):
    # 见式(8)
    # tf.nn.moments返回由参数x特定维度上的均值和方差的元组
    mean_x, var_x = tf.nn.moments(x, [1, 2], keep_dims=True)
    mean_y, var_y = tf.nn.moments(y, [1, 2], keep_dims=True)
    stand_x, stand_y = tf.sqrt(var_x + EPSILON), tf.sqrt(var_y + EPSILON)
    return stand_y * ((x - mean_x) / stand_x) + mean_y
