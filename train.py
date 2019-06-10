import numpy as np
import tensorflow as tf

from trans_style_network import style_trans, get_features, through_encoder

from get_data import get_resized_images
from get_data import list_images

# In our experiments we use relu1 1,relu2 1, relu3 1, relu4 1 layers with equal weights
style_loss_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
resized_images_shape = (256, 256, 3)

EPSILON = 1e-5
# 过几遍数据
EPOCHS = 1
# We use the adam optimizer [26] and a batch size of 8 content-style image pairs.
BATCH_SIZE = 8
# 学习率
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0

TRAINING_CONTENT_DIR = "E:\\dataSet_of_ML\\MS_coco"
TRAINING_STYLE_DIR = "E:\\dataSet_of_ML\\WikiArt"
ENCODER_WEIGHTS_PATH = "vgg19_normalised.npz"
LOGGING_PERIOD = 20

STYLE_WEIGHTS = 2.0
MODEL_SAVE_PATHS = 'models_data/weight.ckpt'

print("正在读取可使用的文件列表……时间较长")
content_imgs_path = list_images(TRAINING_CONTENT_DIR)
style_imgs_path = list_images(TRAINING_STYLE_DIR)

num_imgs = min(len(content_imgs_path), len(style_imgs_path))

print("可训练的图片有",num_imgs,"张")
content_imgs_path = content_imgs_path[:num_imgs]
style_imgs_path = style_imgs_path[:num_imgs]
# 保证训练图片数量是BATCH_SIZE的整数倍
mod = num_imgs % BATCH_SIZE
if mod > 0:
    content_imgs_path = content_imgs_path[:-mod]
    style_imgs_path = style_imgs_path[:-mod]

# 定义损失
content = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='content')
style = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3), name='style')
generated_img = style_trans(content, style)
target_features = get_features(content, style)
encoder_gen, encoder_gen_layer = through_encoder(generated_img)

print(target_features.shape)

# 内容损失,计算每个Batch上的二范数（未开方）之和
content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(encoder_gen - target_features), axis=[1, 2]))

# 开始计算风格损失,见式13
style_encoder_gen, style_encoder_gen_layer = through_encoder(style)
style_loss_per_layer = []
for layer in style_loss_layers:
    encoder_style_feat = style_encoder_gen_layer[layer]
    encoder_gen_feat = encoder_gen_layer[layer]

    mean_s, var_s = tf.nn.moments(encoder_style_feat, [1, 2])
    mean_g, var_g = tf.nn.moments(encoder_gen_feat, [1, 2])

    sigmaS = tf.sqrt(var_s + EPSILON)
    sigmaG = tf.sqrt(var_g + EPSILON)
    # 二范数
    l2_mean = tf.reduce_sum(tf.square(mean_g - mean_s))
    l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

    style_loss_per_layer.append(l2_mean + l2_sigma)

style_loss = tf.reduce_sum(style_loss_per_layer)
all_loss = content_loss + STYLE_WEIGHTS * style_loss
# We use the adam optimizer and a batch size of 8 content-style image pairs.
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(all_loss, global_step=global_step)

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    # 开始训练

    step = 0
    # batch数
    n_batches = len(content_imgs_path) // BATCH_SIZE
    # print(len(content_imgs_path))
    # print(len(style_imgs_path))
    # print(n_batches)

    for epoch in range(EPOCHS):
        np.random.shuffle(content_imgs_path)
        np.random.shuffle(style_imgs_path)
        for batch_id in range(n_batches):
            start = batch_id * BATCH_SIZE
            end = start + BATCH_SIZE
            # print("start=", start)
            # print("end=", end)
            content_batch_path = content_imgs_path[start:end]
            style_batch_path = style_imgs_path[start:end]

            content_batch = get_resized_images(content_batch_path)
            style_batch = get_resized_images(style_batch_path)

            sess.run(train_op, feed_dict={content: content_batch, style: style_batch})
            step += 1

            if step % 1000 == 0:
                saver.save(sess, MODEL_SAVE_PATHS, global_step=step, write_meta_graph=False)
            _content_loss, _style_loss, _loss = sess.run([content_loss, style_loss, all_loss],
                                                         feed_dict={content: content_batch, style: style_batch})
            if step % 10 == 0:
                print('step: %d,  total loss: %.3f' % (step, _loss))
                print('content loss: %.3f' % (_content_loss))
                print('style loss  : %.3f,  weighted style loss: %.3f\n' % (_style_loss, STYLE_WEIGHTS * _style_loss))

    saver.save(sess, MODEL_SAVE_PATHS)
