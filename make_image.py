import tensorflow as tf
from get_data import list_images, get_orginal_image, save_images, get_one_orginal_image
from trans_style_network import style_trans

INFERRING_CONTENT_DIR = 'image/content'
INFERRING_STYLE_DIR = 'image/style'
OUTPUTS_DIR = 'outputs'

MODEL_SAVE_PATHS = 'models_data/weight.ckpt'

contents_path = list_images(INFERRING_CONTENT_DIR)
styles_path = list_images(INFERRING_STYLE_DIR)

content = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content')
style = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='style')

output_img = style_trans(content, style)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_SAVE_PATHS)

    outputs = []
    for content_path in contents_path:
        content_img = get_one_orginal_image(content_path)
        for style_path in styles_path:
            style_img = get_one_orginal_image(style_path)
            result = sess.run(output_img, feed_dict={content: content_img, style: style_img})
            outputs.append(result[0])
    save_images(outputs, contents_path, styles_path, OUTPUTS_DIR)
    print("成功！")
