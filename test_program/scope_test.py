i = 1
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

def add_i():
    print(encoder_layers)
    a=i+1
    print(a)
add_i()