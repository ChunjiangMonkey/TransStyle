import numpy as np
try:
    weights = np.load("../models_data/vgg19_normalised.npz")
    for i in range(len(np.array(weights))):
        print(weights['arr_%d' % i].shape)
except Exception:
    print("错误！")