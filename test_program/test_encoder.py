import encoder
weights=encoder.get_layer_weights_from_npy()
for i in range(len(weights)):
    kernel, bias = weights[i]
    print(kernel,bias)
