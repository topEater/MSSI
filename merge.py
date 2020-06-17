from libs.MSSI import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model, low_model, high_model = MSSI_Model_parts()
model.load_weights("./model/MSSI_random_val_loss_best_weights.h5")

# model.summary()
# low_model.summary()
# high_model.summary()

features = Model(inputs=high_model.input,
                 outputs=high_model.get_layer('add').output)
terrains = Model(inputs=high_model.input, outputs=high_model.output)

sample_high = np.load("./data/terrain-41-test.npy")
mask = np.load("./data/mask-50.npy")

sample_low = np.multiply(mask, sample_high)

count = 10000

for i in range(count):
    j = np.random.randint(0, count)
    low = low_model.predict([sample_low[j:j+1, :, :, :], mask[j:j+1, :, :, :]])
    feature = features.predict([low[:, :, :, :], mask[j:j+1, :, :, :]])
    terrain = terrains.predict([low[:, :, :, :], mask[j:j+1, :, :, :]])
    image = np.ones((8*41, 8*41))
    for j in range(64):
        row = 0+np.floor_divide(j, 8)*41
        col = 0+np.mod(j, 8)*41
        image[row:row+41, col:col+41] = feature[0, :, :, j]
    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.axis('off') 
    plt.subplot(122)
    plt.imshow(terrain[0,:,:,0], cmap="gray")
    plt.axis('off') 
    plt.show()

