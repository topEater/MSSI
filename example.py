import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import Sequential, Model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
np.set_printoptions(threshold = np.inf) 

model = tf.keras.models.load_model("./model/VSPD-test3.h5")
model.summary()
for i in range(20,30,1):
    print(i)
    print(model.layers[i].output_shape)
layer_model = Model(inputs=model.input, outputs=model.get_layer('activation_9').output)
sample_high = np.load("./data/originalTerrain-41-4.npy")
# sample_high = np.load("./data/originalTerrain-41.npy")


sample_low = sample_high[:, 0:41:4, 0:41:4, :]

sample_inter = tf.image.resize(
    sample_low, (41, 41), method=tf.image.ResizeMethod.BICUBIC)

# mask = np.zeros((40000,41,41,1))
# mask[:,0:41:4,0:41:4,0] = 1

mask = np.random.binomial(1,0.3,(sample_high.shape))
mask=mask.astype(np.float32)

# sample_high = train_images
sample_low = np.multiply(mask, sample_high)

def mse(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.power(a-b,2)
    # print(c)
    results = np.sum(c.flatten()) / (41*41)
    return results

sample_res = sample_high-sample_inter

result = model.predict([sample_inter[0:40000,:,:,:],mask[0:40000,:,:,:]])
a = np.power(result - sample_high[0:40000,:,:,:], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*40000))
print(b)

a = np.power(sample_inter[0:40000,:,:,:] - sample_high[0:40000,:,:,:],2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*40000))
print(b)


for i in range(1000):
    j = np.random.randint(0,40000)
    # lowf = layer_model.predict([sample_inter[j:j+1,:,:,:],mask[j:j+1,:,:,:]])
    print(mask[j,:,:,0])
    plt.subplot(231)
    plt.title(np.sqrt(mse(sample_high[j,:,:,0], sample_inter[j,:,:,0])).round(3))
    plt.imshow(sample_inter[j,:,:,0])
    plt.subplot(232)
    plt.title(np.sqrt(mse(sample_high[j,:,:,0], result[j,:,:,0])).round(3))
    plt.imshow(result[j,:,:,0])
    plt.subplot(233)
    plt.imshow(sample_high[j,:,:,0])
    plt.subplot(234)
    plt.imshow(sample_high[j,0:41:4,0:41:4,0])
    plt.subplot(235)
    plt.imshow(result[j,:,:,0] - sample_inter[j,:,:,0])
    plt.subplot(236)
    plt.imshow(sample_high[j,:,:,0] - sample_inter[j,:,:,0])
    print(j)
    plt.show()