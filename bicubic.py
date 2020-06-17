import numpy as np
import tensorflow as tf

sample_high = np.load("./data/terrain-41-test.npy")

sample_low = sample_high[0:10000, 0:41:4, 0:41:4, :]

sample_inter = tf.image.resize(
    sample_low, (41, 41), method=tf.image.ResizeMethod.BICUBIC)

np.save("./data/uniformresultsWithBicubicX4.npy", sample_inter)

sample_low = sample_high[0:10000, 0:41:3, 0:41:3, :]

sample_inter = tf.image.resize(
    sample_low, (40, 40), method=tf.image.ResizeMethod.BICUBIC)

np.save("./data/uniformresultsWithBicubicX3.npy", sample_inter)

sample_low = sample_high[0:10000, 0:41:4, 0:41:4, :]

sample_inter = tf.image.resize(
    sample_low, (41, 41), method=tf.image.ResizeMethod.BICUBIC)

np.save("./data/uniformresultsWithBicubicX4.npy", sample_inter)
