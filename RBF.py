import numpy as np
from libs.RBFlib import *

sample_high = np.load("./data/terrain-41-test.npy")

# # mask = np.ones(sample_high.shape)
# # mask[0:  10000, 5:-5, 5:-5, 0] = 0
# mask = np.load("./data/mask-25.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF25.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.ones(sample_high.shape)
# mask[0:  10000, 10:-10, 10:-10, 0] = 0

# # mask = np.ones(sample_high.shape)
# # mask[0:  10000, 10:-10, 10:-10, 0] = 0
# mask = np.load("./data/mask-50.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF50.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# # mask = np.ones(sample_high.shape)
# # mask[0:  10000, 15:-15, 15:-15, 0] = 0
# mask = np.load("./data/mask-75.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF75.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)


# mask = np.load("./data/mask-30.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF30.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-40.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF40.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-60.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF60.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-70.npy")
# results = RBFInterpolation(sample_high, mask, "./data/randomresultsWithRBF70.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

mask = np.ones(sample_high.shape)
mask[0:  10000, 8:-8, 8:-8, 0] = 0
results = RBFInterpolation(sample_high, mask, "./data/voidresultsWithRBF25.npy")

a = np.power(results - sample_high[:,:,:,:], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
print(b)

mask = np.ones(sample_high.shape)
mask[0:  10000, 13:-13, 13:-13, 0] = 0
results = RBFInterpolation(sample_high, mask, "./data/voidresultsWithRBF15.npy")

a = np.power(results - sample_high[:,:,:,:], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
print(b)
