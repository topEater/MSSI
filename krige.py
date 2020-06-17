import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

from libs.krigelib import *

sample_high = np.load("./data/terrain-41-test.npy")

# # mask = np.zeros(sample_high.shape)
# # mask[:,0:41:3,0:41:3,0] = 1

# mask = np.load("./data/mask-25.npy")

# # mask = np.ones(sample_high.shape)
# # mask[0:  10000, 5:-5, 5:-5, 0] = 0

# # results = krigeInterpolation(sample_high, mask, "./data/voidresultsWithKrige31.npy")
# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige25.npy")


# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)


# # mask = np.ones(sample_high.shape)
# # mask[0:  10000, 10:-10, 10:-10, 0] = 0

# mask = np.load("./data/mask-50.npy")

# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige50.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# # mask = np.ones(sample_high.shape)
# # mask[0:  10000, 15:-15, 15:-15, 0] = 0

# mask = np.load("./data/mask-75.npy")

# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige75.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-30.npy")
# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige30.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-40.npy")
# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige40.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-60.npy")
# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige60.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

# mask = np.load("./data/mask-70.npy")
# results = krigeInterpolation(sample_high, mask, "./data/randomresultsWithKrige70.npy")

# a = np.power(results - sample_high[:,:,:,:], 2)
# print(a.shape)
# b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
# print(b)

mask = np.ones(sample_high.shape)
mask[0:  10000, 8:-8, 8:-8, 0] = 0
results = krigeInterpolation(sample_high, mask, "./data/voidresultsWithKrige25.npy")

a = np.power(results - sample_high[:,:,:,:], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
print(b)

mask = np.ones(sample_high.shape)
mask[0:  10000, 13:-13, 13:-13, 0] = 0
results = krigeInterpolation(sample_high, mask, "./data/voidresultsWithKrige15.npy")

a = np.power(results - sample_high[:,:,:,:], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
print(b)