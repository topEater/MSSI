import numpy as np
from libs.bilinearlib import *

sample_high = np.load("./data/terrain-41-test.npy")

# mask = np.zeros(sample_high.shape)
# mask[:,0:41:3,0:41:3,0] = 1

# mask = np.random.binomial(1,0.50,sample_high.shape)
# mask = mask.astype(np.float32)
# mask[:,0,0,:] = 1
# mask[:,0,-1,:] = 1
# mask[:,-1,0,:] = 1
# mask[:,-1,-1,:] = 1
mask = np.ones(sample_high.shape)
# mask[0:  10000, 5:-5, 5:-5, 0] = 0
mask[0:  10000, 10:-10, 10:-10, 0] = 0

print(mask[0,10,:,0])

np.save("./data/void21-mask.npy",mask)

results = BilinearInterpolation(sample_high, mask, "./data/voidresultsWithBilinear21.npy")

# results = np.load("./data/voidresultsWithBilinear31.npy")

a = np.power(results - sample_high[:,:,:,:], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
print(b)

a = np.abs(results - sample_high[:,:,:,:])
print(a.shape)
b = np.sum(a.flatten() / (41*41*10000))
print(b)