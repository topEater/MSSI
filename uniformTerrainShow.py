import numpy as np
import matplotlib.pyplot as plt

sample_high = np.load("./data/terrain-41-test.npy")
RBFX4 = np.load("./data/uniformresultsWithRBFX4.npy")
KrigeX4 = np.load("./data/uniformresultsWithKrigeX4.npy")
BicubicX4 = np.load("./data/uniformresultsWithBicubicX4.npy")
MSSIX4 = np.load("./data/uniformresultsWithMSSIX4.npy")
MSSIPlus = np.load("./data/uniformresultsWithMSSI-selfX4.npy")

mask = np.zeros(sample_high.shape)
mask[:,0:41:4,0:41:4,0] = 1
sample_low = np.multiply(mask, sample_high)

for i in range(10000):
    j = np.random.randint(0,10000)
    plt.subplot(171)
    plt.imshow(sample_low[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(172)
    plt.imshow(RBFX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(173)
    plt.imshow(KrigeX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(174)
    plt.imshow(BicubicX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(175)
    plt.imshow(MSSIX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(176)
    plt.imshow(MSSIPlus[j,:,:,0],cmap='gray')
    plt.axis('off') 
    plt.subplot(177)
    plt.imshow(sample_high[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.show()


