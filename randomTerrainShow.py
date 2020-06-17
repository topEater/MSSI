import numpy as np
import matplotlib.pyplot as plt

sample_high = np.load("./data/terrain-41-test.npy")
RBF = np.load("./data/randomresultsWithRBF75.npy")
Krige = np.load("./data/randomresultsWithKrige75.npy")
MSSI = np.load("./data/randomresultsWithMSSI75.npy")
MSSIPlus = np.load("./data/randomresultsWithMSSI-self75.npy")

mask = np.load("./data/mask-75.npy")
sample_low = np.multiply(mask, sample_high)

for i in range(10000):
    j = np.random.randint(0,10000)
    plt.subplot(161)
    vmax = sample_high[j,:,:,0].max()
    vmin = sample_high[j,:,:,0].min()
    plt.imshow(sample_low[j,:,:,0], cmap='gray',vmax=vmax, vmin=vmin)
    plt.axis('off') 
    plt.subplot(162)
    plt.imshow(RBF[j,:,:,0], cmap='gray',vmax=vmax, vmin=vmin)
    plt.axis('off') 
    plt.subplot(163)
    plt.imshow(Krige[j,:,:,0], cmap='gray',vmax=vmax, vmin=vmin)
    plt.axis('off') 
    plt.subplot(164)
    plt.imshow(MSSI[j,:,:,0], cmap='gray',vmax=vmax, vmin=vmin)
    plt.axis('off') 
    plt.subplot(165)
    plt.imshow(MSSIPlus[j,:,:,0],cmap='gray',vmax=vmax, vmin=vmin)
    plt.axis('off') 
    plt.subplot(166)
    plt.imshow(sample_high[j,:,:,0], cmap='gray',vmax=vmax, vmin=vmin)
    plt.axis('off') 
    plt.show()


