import numpy as np
import matplotlib.pyplot as plt

sample_high = np.load("./data/terrain-41-test.npy")
RBFX4 = np.load("./data/voidresultsWithRBF11.npy")
KrigeX4 = np.load("./data/voidresultsWithKrige11.npy")
MSSIX4 = np.load("./data/voidresultsWithMSSI_5_11.npy")
MSSIPlus = np.load("./data/voidresultsWithMSSI_5-self11.npy")

# mask = np.zeros(sample_high.shape)
# mask[:,0:41:4,0:41:4,0] = 1
mask = np.ones(sample_high.shape)
# mask[0:10000, 5:-5 , 5:-5  , 0] = 0
# mask[0:10000, 10:-10, 10:-10, 0] = 0
mask[0:  10000, 15:-15, 15:-15, 0] = 0
sample_low = np.multiply(mask, sample_high)

for i in range(10000):
    j = np.random.randint(0,10000)
    plt.subplot(161)
    plt.imshow(sample_low[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(162)
    plt.imshow(RBFX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(163)
    plt.imshow(KrigeX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(164)
    plt.imshow(MSSIX4[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(165)
    plt.imshow(MSSIPlus[j,:,:,0],cmap='gray')
    plt.axis('off') 
    plt.subplot(166)
    plt.imshow(sample_high[j,:,:,0], cmap='gray')
    plt.axis('off') 
    plt.show()


