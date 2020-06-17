import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm

def RBFInterpolation(data = None, mask = None, filePath= ""):
    results = np.zeros(data.shape)
    for i in range(data.shape[0]):
        if i % 100 == 0:
            print(i)
        temp = np.zeros((int(np.sum(mask[i,:,:,0].flatten())),3))
        index = 0
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if mask[i,j,k,0] == 1:
                    temp[index,0] = k
                    temp[index,1] = j
                    temp[index,2] = data[i,j,k,0]
                    index = index + 1
        ti = np.linspace(0, 41, 41)
        x, y = np.meshgrid(ti, ti)
        rbf = Rbf(temp[:,0], temp[:,1], temp[:,2], epsilon=2)
        z = rbf(x, y)
        results[i,:,:,0] = z
        # plt.subplot(121)
        # plt.imshow(z)
        # plt.subplot(122)
        # plt.imshow(data[i,:,:,0])
        # plt.show()
    np.save(filePath, results)
    return results