import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

def krigeInterpolation(data = None, mask = None, filePath= ""):
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
        gridx = np.arange(0.0, 41.0, 1.0)
        gridy = np.arange(0.0, 41.0, 1.0)
        try:
            OK = OrdinaryKriging(temp[:, 0], temp[:, 1], temp[:, 2], variogram_model="spherical",
                        verbose=False, enable_plotting=False)
            z,ss = OK.execute('grid', gridx, gridy)
        except ValueError:
            print("Error: All the values ​​are the same the distance is zero, then it fails to compute L1 norm.")
            print(temp)
            z = np.ones((41,41)) * temp[0,2]
            print(z)
        results[i,:,:,0] = z
        # plt.subplot(121)
        # plt.imshow(z)
        # plt.subplot(122)
        # plt.imshow(data[i,:,:,0])
        # plt.show()
    np.save(filePath, results)
    return results

