import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.mlab import griddata

import scipy



def CubicSplineInterpolation(data = None, mask = None,filePath= ""):
    # griddata()
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
        x = np.linspace(0, 40, 41)
        y = np.linspace(0, 40, 41)
        # print(temp[:,2].max())
        # x, y = np.meshgrid(x, y)#20*20的网格数据
        # z = interpolate.griddata(temp[:,0:2], temp[:,2], (x, y), method='cubic')
        # func = interpolate.interp2d(temp[:,0], temp[:,1], temp[:,2], kind='cubic')
        w = np.ones((len(temp[:,0]),1))
        zum = np.sum(temp[:,2]**2)
        smooth = 0.01
        print(zum)
        func = interpolate.SmoothBivariateSpline(temp[:,1], temp[:,0], temp[:,2], w=w, s = 0)
        z = func(x, y)
        results[i,:,:,0] = z
        # plt.subplot(121)
        # print(z.shape)
        # plt.imshow(z)
        # plt.subplot(122)
        # plt.imshow(data[i,:,:,0])
        # plt.show()
    np.save(filePath, results)
    return results