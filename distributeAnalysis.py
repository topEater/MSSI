import numpy as np
import matplotlib.pyplot as plt

dataPath = r"D:/MyPySpace/Terrian matching/data/SRTM15/SRTM15"

array = np.zeros((432, 864),  dtype=np.float)
for i in range(432):
    data = np.load(dataPath+"/"+str(0+i*100)+".npy")
    array[i] = data[0:86400:100]
plt.imshow(array[432:0:-1,:])
plt.show()
print(array.flatten().max())
print(array.flatten().min())
plt.hist(array.flatten(), 1000, density=True, facecolor='g', alpha=0.75)
plt.show()