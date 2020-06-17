import numpy as np
import matplotlib.pyplot as plt
import _thread
import sys
import random

dataPath = r"D:/MyPySpace/Terrian matching/data/SRTM15/SRTM15"

def readDataToArray(start, shape):
    """
    start : 数据左上角的起始点
    shape : 数据的维度形状
    """
    lat = start[0]
    lon = start[1]
    height = shape[0]
    width = shape[1]
    array = np.zeros((height, width),  dtype=np.float)
    for i in range(height):
        # print(type(lat))
        data = np.load(dataPath+"/"+str(lat+i)+".npy")
        array[i] = data[lon:lon+width]
    return array

def readDataThread(size, name):
    lat = np.random.randint(3600,39600,size=[size,1])
    lon = np.random.randint(0,86359,size=[size,1])
    highSample = np.zeros((size, 41, 41, 1))
    for i in range(0, size, 1):
        if i % 1000 == 0:
            print(i)
        highSample[i, :, :, 0] = readDataToArray((lat[i,0], lon[i,0]), (41,41))
    np.save("./data/originalTerrain-41-"+str(name)+".npy", highSample)
    np.save("./data/originalTerrain-lat-"+str(name)+".npy", lat)
    np.save("./data/originalTerrain-lon-"+str(name)+".npy", lon)
    print("finished")

def readDataWithIndex(name):
    latIndex = np.load("./data/latIndex-"+ str(name) + ".npy")
    lonIndex = np.load("./data/lonIndex-"+ str(name) + ".npy")
    size = latIndex.shape[0]
    print(size)
    terrian = np.zeros((size, 41, 41, 1))
    print(size)
    for i in range(size):
        if i % 1000 == 0:
            print(i)
        terrian[i,:,:,0] = readDataToArray((latIndex[i], lonIndex[i]), (41, 41))
    np.save("./data/terrain-41-"+str(name)+".npy", terrian)

def creatIndex(size = 100000):
    max = 877 * 2106
    index = random.sample(range(max), size)
    index = np.array(index)
    latIndex = np.floor_divide(index, 2106) * 41 + 3600
    lonIndex = np.mod(index,2106) * 41
    np.save("./data/lonIndex.npy", lonIndex)
    np.save("./data/latIndex.npy", latIndex)

def mergeData():
    data1 = np.load('./data/terrain-41-1.npy')
    data2 = np.load('./data/terrain-41-2.npy')
    data3 = np.load('./data/terrain-41-3.npy')
    data4 = np.load('./data/terrain-41-4.npy')

    data = np.concatenate((data1, data2, data3, data4))

    np.save("./data/terrain-41-train.npy", data[0:90000,:,:,:])
    np.save("./data/terrain-41-test.npy", data[90000:100000,:,:,:])


# args = sys.argv
# print(args)
# readDataWithIndex(int(args[1]))

mergeData()

# data = np.load("./data/originalTerrain-41-1.npy")
# for i in range(0,40000):
#     j = np.random.randint(0,40000)
#     print(j)
#     plt.imshow(data[j,:,:,0])
#     plt.show()

# latIndex = np.load("./data/latIndex.npy")
# lonIndex = np.load("./data/lonIndex.npy")
# plt.scatter(lonIndex,latIndex)
# plt.show()
# np.save("./data/lonIndex-1.npy", lonIndex[0:25000])
# np.save("./data/latIndex-1.npy", latIndex[0:25000])
# np.save("./data/lonIndex-2.npy", lonIndex[25000:50000])
# np.save("./data/latIndex-2.npy", latIndex[25000:50000])
# np.save("./data/lonIndex-3.npy", lonIndex[50000:75000])
# np.save("./data/latIndex-3.npy", latIndex[50000:75000])
# np.save("./data/lonIndex-4.npy", lonIndex[75000:100000])
# np.save("./data/latIndex-4.npy", latIndex[75000:100000])



creatIndex()
