import numpy as np
import tensorflow as tf
import os

sample_high = np.load("./data/terrain-41-test.npy")
fileList = os.listdir("./data")
for file in fileList:
    # if file.startswith("uniform"):
    # if file.startswith("random"):
    if file.startswith("void"):
        results = np.load("./data/"+file)
        a = np.power(results - sample_high[:, :, :, :], 2)
        b = np.sqrt(np.sum(a.flatten()) / (41*41*10000))
        print(file+" RMSE: "+str(b))
        a = np.abs(results - sample_high[:, :, :, :])
        b = np.sum(a.flatten() / (41*41*10000))
        print(file+" MAE: "+str(b))
