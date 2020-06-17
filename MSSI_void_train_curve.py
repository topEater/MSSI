import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


history = ["MSSI_5-void-weights1", "MSSI_5-void-weights2", "MSSI_5-void-weights3",
           "MSSI_5-void-weights4", "MSSI_5-void-weights5"]
loss = []
val_loss = []
for i in range(5):
    temp = np.load("./model/"+history[i])["loss"]
    print(len(temp))
    loss = loss+temp
    temp = np.load("./model/"+history[i])["val_loss"]
    val_loss = val_loss+temp
print(np.array(val_loss).min())
# plt.subplot(121)
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.plot(np.arange(0,30), loss[0:30])
# # plt.scatter(np.arange(0,len(loss)), loss)
# plt.plot(np.arange(0,30), val_loss[0:30])
# plt.subplot(111)
# plt.plot(np.arange(30,210), loss[30:210])
# # plt.scatter(np.arange(0,len(loss)), loss)
# plt.plot(np.arange(30,210), val_loss[30:210])
# plt.show()
temp = np.zeros((len(loss), 2))
temp[:,0] = np.arange(1, len(loss)+1)
temp[:,1] = loss
loss = temp
print(temp)

temp = np.zeros((len(loss), 2))
temp[:,0] = np.arange(1, len(loss)+1)
temp[:,1] = val_loss
val_loss = temp
print(temp)
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.plot(np.arange(0, 30), loss[0:30])
# # plt.scatter(np.arange(0,len(loss)), loss)
# plt.plot(np.arange(0, 30), val_loss[0:30])
# plt.show()
# plt.plot(np.arange(0, 330), loss[30:360])
# # plt.scatter(np.arange(0,len(loss)), loss)
# plt.plot(np.arange(0, 330), val_loss[30:360])
# plt.show()
loss = pd.DataFrame(loss, columns=["epochs",'loss'])
val_loss = pd.DataFrame(val_loss, columns=["epochs",'loss'])
# data = sns.lineplot

plt.figure(figsize=(9, 2))
ax = sns.lineplot(x="epochs", y="loss", data=loss)
ax = sns.lineplot(x="epochs", y="loss", data=val_loss)
plt.legend(['loss', 'val_loss'])
plt.title("Adam")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

print(loss)
