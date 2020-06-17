import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style="darkgrid") #这是seaborn默认的风格


history = ["MSSI-weights1", "MSSI-weights2", "MSSI-weights3", "MSSI-weights4",
           "MSSI-weights5", "MSSI-weights6", "MSSI-weights7", "MSSI-weights8"]
loss = []
val_loss = []
for i in range(8):
    temp = np.load("./model/"+history[i])["loss"]
    print(len(temp))
    loss = loss+temp
    temp = np.load("./model/"+history[i])["val_loss"]
    val_loss = val_loss+temp

temp = np.zeros((len(loss), 2))
temp[0:30,0] = np.arange(1, 31)
temp[30:60,0] = np.arange(1, 31)
temp[60:,0] = np.arange(1, len(loss)+1-60)
temp[:,1] = loss
loss = temp
print(temp)

temp = np.zeros((len(loss), 2))
temp[0:30,0] = np.arange(1, 31)
temp[30:60,0] = np.arange(1, 31)
temp[60:,0] = np.arange(1, len(loss)+1-60)
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

plt.figure(figsize=(4.5, 2))
ax = sns.lineplot(x="epochs", y="loss", data=loss[0:30])
ax = sns.lineplot(x="epochs", y="loss", data=val_loss[0:30])
plt.legend(['loss', 'val_loss'])
plt.title("SGD")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
plt.figure(figsize=(4.5, 2))
ax = sns.lineplot(x="epochs", y="loss", data=loss[30:60])
ax = sns.lineplot(x="epochs", y="loss", data=val_loss[30:60])
plt.legend(['loss', 'val_loss'])
plt.title("Adam lr=1e-5")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
plt.figure(figsize=(9, 2))
ax = sns.lineplot(x="epochs", y="loss", data=loss[60:])
ax = sns.lineplot(x="epochs", y="loss", data=val_loss[60:])
plt.legend(['loss', 'val_loss'])
plt.title("Adam lr=1e-6")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
print(np.min(np.array(val_loss)))
print(np.min(np.array(loss)))
print(loss)
