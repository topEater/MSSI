import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


history = ["MSSI-random-weights1", "MSSI-random-weights2", "MSSI-random-weights3", "MSSI-random-weights4", "MSSI-random-weights5", 
           "MSSI-random-weights6", "MSSI-random-weights7", "MSSI-random-weights8", "MSSI-random-weights9", "MSSI-random-weights10", "MSSI-random-weights11"]
loss = []
val_loss = []
for i in range(11):
    temp = np.load("./model/"+history[i])["loss"]
    print(len(temp))
    loss = loss+temp
    temp = np.load("./model/"+history[i])["val_loss"]
    val_loss = val_loss+temp
temp = np.zeros((len(loss), 2))
temp[0:30,0] = np.arange(1, 31)
temp[30:,0] = np.arange(1, 361)
# temp[210:,0] = np.arange(1, 181)
temp[:,1] = loss
loss = temp
print(temp)

temp = np.zeros((len(loss), 2))
temp[0:30,0] = np.arange(1, 31)
temp[30:,0] = np.arange(1, 361)
# temp[210:,0] = np.arange(1, 181)
temp[:,1] = val_loss
val_loss = temp
print(temp)

loss = pd.DataFrame(loss, columns=["epochs",'loss'])
val_loss = pd.DataFrame(val_loss, columns=["epochs",'loss'])
# data = sns.lineplot

plt.figure(figsize=(4.5, 2))
ax = sns.lineplot(x="epochs", y="loss", data=loss[0:30])
ax = sns.lineplot(x="epochs", y="loss", data=val_loss[0:30])
plt.legend(['loss', 'val_loss'])
plt.title("SGD")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()
plt.figure(figsize=(4.5, 2))
ax = sns.lineplot(x="epochs", y="loss", data=loss[30:])
ax = sns.lineplot(x="epochs", y="loss", data=val_loss[30:])
plt.legend(['loss', 'val_loss'])
plt.title("Adam")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()
# plt.figure(figsize=(9, 2))
# ax = sns.lineplot(x="epochs", y="loss", data=loss[210:])
# ax = sns.lineplot(x="epochs", y="loss", data=val_loss[210:])
# # plt.ylim(2, 2.5)
# plt.legend(['loss', 'val_loss'])
# plt.title("Adam lr=1e-6")
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
print(np.min(np.array(val_loss)))
print(np.min(np.array(loss)))
print(loss)