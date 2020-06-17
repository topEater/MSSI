from libs.MSSI import *
import tensorflow as tf

model = MSSI_Model()

adam = Adam(lr=0.000001)
sgd = SGD(lr=1e-6, momentum=0.5, decay=1e-4, nesterov=False)
model.compile(adam, loss=tf.keras.losses.mean_absolute_error)
# model.compile(adam, loss = smooth_l1_loss)
model.load_weights("./model/MSSI-random-weights11.h5")


data = np.load("./data/terrain-41-train.npy")

y_train = np.concatenate(
    (data[0:80000, :, :, :], data[0:80000, :, :, :], data[0:80000, :, :, :]))
mask_train = np.zeros(y_train.shape)
# mask_train[0:   80000, 0:41:4, 0:41:4, 0] = 1
# mask_train[80000:  160000, 0:41:3, 0:41:3, 0] = 1
# mask_train[160000: 240000, 0:41:2, 0:41:2, 0] = 1

# mask_train = np.ones(y_train.shape)
# mask_train[0    :  80000, 5 :-5 , 5:-5  , 0] = 0
# mask_train[80000:  160000, 10:-10, 10:-10, 0] = 0
# mask_train[160000: 240000, 15:-15, 15:-15, 0] = 0
# print(mask_train[0,:,:,0])

mask_train[0:   80000, 0:41, 0:41, :] = np.random.binomial(1,0.25,(80000,41,41,1))
mask_train[80000:  160000, 0:41, 0:41, :] = np.random.binomial(1,0.50,(80000,41,41,1))
mask_train[160000: 240000, 0:41, 0:41, :] = np.random.binomial(1,0.75,(80000,41,41,1))

x_train = np.multiply(mask_train, y_train)

y_val = np.concatenate(
    (data[80000:90000, :, :, :], data[80000:90000, :, :, :], data[80000:90000, :, :, :]))
mask_val = np.zeros(y_val.shape)
# mask_val[0:   10000, 0:41:4, 0:41:4, 0] = 1
# mask_val[10000:  20000, 0:41:3, 0:41:3, 0] = 1
# mask_val[20000:  30000, 0:41:2, 0:41:2, 0] = 1

# mask_val = np.ones(y_val.shape)
# mask_val[0    :  10000, 5 :-5 , 5:-5  , 0] = 0
# mask_val[10000:  20000, 10:-10, 10:-10, 0] = 0
# mask_val[20000:  30000, 15:-15, 15:-15, 0] = 0

mask_val[0:   10000, 0:41, 0:41, :] = np.random.binomial(1,0.25,(10000,41,41,1))
mask_val[10000:  20000, 0:41, 0:41, :] = np.random.binomial(1,0.50,(10000,41,41,1))
mask_val[20000:  30000, 0:41, 0:41, :] = np.random.binomial(1,0.75,(10000,41,41,1))

x_val = np.multiply(mask_val, y_val)
del data

checkpoint = ModelCheckpoint("./model/MSSI_random_val_loss_best_weights.h5",
    monitor='val_loss', save_weights_only=True, verbose=1,save_best_only=True, period=1)
# loss,accuracy = model.evaluate([x_val, mask_val], y_val)
# print(loss)

history = model.fit(x=[x_train, mask_train], y=y_train, validation_data=(
    [x_val, mask_val], y_val), shuffle=True, epochs=60, callbacks = [checkpoint])

model.save_weights("./model/MSSI-random-weights12.h5")
with open("./model/MSSI-random-weights12", 'wb') as file_pi:
    pickle.dump(history.history, file_pi)