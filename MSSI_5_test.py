from libs.MSSI_5 import *

# model.load_weights("./model/new-pconv2-L2-weights.h5")

model, low_model, high_model = MSSI_Model_parts()
# low_model.load_weights("./model/MSSI_5-low-weights1.h5")
model.load_weights("./model/MSSI_5_void_val_loss_best_weights.h5")

# model.summary()
# high_model.summary()
low_model.summary()

layer_model_high = Model(inputs=high_model.input,
                         outputs=high_model.get_layer('conv2d_27').output)

sample_high = np.load("./data/terrain-41-test.npy")

# mask = np.zeros(sample_high.shape)
# mask[:,0:41:2,0:41:2,0] = 1

# mask = np.load("./data/random50-mask.npy")

# mask = np.zeros(sample_high.shape)
# mask[:,0:41:2,0:41:2,0] = 1

# mask = np.ones(sample_high.shape)
# mask[:,10:14,20:14,0] = 0
# mask = np.load("./data/mask-50.npy")

mask = np.ones(sample_high.shape)
# mask[0:  10000, 5:-5, 5:-5, 0] = 0
# mask[0:  10000, 8:-8, 8:-8, 0] = 0
# mask[0:  10000, 10:-10, 10:-10, 0] = 0
mask[0:  10000, 13:-13, 13:-13, 0] = 0
# mask[0:  10000, 15:-15, 15:-15, 0] = 0


sample_low = np.multiply(mask, sample_high)

count = 10000

result = model.predict([sample_low[0:count, :, :, :], mask[0:count, :, :, :]])
np.save("./data/voidresultsWithMSSI_5_15.npy", result)
# result = np.multiply(result, 1 - mask[0:count, :, :, :]) + sample_low[0:count, :, :, :]
a = np.power(result - sample_high[0:count, :, :, :], 2)
print(a.shape)
b = np.sqrt(np.sum(a.flatten()) / (41*41*count))
print(b)

a = np.abs(result - sample_high[0:count, :, :, :])
print(a.shape)
b = np.sum(a.flatten() / (41*41*count))
print(b)

# sample_inter = np.load("./data/randomresultsWithBilinear50.npy")

for i in range(count):
    j = np.random.randint(0, count)
    k = np.random.randint(0, 64)
    # result = model.predict([sample_low[j:j+1,:,:,:], mask[j:j+1,:,:,:]])[0,:,:,0]
    # tempResult = result[j,:,:,0]
    low = low_model.predict([sample_low[j:j+1, :, :, :], mask[j:j+1, :, :, :]])
    high = layer_model_high.predict(low)
    plt.subplot(151)
    plt.imshow(sample_high[j, :, :, 0])
    plt.subplot(152)
    plt.imshow(result[j, :, :, 0])
    # plt.colorbar()
    plt.subplot(153)
    plt.imshow(low[0, :, :, k])
    # plt.colorbar()
    plt.subplot(154)
    plt.imshow(high[0, :, :, k])
    # plt.colorbar()
    plt.subplot(155)
    plt.imshow(low[0, :, :, k]+high[0, :, :, k])
    # plt.colorbar()
    print(j)
    print(k)
    plt.show()
