from libs.MSSI import *

model, low_model, high_model = MSSI_Model_parts()
model.load_weights("./model/MSSI_random_val_loss_best_weights.h5")

# model.summary()
# high_model.summary()
low_model.summary()

sample_high = np.load("./data/terrain-41-test.npy")

mask = np.load("./data/mask-25.npy")
# mask = np.ones(sample_high.shape)
# mask[0 : 10000, 5:-5 , 5:-5 , 0] = 0
mask = np.zeros(sample_high.shape)
mask[:,0:41:4,0:41:4,0] = 1
sample_low = np.multiply(mask, sample_high)

masks = []

masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_1').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_3').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_5').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_7').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_9').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_11').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_13').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_15').output))
masks.append(Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_17').output))
# mask_layer = Model(inputs=low_model.input, outputs=low_model.get_layer('lambda_1').output)


count = 10000

for i in range(count):
    j = np.random.randint(0,count)
    k = np.random.randint(0,64)
    result = model.predict([sample_low[i:i+1,:,:,:], mask[i:i+1,:,:,:]])[0,:,:,0]
    # tempResult = result[j,:,:,0]
    plt.subplot(191)
    plt.imshow(mask[i,:,:,0]== 0, cmap='gray', vmin=0, vmax=1)
    plt.axis("off")
    for j in range(9):
        plt.subplot(190+j+1)
        plt.imshow(masks[j].predict([sample_low[i:i+1,:,:,:], mask[i:i+1,:,:,:]])[0,:,:,0]== 0, cmap='gray',vmin=0, vmax=1)
        plt.axis("off")
    # temp = mask_layer.predict([sample_low[j:j+1,:,:,:], mask[j:j+1,:,:,:]])[0,:,:,0]
    # plt.imshow(temp)
    plt.show()
