from libs.MSSI import *

model, low_model, high_model = MSSI_Model_parts()
model.load_weights("./model/MSSI_random_val_loss_best_weights.h5")

# model.summary()
low_model.summary()
high_model.summary()

layer_model_high = Model(inputs=high_model.input, outputs=high_model.get_layer('activation_17').output)

sample_high = np.load("./data/terrain-41-test.npy")
mask = np.load("./data/mask-25.npy")

sample_low = np.multiply(mask, sample_high)

count = 10000

for i in range(count):
    j = np.random.randint(0,count)
    k = np.random.randint(0,64)
    # j = 5664
    # k = 6
    result = model.predict([sample_low[j:j+1,:,:,:], mask[j:j+1,:,:,:]])
    low = low_model.predict([sample_low[j:j+1,:,:,:], mask[j:j+1,:,:,:]])
    high = layer_model_high.predict(low)
    plt.subplot(151)
    plt.imshow(sample_high[j,4:-4,4:-4,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(152)
    plt.imshow(result[0,4:-4,4:-4,0], cmap='gray')
    plt.axis('off') 
    plt.subplot(153)
    plt.imshow(low[0,4:-4,4:-4,k], cmap='gray')
    plt.axis('off') 
    plt.subplot(154)
    plt.imshow(high[0,4:-4,4:-4,k], cmap='gray')
    plt.axis('off') 
    plt.subplot(155)
    plt.imshow(sample_low[j,4:-4,4:-4,0], cmap='gray')
    plt.axis('off') 
    print(j)
    print(k)
    plt.show()