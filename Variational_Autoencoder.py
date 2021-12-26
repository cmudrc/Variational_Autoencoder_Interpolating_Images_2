import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
from tensorflow.keras.datasets import mnist
from tensorflow.python.framework.ops import disable_eager_execution
import random
warnings.filterwarnings('ignore')
disable_eager_execution()

"""
This code is used to download the MNIST data set, then a few sample values from the set are chosen to test
"""

(trainX, trainy), (testX, testy) = mnist.load_data()
print('Training data shapes: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Testing data shapes: X=%s, y=%s' % (testX.shape, testy.shape))
print('Training data type:' + str(type(trainy)))
print(testy)
print(testX)

for j in range(5):  # shows  5 random images to the users to view samples of the dataset
    i = np.random.randint(0, 10000)
    plt.subplot(550 + 1 + j)
    plt.imshow(trainX[i], cmap='gray')
    plt.title(trainy[i])
plt.show()


"""
The images imported have an intensity ranging from 0 to 255, these values are normalized, making them between 0 and 1
This data is then re-shaped
"""
train_data = trainX.astype('float32') / 255
test_data = testX.astype('float32') / 255
train_data = np.reshape(train_data, (60000, 28, 28, 1))
test_data = np.reshape(test_data, (10000, 28, 28, 1))

print(train_data.shape, test_data.shape)
########################################################################################################################
# The framework for the encoder is established
input_data = tensorflow.keras.layers.Input(shape=(28, 28, 1))
print("Encoder Input Data" + str(input_data.shape))
encoder = tensorflow.keras.layers.Conv2D(64, (5, 5), activation='relu')(input_data)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)

encoder = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu')(encoder)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)

encoder = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu')(encoder)
encoder = tensorflow.keras.layers.MaxPooling2D((2, 2))(encoder)

encoder = tensorflow.keras.layers.Flatten()(encoder)
encoder = tensorflow.keras.layers.Dense(16)(encoder)


def sample_latent_features(distribution):
    distribution_mean = distribution[0]
    distribution_variance = distribution[1]
    batch_size = tensorflow.shape(distribution_variance)[0]
    random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
    return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random


distribution_mean = tensorflow.keras.layers.Dense(2, name='mean')(encoder)
distribution_variance = tensorflow.keras.layers.Dense(2, name='log_variance')(encoder)
latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])

print("Latent Encoding" + str(latent_encoding.shape))
print("Encoder Input Data" + str(input_data.shape))
encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
encoder_model.summary()


########################################################################################################################
# The framework for the decoder is established
decoder_input = tensorflow.keras.layers.Input(shape=2)
print("Decoder Input Data" + str(decoder_input.shape))
decoder = tensorflow.keras.layers.Dense(64)(decoder_input)
decoder = tensorflow.keras.layers.Reshape((1, 1, 64))(decoder)
decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)

decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2, 2))(decoder)

decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2, 2))(decoder)

decoder_output = tensorflow.keras.layers.Conv2DTranspose(1, (5, 5), activation='relu')(decoder)


decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
decoder_model.summary()
########################################################################################################################
# The framework for the autoencoder is established
encoded = encoder_model(input_data)
decoded = decoder_model(encoded)
autoencoder = tensorflow.keras.models.Model(input_data, decoded)
autoencoder.summary()

########################################################################################################################
# Autoencoder is trained using the training data

def get_loss(distribution_mean, distribution_variance):
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch * 28 * 28

    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tensorflow.square(distribution_mean) - tensorflow.exp(
            distribution_variance)
        kl_loss_batch = tensorflow.reduce_mean(kl_loss)
        return kl_loss_batch * (-0.5)

    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch

    return total_loss


autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
autoencoder.fit(train_data, train_data, epochs=1, batch_size=64, validation_data=(test_data, test_data))


########################################################################################################################
# Saving the Encoder Model
# Saving model architecture to JSON file and Weights Separately
model_json = encoder_model.to_json()

# Saving to local directory
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Saving weights of model
encoder_model.save_weights('model_tf', save_format='tf')  # tf format

########################################################################################################################
# Model to Generate New Images
#encoder_model.save('encoder_model')
decoder_model.save('decoder_model')
########################################################################################################################
# Used to print out an interpolation between two values
# Establish the Framework of the Interpolation
number_internal = 13  # the number of interpolations that the model will find between two points
num_interp = number_internal + 2  # the number of images to be pictured
figure = np.zeros((28, 28 * num_interp))  # The matrix that will hold the pixel values of the images
xy1 = [0, -3]  # Latent Point 1
xy2 = [0, 3]  # Latent Point 2
x_values = np.linspace(xy1[0], xy2[0], num_interp)  # Interpolates the values of x between the two latent points

print("Linspace X")
print(x_values)
y_values = np.linspace(xy1[1], xy2[1], num_interp) # Interpolates the values of y between the two latent points
print("Linspace Y")
print(y_values)

########################################################################################################################
# Interpolate the Images and Print out to User
print('Latent Points')
for iy, y in enumerate(y_values):  # enumerate creates a list of tuples [(0,-3), ... (29, 3)]
    # iy will print out the column numbers
    latent_point = np.array([[x_values[iy], y]])  # this creates the points that will establish the framework of the matrix
    # for the images to populate around, in this case makes:
    # [ [0,-3]  ... [0,3] ]
    print(latent_point)
    generated_image = decoder_model.predict(latent_point)[0]  # generates an interpolated image based on the latent point
    figure[0: 28,  iy * 28:(iy + 1) * 28, ] = generated_image[:, :, -1]  # Inserts the Pixel Value for Each Image

plt.figure()
plt.imshow(figure,cmap='gray')
plt.show()
