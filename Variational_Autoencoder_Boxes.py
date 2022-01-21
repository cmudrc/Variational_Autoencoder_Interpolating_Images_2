import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
import random
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.python.framework.ops import disable_eager_execution
from box_data_set import make_boxes
from sklearn.model_selection import train_test_split
from tempfile import TemporaryFile
warnings.filterwarnings('ignore')
disable_eager_execution()
latent_dimensionality = 2
"""
This code is used to download the MNIST data set, then a few sample values from the set are chosen to test

(trainX, trainy), (testX, testy) = mnist.load_data()
print('Training data shapes: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Testing data shapes: X=%s, y=%s' % (testX.shape, testy.shape))
print('Training data type:' + str(type(trainy)))
print(testy)
print(testX)
"""
image_size = 28
number_of_densities = 5
min_density = 0
max_density = 1
box_data = make_boxes(image_size, number_of_densities, min_density, max_density)
box_data_train, box_data_test = train_test_split(box_data, test_size=0.15)  # Uses a similar percentage as MNIST Data

box_matrix_train, box_density_train, additional_pixels_train, box_shape_train = list(zip(*box_data_train))[0], list(zip(*box_data_train))[1], list(zip(*box_data_train))[2], list(zip(*box_data_train))[3]
box_matrix_test, box_density_test, additional_pixels_test, box_shape_test = list(zip(*box_data_test))[0], list(zip(*box_data_test))[1], list(zip(*box_data_test))[2], list(zip(*box_data_test))[3]

# train_test_split_data = TemporaryFile()
np.savez_compressed('train_test_split_data', box_matrix_train=box_matrix_train, box_density_train=box_density_train,
                    additional_pixels_train=additional_pixels_train, box_shape_train=box_shape_train,
                    box_matrix_test=box_matrix_test, box_density_test=box_density_test,
                    additional_pixels_test=additional_pixels_test, box_shape_test=box_shape_test)

trainX = box_matrix_train
testX = box_matrix_test
for j in range(5):  # shows  5 random images to the users to view samples of the dataset
    i = np.random.randint(0, len(box_matrix_train))
    plt.subplot(550 + 1 + j)
    plt.imshow(trainX[i], cmap='gray')
    plt.title(str(box_shape_train[i]) + "\nPixel Density: " + str(
            box_density_train[i]) + "\nAdditional Pixels: " + str(additional_pixels_train[i]))
plt.show()


"""
The images imported have an intensity ranging from 0 to 255, these values are normalized, making them between 0 and 1
This data is then re-shaped

train_data = trainX.astype('float32') / 255
test_data = testX.astype('float32') / 255
"""
train_data = np.reshape(trainX, (len(trainX), image_size, image_size, 1))
test_data = np.reshape(testX, (len(testX), image_size, image_size, 1))

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


distribution_mean = tensorflow.keras.layers.Dense(latent_dimensionality, name='mean')(encoder)
distribution_variance = tensorflow.keras.layers.Dense(latent_dimensionality, name='log_variance')(encoder)
latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])

print("Latent Encoding" + str(latent_encoding.shape))
print("Encoder Input Data" + str(input_data.shape))
encoder_model_boxes = tensorflow.keras.Model(input_data, latent_encoding)
encoder_model_boxes.summary()
########################################################################################################################
# The framework for the decoder is established
decoder_input = tensorflow.keras.layers.Input(shape=latent_dimensionality)
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
encoded = encoder_model_boxes(input_data)
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
autoencoder.fit(train_data, train_data, epochs=150, batch_size=32, validation_data=(test_data, test_data))
########################################################################################################################
# Saving the Encoder Model
# Saving model architecture to JSON file and Weights Separately
model_json = encoder_model_boxes.to_json()

# Saving to local directory
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Saving weights of model
encoder_model_boxes.save_weights('model_tf', save_format='tf')  # tf format
########################################################################################################################
# Model to Generate New Images
decoder_model.save('decoder_model_boxes')
########################################################################################################################
# Latent Feature Cluster for Test Data
x = []
y = []
z = []
for i in range(len(box_shape_test)):
    z.append(box_shape_test[i])
    op = encoder_model_boxes.predict(np.array([test_data[i]]))
    x.append(op[0][0])
    y.append(op[0][1])

df = pd.DataFrame()
df['x'] = x
df['y'] = y
df['z'] = ["digit-" + str(k) for k in z]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', hue='z', data=df)
plt.show()
########################################################################################################################
# Latent Feature Cluster for Training Data
x = []
y = []
z = []
for i in range(len(box_shape_train)):
    z.append(box_shape_train[i])
    op = encoder_model_boxes.predict(np.array([train_data[i]]))
    x.append(op[0][0])
    y.append(op[0][1])

df = pd.DataFrame()
df['x'] = x
df['y'] = y
df['z'] = ["Shape:" + str(k) for k in z]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', hue='z', data=df)
plt.show()
