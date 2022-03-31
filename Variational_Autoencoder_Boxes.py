import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
import pandas as pd
import seaborn as sns
from tensorflow.python.framework.ops import disable_eager_execution
from box_data_set import make_boxes
from sklearn.model_selection import train_test_split
from keras import backend as K


warnings.filterwarnings('ignore')
disable_eager_execution()

########################################################################################################################
# Define the parameters of the Data
image_size = 28  # pixel size of the data you wish you compute (Even numbers required)
number_of_densities = 5  # The number of densities that will be equally spaced between the min and max
min_density = 0  # (Recommend: 0) The minimum density IS NOT included in the data created, it only serves as a placeholder
max_density = 1  # (Recommend: 1) The maximum density IS included in the data created
# Choosing different mins and maxes will require the data to be normalized


# Define the parameters of the Autoencoder
latent_dimensionality = 4
early_stopping_patience = 9  # "Number of epochs with no improvement after which training will be stopped." - Keras

########################################################################################################################
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
# If the data does not range between 0 and 1, it must first be normalized, this data is then re-shaped

train_data = trainX.astype('float32') / (max_density-min_density)
test_data = testX.astype('float32') / (max_density-min_density)
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

decoder_output = tensorflow.keras.layers.Conv2DTranspose(1, (5, 5), activation='sigmoid')(decoder)


decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
decoder_model.summary()
########################################################################################################################
# The framework for the autoencoder is established
encoded = encoder_model_boxes(input_data)
decoded = decoder_model(encoded)
autoencoder = tensorflow.keras.models.Model(input_data, decoded)
autoencoder.summary()


########################################################################################################################
# Functions defined to be used by the Autoencoder

def coeff_determination(y_true, y_pred):  # This function will be used by the autoencoder to return the R^2 value
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


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


########################################################################################################################
# Autoencoder is trained using the training data
checkpoint_filepath = 'train_ckpt/cp.ckpt'  # A filepath is defined for the checkpoint data to be saved

callback = [tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping_patience),
            tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')]
autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam', metrics=[coeff_determination])
autoencoder_fit = autoencoder.fit(train_data, train_data, epochs=150, batch_size=32, validation_data=(test_data, test_data), callbacks=[callback])
########################################################################################################################
# Plotting the Loss Compared to R^2
plt.plot(autoencoder_fit.history["loss"], label="Training Loss")
plt.plot(autoencoder_fit.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

plt.plot(autoencoder_fit.history["coeff_determination"], label="Training Coefficient of Determination")
plt.plot(autoencoder_fit.history["val_coeff_determination"], label="Validation Coefficient of Determination")
plt.title("Best Training R^2: " + str(max(autoencoder_fit.history["coeff_determination"])) + "\nBest Validation R^2: " + str(max(autoencoder_fit.history["val_coeff_determination"])))
plt.axhline(y=0.95, color='r', linestyle='-', label="95% Coefficient of Determination")
plt.ylim(bottom=0)
plt.legend()
plt.show()


# Plot Loss and R^2 Simultaneously
fig, ax1 = plt.subplots()
plt.title("Latent Space Dimensionality: " + str(latent_dimensionality) + "\nBest Training R^2: " + str(max(autoencoder_fit.history["coeff_determination"])) + "\nBest Validation R^2: "
          + str(max(autoencoder_fit.history["val_coeff_determination"])) + "\nTotal Epochs: " + str(len(autoencoder_fit.history["coeff_determination"])))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
plt.xlim(0, 150)
plt.ylim(0, 150)
ax1.plot(autoencoder_fit.history["loss"], label="Training Loss", color='blue')
ax1.plot(autoencoder_fit.history["val_loss"], label="Validation Loss", color='orange')
# ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Coefficient of Determination')  # we already handled the x-label with ax1
plt.ylim(0, 1.1)
ax2.plot(autoencoder_fit.history["coeff_determination"], label="Training Coefficient of Determination", color='cornflowerblue')
ax2.plot(autoencoder_fit.history["val_coeff_determination"], label="Validation Coefficient of Determination", color='moccasin')
ax2.axhline(y=0.95, color='r', linestyle='-', label="95% Coefficient of Determination")
# ax2.tick_params(axis='y')
fig.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
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
# Prints out the important information about the model
print("Number of Training Points: " + str(len(trainX)))
print("Number of Test Points: " + str(len(testX)))
print("Latent Space Dimensionality: " + str(latent_dimensionality))


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
