import numpy as np
# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
"""
By changing this to 64, we get a val_loss of 0.0723, compared to 0.0915 from an encoding dimension of 32,
indicating that this model fits the data better, but we are not decompressing the data as small
when the encoding dimension is decreased to 8, the val_loss becomes 0.1631 and the output images are extremely blurry,
because the decoder has very little data to go off of and must interpolate a lot of values
It should be noted that increasing the dimension to 128, we see an longer training time, with 3ms/step
"""


# This is our input image
input_img = keras.Input(shape=(784,))  # We are defining the input images to be a 784 length vector
# This is our input placeholder

"""
.Input: will create a tensor which can be modified with attributes in order to build a Keras model, just based on the
inputs and outputs of the model
tensor: is a dynamic mathematical entity that obeys specific transformation rules as a part of its inhabited structure
Argument: the way in which we are modifying the data, in this case "shape" is used
shape: a tuple that has an expected input of 784-dimensional vectors. If 'None' is inserted, then we indicate that we
don't know the shape and its dimensions
"""
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
print(encoded.shape[1])
decoded = layers.Dense(784, activation='sigmoid')(encoded)
"""
Layers: are another class that maintain a state that is updated when the layer receives data during training, and is 
stored in layer.weights
Dense: is a class used to implement the activation function
Activation:  a function available on an activation layer or argument, 
the available activations include:
    relu - applies the rectified linear unit function
    sigmoid - 1 / (1 + exp(-x)), and will return a value of 0 for numbers <-5, >5 the function gets close to 1
    tanh - hyperbolic tangent activation function
    and more that are not common in ML
"""

# tf.keras.backend.clear_session()
# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)  # this will model the input and output from the model
"""
Model: a class that groups layers into an object with training and inference features
Arugments: inputs- a keras.Input object or list of objects, outputs- , and name
Only dictionaires, lists or tuples are supported by this class. Cannot use nested inputs.
"""

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

"""
Sets the "encoder" model inputs equal to the input_img vector and the output model equal to the encoded. Encoder is 
acting as a functional object.
"""
# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))

# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
"""
This will index the last layer from the autoencoder
"""

# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
"""
This indicates that the last layer of the autoencoder is where the decoder output is coming from
"""
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
"""
Compile: a method that is used to assign more information to the model, in this case, we want to know the optimizer
(a weight tuner) and the loss function
"""

(x_train, _), (x_test, _) = mnist.load_data()
"""
x_train: a data set that consists of 60,000 images. It is defined using unsigned integers for the gray-scale values
ranging from 0 to 255. 
x_test: a data set of 10000 images with vectors of length 784 assigned to each
"""

x_train = x_train.astype('float32') / 255.  # This will make the values of the gray-scale vary from 0 to 1
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))  # figsize defines the width and height of the total number of figures in inches
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
