import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
from tensorflow.keras.datasets import mnist
from tensorflow.python.framework.ops import disable_eager_execution
warnings.filterwarnings('ignore')
disable_eager_execution()

# import the decoder model
decoder_model = tensorflow.keras.models.load_model('decoder_model')

# import the encoder model architecture
json_file_loaded = open('model.json', 'r')
loaded_model_json = json_file_loaded.read()

# load model using the saved json file
encoder_model = tensorflow.keras.models.model_from_json(loaded_model_json)

# load weights into newly loaded_model
# encoder_model.load_weights('model_h5')
encoder_model.load_weights('model_tf')
########################################################################################################################
# Import the Test Data
(trainX, trainy), (testX, testy) = mnist.load_data()
test_data = testX.astype('float32') / 255
test_data = np.reshape(test_data, (10000, 28, 28, 1))
########################################################################################################################
# Selecting a particular set of numbers of interpolation
number_1 = 0
number_2 = 3

# Creates a sequence of input values for the desired label of number_1 and number_2
indices_1 = [i for i in range(len(testy)) if testy[i] == number_1]
indices_2 = [i for i in range(len(testy)) if testy[i] == number_2]

# choose a random index
number_1 = random.choice(indices_1)
number_2 = random.choice(indices_2)

# resize the array to match the prediction size requirement
number_1 = np.expand_dims(number_1, axis=0)
number_2 = np.expand_dims(number_2, axis=0)

# Determine the latent point that will represent our desired number
xy1 = encoder_model.predict(test_data[number_1])
xy2 = encoder_model.predict(test_data[number_2])

xy1 = xy1[0]
xy2 = xy2[0]
print(xy1)
print(xy2)
########################################################################################################################
# Establish the Framework of the Interpolation
number_internal = 13  # the number of interpolations that the model will find between two points
num_interp = number_internal + 2  # the number of images to be pictured
figure = np.zeros((28, 28 * num_interp))  # The matrix that will hold the pixel values of the images
#xy1 = [0, -3]  # Latent Point 1
#xy2 = [0, 3]  # Latent Point 2
x_values = np.linspace(xy1[0], xy2[0], num_interp)  # Interpolates the values of x between the two latent points

print("Linspace X")
#print(x_values)
y_values = np.linspace(xy1[1], xy2[1], num_interp) # Interpolates the values of y between the two latent points
print("Linspace Y")
#print(y_values)

########################################################################################################################
# Interpolate the Images and Print out to User
print('Latent Points')
for iy, y in enumerate(y_values):  # enumerate creates a list of tuples [(0,-3), ... (29, 3)]
    # iy will print out the column numbers
    latent_point = np.array([[x_values[iy], y]])  # this creates the points that will establish the framework of the matrix
    # for the images to populate around, in this case makes:
    # [ [0,-3]  ... [0,3] ]
    #print(latent_point)
    generated_image = decoder_model.predict(latent_point)[0]  # generates an interpolated image based on the latent point
    figure[0: 28,  iy * 28:(iy + 1) * 28, ] = generated_image[:, :, -1]  # Inserts the Pixel Value for Each Image

plt.figure()
plt.imshow(figure,cmap='gray')
plt.show()
