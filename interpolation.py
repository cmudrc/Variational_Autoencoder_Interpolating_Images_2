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
number_2 = 7

# Creates a sequence of input values for the desired label of number_1 and number_2
indices_1 = [i for i in range(len(testy)) if testy[i] == number_1]
indices_2 = [i for i in range(len(testy)) if testy[i] == number_2]

# choose a random index
number_1 = random.choice(indices_1)
number_2 = random.choice(indices_2)

# resize the array to match the prediction size requirement
number_1_expand = np.expand_dims(number_1, axis=0)
number_2_expand = np.expand_dims(number_2, axis=0)

latent_point_1 = encoder_model.predict(test_data[number_1_expand])[0]
latent_point_2 = encoder_model.predict(test_data[number_2_expand])[0]

# Determine the latent point that will represent our desired number
xy1 = encoder_model.predict(test_data[number_1_expand])
xy2 = encoder_model.predict(test_data[number_2_expand])

xy1 = xy1[0]
xy2 = xy2[0]
latent_dimensionality = len(xy1)
print(xy1)
print(xy2)
########################################################################################################################
# Establish the Framework of the Interpolation
number_internal = 3  # the number of interpolations that the model will find between two points
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
#Add z values

########################################################################################################################
latent_matrix = []  # This will contain the latent points of the interpolation
for column in range(latent_dimensionality):
    new_column = np.linspace(latent_point_1[column], latent_point_2[column], num_interp)
    latent_matrix.append(new_column)
latent_matrix = np.array(latent_matrix).T  # Transposes the matrix so that each row can be easily indexed

plot_rows = 2
plot_columns = num_interp + 2

# Plot the First Interpolation Point
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(testX[number_1], cmap='gray', vmin=0, vmax=1)

predicted_interps = []  # Used to store the predicted interpolations
# Interpolate the Images and Print out to User
for latent_point in range(2, num_interp + 2):  # cycles the latent points through the decoder model to create images
    generated_image = decoder_model.predict(np.array([latent_matrix[latent_point - 2]]))[0]  # generates an interpolated image based on the latent point
    predicted_interps.append(generated_image[:, :, -1])
    plt.subplot(plot_rows, plot_columns, latent_point), plt.imshow(generated_image[:, :, -1], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

# Plot the Second Interpolation Point
plt.subplot(plot_rows, plot_columns, num_interp + 2), plt.imshow(testX[number_2], cmap='gray', vmin=0, vmax=1)

plt.show()
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

plot_rows = 1
plot_columns = 3
plot_height = 1
plot_width = plot_height*num_interp
plt.figure(figsize=(plot_width, plot_height))
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(testX[number_1], cmap='gray')
plt.title("Number: " + str(testy[number_1]))
plt.subplot(plot_rows, plot_columns, 2), plt.imshow(figure, cmap='gray')
plt.title("Interpolation from " + str(testy[number_1]) + " to " + str(testy[number_2]))
plt.subplot(plot_rows, plot_columns, 3), plt.imshow(testX[number_2], cmap='gray')
plt.title("Number: " + str(testy[number_2]))

plt.show()
