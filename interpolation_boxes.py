import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
from tensorflow.python.framework.ops import disable_eager_execution
warnings.filterwarnings('ignore')
disable_eager_execution()
########################################################################################################################
# Import the Models created by the VAE
# import the decoder model
decoder_model_boxes = tensorflow.keras.models.load_model('decoder_model_boxes')

# import the encoder model architecture
json_file_loaded = open('model.json', 'r')
loaded_model_json = json_file_loaded.read()

# load model using the saved json file
encoder_model_boxes = tensorflow.keras.models.model_from_json(loaded_model_json)

# load weights into newly loaded_model
encoder_model_boxes.load_weights('model_tf')
########################################################################################################################
# Import the Test and Training Data used by the VAE
load_data = np.load('train_test_split_data.npz')  # Data saved by the VAE

# Convert Data to Tuples and Assign to respective variables
box_matrix_train, box_density_train, additional_pixels_train, box_shape_train = tuple(load_data['box_matrix_train']), tuple(load_data['box_density_train']), tuple(load_data['additional_pixels_train']), tuple(load_data['box_shape_train'])
box_matrix_test, box_density_test, additional_pixels_test, box_shape_test = tuple(load_data['box_matrix_test']), tuple(load_data['box_density_test']), tuple(load_data['additional_pixels_test']), tuple(load_data['box_shape_test'])
testX = box_matrix_test  # Shows the relationship to the MNIST Dataset vs the Shape Dataset
image_size = np.shape(testX)[-1]  # Determines the size of the images
test_data = np.reshape(testX, (len(testX), image_size, image_size, 1))
########################################################################################################################
# Select Latent Points to Interpolate Between
""" # USE ONCE TEST DATA IS LARGE ENOUGH
# Selecting a particular set of boxes for interpolation
box_density_1 = 0.2
box_density_2 = 1

additional_pixels_1 = 1
additional_pixels_2 = 0

box_shape_1 = "Basic_Box"  # Select from "Basic_Box", "Diagonal_Box_Split", and "Horizontal_Box_Split"
box_shape_2 = "Basic_Box"

# Creates a sequence of input values for the desired label of number_1 and number_2
number_1 = [i for i in range(len(testX)) if box_density_test[i] == box_density_1 and additional_pixels_test[i] == additional_pixels_1 and box_shape_test[i] == box_shape_1]
number_2 = [i for i in range(len(testX)) if box_density_test[i] == box_density_2 and additional_pixels_test[i] == additional_pixels_2 and box_shape_test[i] == box_shape_2]
"""

# Randomly selects two points for interpolation
number_1 = [i for i in range(len(testX))]  # creates a list of possible indices
number_1 = random.choice(number_1)  # chooses a random index

number_2 = [i for i in range(len(testX)) if i != number_1]
number_2 = random.choice(number_2)  # chooses a random index

# resize the array to match the prediction size requirement
number_1_expand = np.expand_dims(number_1, axis=0)
number_2_expand = np.expand_dims(number_2, axis=0)

# Determine the latent point that will represent our desired number
latent_point_1 = encoder_model_boxes.predict(test_data[number_1_expand])
latent_point_2 = encoder_model_boxes.predict(test_data[number_2_expand])

# Index the latent point as a point rather than an array
latent_point_1 = latent_point_1[0]
latent_point_2 = latent_point_2[0]
latent_dimensionality = len(latent_point_1)  # define the dimensionality of the latent space
########################################################################################################################
# Establish the Framework a LINEAR Interpolation
number_internal = 13  # the number of interpolations that the model will find between two points
num_interp = number_internal + 2  # the number of images to be pictured
figure = np.zeros((image_size, image_size * num_interp))  # The matrix that will hold the pixel values of the images
latent_matrix = []

for column in range(latent_dimensionality):
    new_column = np.linspace(latent_point_1[column], latent_point_2[column], num_interp)
    latent_matrix.append(new_column)
latent_matrix = np.array(latent_matrix).T  # Transposes the matrix so that each row can be easily indexed
########################################################################################################################
# Interpolate the Images and Print out to User
for latent_point in range(num_interp):  # cycles the latent points through the decoder model to create images
    generated_image = decoder_model_boxes.predict(np.array([latent_matrix[latent_point]]))[0]  # generates an interpolated image based on the latent point
    figure[0: image_size,  latent_point * image_size:(latent_point + 1) * image_size, ] = generated_image[:, :, -1]  # Inserts the Pixel Value for Each Image

plot_rows = 1
plot_columns = 3
plot_height = 1
plot_width = plot_height*num_interp
plt.figure(figsize=(plot_width, plot_height))
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(testX[number_1], cmap='gray')
plt.title("First Interpolation Point:\n" + str(box_shape_test[number_1]) + "\nPixel Density: " + str(
            box_density_test[number_1]) + "\nAdditional Pixels: " + str(additional_pixels_test[number_1]))  # + "\nPredicted Latent Point 1: " + str(latent_point_1)
plt.subplot(plot_rows, plot_columns, 2), plt.imshow(figure, cmap='gray')
plt.title("Interpolation from First to Second Interpolation Point")
plt.subplot(plot_rows, plot_columns, 3), plt.imshow(testX[number_2], cmap='gray')
plt.title("Second Interpolation Point:\n" + str(box_shape_test[number_2]) + "\nPixel Density: " + str(
            box_density_test[number_2]) + "\nAdditional Pixels: " + str(additional_pixels_test[number_2]))  # + "\nPredicted Latent Point 2: " + str(latent_point_2)
plt.show()
