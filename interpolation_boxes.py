import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
from box_data_set import make_boxes
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
disable_eager_execution()

# import the decoder model
decoder_model = tensorflow.keras.models.load_model('decoder_model_boxes')

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
image_size = 28
box_data = make_boxes(image_size)
box_data_train, box_data_test = train_test_split(box_data, test_size=0.15)  # Uses a similar percentage as MNIST Data

box_matrix_train, box_density_train, additional_pixels_train, box_shape_train = list(zip(*box_data_train))[0], list(zip(*box_data_train))[1], list(zip(*box_data_train))[2], list(zip(*box_data_train))[3]
box_matrix_test, box_density_test, additional_pixels_test, box_shape_test = list(zip(*box_data_test))[0], list(zip(*box_data_test))[1], list(zip(*box_data_test))[2], list(zip(*box_data_test))[3]
testX = box_matrix_test
test_data = np.reshape(testX, (len(testX), image_size, image_size, 1))
########################################################################################################################
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

# choose a random index
# number_1 = random.choice(indices_1)
# number_2 = random.choice(indices_2)

# resize the array to match the prediction size requirement
number_1 = np.expand_dims(number_1, axis=0)
number_2 = np.expand_dims(number_2, axis=0)
print(number_1)
# Determine the latent point that will represent our desired number
xy1 = encoder_model.predict(test_data[number_1])
xy2 = encoder_model.predict(test_data[number_2])

xy1 = xy1[0]
xy2 = xy2[0]
latent_dimensionality = len(xy1)
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
    print(latent_point)
    generated_image = decoder_model.predict(latent_point)[0]  # generates an interpolated image based on the latent point
    figure[0: 28,  iy * 28:(iy + 1) * 28, ] = generated_image[:, :, -1]  # Inserts the Pixel Value for Each Image

plt.figure()
plt.imshow(figure,cmap='gray')
plt.show()
