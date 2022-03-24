import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
from tensorflow.python.framework.ops import disable_eager_execution
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import gradio as gr
from sklearn.decomposition import PCA
from smoothness_testing import euclidean_plot, RMSE_plot, smoothness
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pacmap  # will need to change numba version: pip install numba==0.53
from Dimensionality_Reduction_Latent_Space import PaCMAP_plot, PCA_plot, PCA_TSNE_plot, TSNE_plot, plot_dimensionality_reduction
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
# USE ONCE TEST DATA IS LARGE ENOUGH
# Selecting a particular set of boxes for interpolation
shapes = ("basic_box", "diagonal_box_split", "horizontal_vertical_box_split", "back_slash_box", "forward_slash_box",
              "back_slash_plus_box", "forward_slash_plus_box", "hot_dog_box", "hamburger_box", "x_hamburger_box",
              "x_hot_dog_box", "x_plus_box")

box_shape_1 = "x_hot_dog_box"
box_shape_2 = "x_plus_box"

# Creates a sequence of input values for the desired label of number_1 and number_2
indices_1 = [i for i in range(len(testX)) if box_shape_test[i] == box_shape_1]
indices_2 = [i for i in range(len(testX)) if box_shape_test[i] == box_shape_2]

print(indices_1)
print(indices_2)

# choose a random index
number_1 = indices_1[0]  # random.choice(indices_1)
number_2 = indices_2[0]  # random.choice(indices_2)

'''
# Use if not enough test data
# Randomly selects two points for interpolation
number_1 = [i for i in range(len(testX))]  # creates a list of possible indices
number_1 = random.choice(number_1)  # chooses a random index

number_2 = [i for i in range(len(testX)) if i != number_1]
number_2 = random.choice(number_2)  # chooses a random index
'''

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
# Establish the Framework for a LINEAR Interpolation
number_internal = 13  # the number of interpolations that the model will find between two points
num_interp = number_internal + 2  # the number of images to be pictured
latent_matrix = []
for column in range(latent_dimensionality):
    new_column = np.linspace(latent_point_1[column], latent_point_2[column], num_interp)
    latent_matrix.append(new_column)
latent_matrix = np.array(latent_matrix).T  # Transposes the matrix so that each row can be easily indexed
########################################################################################################################
# Plotting the Interpolation in 2D
plot_rows = 2
plot_columns = num_interp + 2

# Plot the First Interpolation Point
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(testX[number_1], cmap='gray', vmin=0, vmax=1)
plt.title("First Interpolation Point:\n" + str(box_shape_test[number_1]) + "\nPixel Density: " + str(
            box_density_test[number_1]) + "\nAdditional Pixels: " + str(additional_pixels_test[number_1]))  # + "\nPredicted Latent Point 1: " + str(latent_point_1)

predicted_interps = []  # Used to store the predicted interpolations
# Interpolate the Images and Print out to User
for latent_point in range(2, num_interp + 2):  # cycles the latent points through the decoder model to create images
    # generated_image.append((decoder_model_boxes.predict(np.array([latent_matrix[latent_point]]))[0]))
    generated_image = decoder_model_boxes.predict(np.array([latent_matrix[latent_point - 2]]))[0]  # generates an interpolated image based on the latent point
    predicted_interps.append(generated_image[:, :, -1])
    plt.subplot(plot_rows, plot_columns, latent_point), plt.imshow(generated_image[:, :, -1], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

# Plot the Second Interpolation Point
plt.subplot(plot_rows, plot_columns, num_interp + 2), plt.imshow(testX[number_2], cmap='gray', vmin=0, vmax=1)
plt.title("Second Interpolation Point:\n" + str(box_shape_test[number_2]) + "\nPixel Density: " + str(
            box_density_test[number_2]) + "\nAdditional Pixels: " + str(additional_pixels_test[number_2]))  # + "\nPredicted Latent Point 2: " + str(latent_point_2)
plt.show()
########################################################################################################################
# Plotting the Interpolation in 3D
voxel_interpolation = np.where(np.array(predicted_interps) > 0.1, predicted_interps, 0)

# Create a new figure
fig = plt.figure()

# Axis with 3D projection
ax = fig.add_subplot(projection='3d')

# Plot the voxels
cmap = plt.get_cmap("gray")
ax.voxels(voxel_interpolation, edgecolor="k", facecolors=cmap(voxel_interpolation))

# Display the plot
plt.show()
########################################################################################################################
# Determining Smoothness using Gradient
smoothness(predicted_interps)
########################################################################################################################
# Plotting the Euclidean and RMSE Values between each step in the interpolation
euclidean_plot(predicted_interps, num_interp)  # will calculate and plot the euclidean distances between each step in the interpolation
RMSE_plot(predicted_interps, num_interp)

# Difference between each Point between two images in interpolation
predicted_interps = np.reshape(predicted_interps, (num_interp, image_size**2))  # flatten the array into a vector

for i in range(num_interp-1):
    difference = predicted_interps[i] - predicted_interps[i+1]
    plt.boxplot(difference, positions=[i])
plt.xlabel("Set of Interpolation")
plt.ylabel("Difference between Interpolation Pixels")
plt.title("\nLatent Space Dimensionality: " + str(latent_dimensionality))
plt.show()

########################################################################################################################
# Use to make an interpolation grid between 4 images
'''
# Grid Interpolation
generator_model = decoder_model_boxes

x_values = np.linspace(-3, 3, 30)
y_values = np.linspace(-3, 3, 30)

figure = np.zeros((28 * 30, 28 * 30))
for ix, x in enumerate(x_values):
    for iy, y in enumerate(y_values):
        latent_point = np.array([[x, y]])
        generated_image = generator_model.predict(latent_point)[0]
        figure[ix * 28:(ix + 1) * 28, iy * 28:(iy + 1) * 28, ] = generated_image[:, :, -1]

plt.figure(figsize=(15, 15))
plt.imshow(figure, cmap='gray', extent=[3, -3, 3, -3])
plt.show()
'''
########################################################################################################################
# Preparing the Data to be Plotted
trainX = box_matrix_train
train_data = np.reshape(trainX, (len(trainX), image_size, image_size, 1))
x = []
y = []
z = []
avg_density = []  # An integer label that is based on the average density of the matrix
latent_points = []
for i in range(len(box_shape_train)):
    z.append(box_shape_train[i])
    op = encoder_model_boxes.predict(np.array([train_data[i]]))
    latent_points.append(op[0])
    x.append(op[0][0])
    y.append(op[0][1])
    avg_density.append(np.average(box_matrix_train[i]))
print(avg_density[0])
print(np.shape(avg_density))
########################################################################################################################
# Latent Feature Cluster for Training Data (Only works for 2-dimensions)
latent_points = np.array(latent_points)
df = pd.DataFrame()
df['x'] = x
df['y'] = y
df['z'] = ["Shape:" + str(k) for k in z]  # Acts as a grouping variable to produce points with different colors

plt.figure(figsize=(8, 6))
plt.title("Plot of Predicted Latent Points without Dimensionality Reduction")
sns.scatterplot(x='x', y='y', hue='z', data=df)
plt.show()
########################################################################################################################
# Latent Feature Cluster for Training Data using T-SNE
flattened = np.reshape(trainX, (np.shape(trainX)[0], np.shape(trainX)[1]**2))
perplexity = 30
learning_rate = 20

# pca = PCA(n_components=latent_dimensionality)  # PCA can be used to assist with reducing dimensionality in cases where the latent space is large
# flattened = pca.fit_transform(flattened)
model = TSNE(n_components=2, random_state=0,  perplexity=perplexity, learning_rate=learning_rate)  # Perplexity(5-50) | learning_rate(10-1000)
# configuring the parameters
# the number of components = dimension of the embedded space
# default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data.
# It is roughly a guess of the number of close neighbors each point has. ..a denser dataset ... requires higher perplexity value"
# default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point
# approximately equidistant from its nearest neighbours. If the learning rate is too low,
# most points may look compressed in a dense cloud with few outliers."
tsne_data = model.fit_transform(flattened) # When there are more data points, trainX should be the first couple hundred points so TSNE doesn't take too long

plt.figure(figsize=(8, 6))
plt.title("T-SNE of Original Training Data\nPerplexity: " + str(perplexity) + "\nLearning Rate: " + str(learning_rate) + "\nLatent Space Dimensionality: " + str(latent_dimensionality))
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue='z', data=df)
plt.show()


########################################################################################################################
# Latent Feature Cluster for Training Data using T-SNE and Predicted Latent Points
x, y, title = TSNE_plot(latent_points, latent_dimensionality)
plot_dimensionality_reduction(x, y, box_shape_train, title)
plt.show()
########################################################################################################################
# Latent Feature Cluster for Training Data using PCA reduced T-SNE and Predicted Latent Points
x, y, title = PCA_TSNE_plot(latent_points, latent_dimensionality)
plot_dimensionality_reduction(x, y, box_shape_train, title)
plt.show()
########################################################################################################################
# Latent Feature Cluster for Training Data using PCA and Predicted Latent Points

all_latent_points = np.vstack((latent_points, latent_matrix))
all_latent_labels = box_shape_train
for i in range(len(latent_matrix)):
    all_latent_labels = np.append(all_latent_labels, "Predicted_Latent_Point")
print(np.shape(all_latent_points))
print(np.shape(np.flipud(all_latent_points)))
print(all_latent_points[0])
print(np.flipud(all_latent_points)[-1])
x, y, title = PCA_plot(np.flipud(all_latent_points), latent_dimensionality,)

plot_dimensionality_reduction(x, y, np.flipud(all_latent_labels), title)

########################################################################################################################
# Latent Feature Cluster for Training Data using PaCMAP and Predicted Latent Points
x, y, title = PaCMAP_plot(latent_points, latent_dimensionality)
plot_dimensionality_reduction(x, y, box_shape_train, title)

plot_dimensionality_reduction(x, y, avg_density, title)

'''
def shape_select(number):
    return "This is your" + str(number)


iface = gr.Interface(
    fn=shape_select,
    inputs=gr.inputs.Dropdown((1, 2, 3), type="value", default=None, label=None, optional=False),
    outputs="text",
    interpretation=None,
)

iface.launch(share=True)
'''
