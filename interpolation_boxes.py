import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import tensorflow
import warnings
from tensorflow.python.framework.ops import disable_eager_execution
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
# import gradio as gr
from sklearn.decomposition import PCA
from smoothness_testing import euclidean_plot, RMSE_plot, smoothness
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from Dimensionality_Reduction_Latent_Space import PaCMAP_reduction, PCA_reduction, PCA_TSNE_reduction, TSNE_reduction, \
    plot_dimensionality_reduction, Latent_Image_Proj, plot_reduction_interpolation, plot_interpolation_smoothness
warnings.filterwarnings('ignore')
disable_eager_execution()
########################################################################################################################
# Import the Models created by the VAE
# import the decoder model
decoder_model_boxes = tensorflow.keras.models.load_model('decoder_model_boxes', compile=False)
# compile=False ignores a warning from tensorflow, can be removed to see warning

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
# Establishing the Latent Points of the Training Dataset
train_latent_points = []
train_data = np.reshape(box_matrix_train, (len(box_matrix_train), image_size, image_size, 1))
for i in range(len(box_shape_train)):
    predicted_train = encoder_model_boxes.predict(np.array([train_data[i]]))
    train_latent_points.append(predicted_train[0])
train_latent_points = np.array(train_latent_points)

########################################################################################################################
# Select Latent Points to Interpolate Between
# USE ONCE TEST DATA IS LARGE ENOUGH
# Selecting a particular set of boxes for interpolation
shapes = ("basic_box", "diagonal_box_split", "horizontal_vertical_box_split", "back_slash_box", "forward_slash_box",
          "back_slash_plus_box", "forward_slash_plus_box", "hot_dog_box", "hamburger_box", "x_hamburger_box",
          "x_hot_dog_box", "x_plus_box")

box_shape_1 = "basic_box"   # End points for the 2 point interpolation
box_shape_2 = "hamburger_box"

box_shape_3 = "hot_dog_box"  # Additional end points to use for grid interpolation
box_shape_4 = "x_plus_box"

# Creates a sequence of input values for the desired label of number_1 and number_2
indices_1 = [i for i in range(len(testX)) if box_shape_test[i] == box_shape_1]
indices_2 = [i for i in range(len(testX)) if box_shape_test[i] == box_shape_2]

indices_3 = [i for i in range(len(testX)) if box_shape_test[i] == box_shape_3] # Additional end points to use for grid interpolation
indices_4 = [i for i in range(len(testX)) if box_shape_test[i] == box_shape_4]

# choose a random index
number_1 = indices_1[0]  # random.choice(indices_1)
number_2 = indices_2[0]  # random.choice(indices_2)
number_3 = indices_3[0]
number_4 = indices_4[0]

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

number_3_expand = np.expand_dims(number_3, axis=0)
number_4_expand = np.expand_dims(number_4, axis=0)

# Determine the latent point that will represent our desired number
latent_point_1 = encoder_model_boxes.predict(test_data[number_1_expand])[0]
latent_point_2 = encoder_model_boxes.predict(test_data[number_2_expand])[0]

latent_point_3 = encoder_model_boxes.predict(test_data[number_3_expand])[0]
latent_point_4 = encoder_model_boxes.predict(test_data[number_4_expand])[0]

latent_dimensionality = len(latent_point_1)  # define the dimensionality of the latent space
########################################################################################################################
# Establish the Framework for a LINEAR Interpolation
number_internal = 8  # the number of interpolations that the model will find between two points
num_interp = number_internal + 2  # the number of images to be pictured
latent_matrix = []  # This will contain the latent points of the interpolation
for column in range(latent_dimensionality):
    new_column = np.linspace(latent_point_1[column], latent_point_2[column], num_interp)
    latent_matrix.append(new_column)
latent_matrix = np.array(latent_matrix).T  # Transposes the matrix so that each row can be easily indexed
########################################################################################################################
# Plotting the Interpolation in 2D Using Chosen Points
plot_rows = 2
plot_columns = num_interp + 2

# Plot the First Interpolation Point
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(testX[number_1], cmap='gray', vmin=0, vmax=1)
plt.title("First Interpolation Point:\n" + str(box_shape_test[number_1]) + "\nPixel Density: " + str(
            box_density_test[number_1]) + "\nAdditional Pixels: " + str(additional_pixels_test[number_1]))  # + "\nPredicted Latent Point 1: " + str(latent_point_1)

predicted_interps = []  # Used to store the predicted interpolations
# Interpolate the Images and Print out to User
for latent_point in range(2, num_interp + 2):  # cycles the latent points through the decoder model to create images
    generated_image = decoder_model_boxes.predict(np.array([latent_matrix[latent_point - 2]]))[0]  # generates an interpolated image based on the latent point
    predicted_interps.append(generated_image[:, :, -1])
    plt.subplot(plot_rows, plot_columns, latent_point), plt.imshow(generated_image[:, :, -1], cmap='gray', vmin=0, vmax=1)
    # plt.axis('off')

# Plot the Second Interpolation Point
plt.subplot(plot_rows, plot_columns, num_interp + 2), plt.imshow(testX[number_2], cmap='gray', vmin=0, vmax=1)
plt.title("Second Interpolation Point:\n" + str(box_shape_test[number_2]) + "\nPixel Density: " + str(
            box_density_test[number_2]) + "\nAdditional Pixels: " + str(additional_pixels_test[number_2]))  # + "\nPredicted Latent Point 2: " + str(latent_point_2)
plt.show()


########################################################################################################################
# Smoothness Evaluation based on Standard Deviations from the mean
run_std = input("Would you like to run standard deviation evaluations? (yes/no)")

if run_std == "yes":
    print("std")
    print(np.std(train_latent_points, axis=0))
    print("mean")
    print(np.mean(train_latent_points, axis=0))

    train_mean = np.mean(train_latent_points, axis=0)
    train_std = np.std(train_latent_points, axis=0)

    latent_point_1_std = train_mean-3*train_std # starting point of the interpolation

    # Currently set to measure a distance of 6 standard deviations

    smoothness_array = []
    num_std = []
    interp_length = []

    for test_distance_interps in [5, 10, 15]:
        count_array = []
        smoothness_percent = []
        for count, latent_point_2_std in enumerate([train_mean-2*train_std, train_mean-train_std, train_mean, train_mean+train_std, train_mean+2*train_std, train_mean+3*train_std]): #

            latent_matrix_std = []
            for column in range(latent_dimensionality):
                new_column = np.linspace(latent_point_1_std[column], latent_point_2_std[column], test_distance_interps)
                latent_matrix_std.append(new_column)
            latent_matrix_std = np.array(latent_matrix_std).T  # Transposes the matrix so that each row can be easily indexed

            predicted_interps_std = []  # Used to store the predicted interpolations
            # Interpolate the Images and Print out to User
            for latent_point in range(2, test_distance_interps + 2):  # cycles the latent points through the decoder model to create images
                generated_image = decoder_model_boxes.predict(np.array([latent_matrix_std[latent_point - 2]]))[0]  # generates an interpolated image based on the latent point
                predicted_interps_std.append(generated_image[:, :, -1])

            # Determining Smoothness using Gradient
            smoothness_average, smoothness_std = smoothness(predicted_interps_std, plot=False)

            count_array.append(count+1)
            smoothness_percent.append(smoothness_average)

            smoothness_array.append(smoothness_average)
            num_std.append(count + 1)
            interp_length.append(test_distance_interps)

        plt.scatter(count_array, smoothness_percent)
        plt.xlabel("Number of Standard Deviations from the Mean", fontsize=16)
        plt.ylabel("Smoothness (%)", fontsize=16)
        plt.title("Number of Transition Points: " + str(test_distance_interps), fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim([60, 100])
        plt.show()
    OLS_array = np.column_stack((num_std, interp_length, np.multiply(num_std, interp_length)))

    Y = smoothness_array

    # reg = LinearRegression().fit(OLS_array, Y)
    # reg.score(OLS_array, Y)

    distance_std = np.linalg.norm(train_std)
    distance_original_interp = np.linalg.norm(np.subtract(latent_point_1, latent_point_2))
    print("Distance between latent points")
    print(distance_original_interp)

        # reg.predict((original_num_std, num_interp, original_num_std*num_interp))

    import statsmodels.api as sm

    OLS_array = sm.add_constant(OLS_array)
    print("OLS")
    print(Y, OLS_array)

    mod = sm.OLS(Y, OLS_array).fit()
    print(mod.summary(yname='Smoothness (%)', xname=['Constant:','Number of Standard Deviations:', 'Transition Length:', 'Cross-Term:']))

    # Testing the Prediction of the Model
    original_num_std = distance_original_interp / distance_std
    original_array = np.array([1, original_num_std, num_interp, original_num_std*num_interp])
    print(original_array)
    print("Smoothness Prediction: ")
    print(mod.predict(original_array))
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
# Create a PCA Plot of the Latent Space with Images Superimposed
Latent_Image_Proj(box_matrix_train, image_size, train_latent_points, latent_dimensionality, reduction_function=PCA_reduction)

Latent_Image_Proj(box_matrix_train, image_size,train_latent_points, latent_dimensionality, reduction_function=PaCMAP_reduction)
########################################################################################################################
# Determining Smoothness using Gradient
smoothness(predicted_interps, plot=True)

########################################################################################################################
# Restricting the images to binary values
constrained_predictions = np.reshape(predicted_interps, (np.shape(predicted_interps)[-1] ** 2, len(predicted_interps)))

for i in range(len(constrained_predictions)):
    constrained_predictions[:][i][np.where(constrained_predictions[:][i] >= abs(box_density_test[number_1] - box_density_test[number_2])/2)] = 1
    constrained_predictions[:][i][np.where(constrained_predictions[:][i] < abs(box_density_test[number_1] - box_density_test[number_2])/2)] = 0

constrained_predictions = np.reshape(constrained_predictions, (len(predicted_interps), image_size, image_size))

'''
for i in range(1, num_interp):  # cycles the latent points through the decoder model to create images
    plt.subplot(plot_rows, plot_columns, i+1), plt.imshow(constrained_predictions[i], cmap='gray', vmin=0, vmax=1)
plt.show()
'''

smoothness(constrained_predictions)
########################################################################################################################
# Plotting the Euclidean and RMSE Values between each step in the interpolation
# This was an old approach to determining smoothness
'''
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
'''
########################################################################################################################
# Use to make an interpolation grid between 4 images

latent_matrix_2 = []  # This will contain the latent points of the interpolation
for column in range(latent_dimensionality):
    new_column = np.linspace(latent_point_3[column], latent_point_4[column], num_interp)
    latent_matrix_2.append(new_column)
latent_matrix_2 = np.array(latent_matrix_2).T  # Transposes the matrix so that each row can be easily indexed

mesh = []  # This will create a mesh by interpolating between the two interpolations
for column in range(num_interp):
    row = np.linspace(latent_matrix[column], latent_matrix_2[column], num_interp)
    mesh.append(row)

mesh = np.transpose(mesh, axes=(1, 0, 2))  # Transpose the array so it matches the original interpolation
generator_model = decoder_model_boxes

figure = np.zeros((28 * num_interp, 28 * num_interp))

mesh_predicted_interps = []
for i in range(num_interp):
    for j in range(num_interp):
        generated_image = generator_model.predict(np.array([mesh[i][j]]))[0]
        figure[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28, ] = generated_image[:, :, -1]
        mesh_predicted_interps.append(generated_image[:, :, -1])

plt.figure(figsize=(15, 15))
plt.imshow(figure, cmap='gray')
plt.show()


# for row in (0,3,5,9):
#     plt.imshow(np.concatenate(mesh_predicted_interps[row,:], axis=1), cmap='gray')
#     plt.show()
#
#
# for col in (0,3,5,9):
#     plt.imshow(np.concatenate(mesh_predicted_interps[:,col], axis=0), cmap='gray')
#     plt.show()

'''
########################################################################################################################
# Preparing the Data to be Plotted
trainX = box_matrix_train
train_data = np.reshape(trainX, (len(trainX), image_size, image_size, 1))
x = []
y = []
z = []
avg_density = []  # An integer label that is based on the average density of the matrix

for i in range(len(box_shape_train)):
    z.append(box_shape_train[i])
    op = encoder_model_boxes.predict(np.array([train_data[i]]))
    x.append(op[0][0])
    y.append(op[0][1])
    avg_density.append(np.average(box_matrix_train[i]))

########################################################################################################################
# Latent Feature Cluster for Training Data (Only works for 2-dimensions)
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
x, y, title = TSNE_reduction(train_latent_points, latent_dimensionality)
plot_dimensionality_reduction(x, y, box_shape_train, title)
plt.show()
########################################################################################################################
# Latent Feature Cluster for Training Data using PCA reduced T-SNE and Predicted Latent Points
x, y, title = PCA_TSNE_reduction(train_latent_points, latent_dimensionality)
plot_dimensionality_reduction(x, y, box_shape_train, title)
plt.show()
########################################################################################################################
# Latent Feature Cluster for Training Data using PCA and Predicted Latent Points
plot_reduction_interpolation(train_latent_points, box_shape_train, latent_matrix, latent_dimensionality,
                             image_size=image_size, image_arrays=box_matrix_train,
                             title="PCA Reduced Latent Space with Visualization of Interpolation")

########################################################################################################################
'''


# Latent Feature Cluster for Training Data using PCA and Predicted Grid Latent Points

mesh_flat = np.reshape(mesh, (num_interp**2, latent_dimensionality))
# train_data_latent_grid = np.append(train_latent_points, mesh_flat, axis=0)

plot_reduction_interpolation(train_latent_points, box_shape_train, mesh_flat, latent_dimensionality,
                             image_size=image_size, image_arrays=box_matrix_train, markersize=8,
                             marker_color='red',
                             title="PCA Reduction of Mesh Interpolation", plot_lines=False)

plot_reduction_interpolation(train_latent_points, box_shape_train, mesh_flat, latent_dimensionality,
                             image_size=image_size, image_arrays=box_matrix_train, markersize=8,
                             marker_color='red',
                             title="PCA Reduction of Mesh Interpolation", plot_lines=False, plot_points=False)

plot_interpolation_smoothness(train_latent_points, box_shape_train, mesh_flat, mesh_predicted_interps, latent_dimensionality,
                             image_size=image_size, number_of_interpolations=num_interp,image_arrays=box_matrix_train, markersize=8,
                             marker_color='red',
                             title="PCA Reduction of Mesh Interpolation", plot_lines=False, plot_points=True)

########################################################################################################################
# Latent Feature Cluster for Training Data using PaCMAP and Predicted Latent Points
x, y, title = PaCMAP_reduction(train_latent_points, latent_dimensionality)
plot_dimensionality_reduction(x, y, box_shape_train, title)


plot_dimensionality_reduction(x, y, avg_density, "PaCMAP Reduction: Labeled with Respect to Average Density of Pixels")

