import random
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import math
import scipy
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D


image_size = 28
pixel_step_change = 2  # the number of pixels that will be added in between each step

zeros = np.zeros((image_size, image_size))
ones = np.ones((image_size, image_size))


########################################################################################################################
# Euclidean Distance between two images in interpolation
def euclidean(matrix1, matrix2):
    matrix1 = np.reshape(matrix1, (1, np.shape(matrix1)[0] ** 2))
    matrix2 = np.reshape(matrix2, (1, np.shape(matrix2)[0] ** 2))
    euclidean_distance = np.linalg.norm(matrix1-matrix2)
    return euclidean_distance


def euclidean_plot(tuple_interpolations, num_interp):
    avg_euclidean = []
    for i in range(num_interp-1):
        euclidean_distance = euclidean(tuple_interpolations[i], tuple_interpolations[i+1])
        avg_euclidean.append(euclidean_distance)
        plt.scatter(i, euclidean_distance)

    plt.xlabel("Set of Interpolation")
    plt.ylabel("Euclidean Distance between Images")
    plt.title("Euclidean Distance to Evaluate Smoothness of Interpolations \nAverage Euclidean Value: " + str(np.average(avg_euclidean)))
    plt.ylim(0,)
    plt.show()


########################################################################################################################
# RMSE

def RMSE(matrix1, matrix2):
    matrix1 = np.reshape(matrix1, (1, np.shape(matrix1)[0] ** 2))
    matrix2 = np.reshape(matrix2, (1, np.shape(matrix2)[0] ** 2))
    RMSE = np.linalg.norm(matrix1 - matrix2) / np.sqrt(np.shape(matrix1)[1])
    return RMSE


def RMSE_plot(tuple_interpolations, num_interp):
    avg_coeff_det = []
    for i in range(num_interp-1):
        coeff_det = RMSE(tuple_interpolations[i], tuple_interpolations[i + 1])
        avg_coeff_det.append(coeff_det)
        plt.scatter(i, coeff_det)

    plt.xlabel("Set of Interpolation")
    plt.ylabel("RMSE between Images")
    plt.title("RMSE to Evaluate Smoothness of Interpolations\n Average RMSE Value: "
              + str(np.average(avg_coeff_det)) + "\nPercent Smoothness: " + str(round(100 - (np.average(avg_coeff_det)*100), 3)) + "%")
    plt.ylim(0, 1.1)
    plt.show()


########################################################################################################################
# Step functions that output a tuple with arrays of step-wise transitions
def forward_slash_step(image_size, pixel_step_change):
    B = []
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == (image_size-1)-j:
                A[i][j] = 1
                if (i % pixel_step_change) == 0:
                    B.append(A.copy())
    return B


def forward_slash_step_density(image_size, pixel_step_change):
    B = []
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == (image_size-1)-j:
                A[i][j] = 1 * (0.9 ** i)
                if (i % pixel_step_change) == 0:
                    B.append(A.copy())
    return B


def hot_dog_array_step(image_size, pixel_step_change):
    # Places pixels down the vertical axis to split the box
    B = []
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if j == math.floor((image_size - 1) / 2) or j == math.ceil((image_size - 1) / 2):
                A[i][j] = 1
                if (i % pixel_step_change or j % pixel_step_change) == 0:
                    B.append(A.copy())
    return B


def hot_dog_array_step_density(image_size, pixel_step_change):
    # Places pixels down the vertical axis to split the box
    B = []
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if j == math.floor((image_size - 1) / 2) or j == math.ceil((image_size - 1) / 2):
                A[i][j] = 1 * (0.9 ** i)
                if (i % pixel_step_change or j % pixel_step_change) == 0:
                    B.append(A.copy())
    return B


def basic_box_array_step_gradient(image_size, pixel_step_change):
    B = []
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    # Creates the outside edges of the box
    for i in range(image_size):
        for j in range(image_size):
            if i == 0 or j == 0 or i == image_size - 1 or j == image_size - 1:
                A[i][j] = 1 * (0.9 ** i)
                if (i % pixel_step_change or j % pixel_step_change) == 0:
                    B.append(A.copy())
    return B


########################################################################################################################
hot_dog = hot_dog_array_step_density(image_size, pixel_step_change)[-1]  # The function used to create the steps
basic_box = basic_box_array_step_gradient(image_size, pixel_step_change) [-1]
forward_slash = forward_slash_step_density(image_size, pixel_step_change)[-1]
gradient_test = np.array([hot_dog, basic_box, forward_slash])

# gradient_test = np.array(gradient_test)
'''
num_interps =1
fig = plt.figure()
ax = fig.gca(projection='3d')

array = []
for y in range(image_size):
    row = []
    for x in range(image_size):
        for z in range(num_interps):
            row.append([x, y, z])
    array.append(row)


ax.voxels(np.array(array), edgecolor="k")
'''
# Set the figure size
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# Random data points between 0 and 1
print(gradient_test[0].shape)
print(gradient_test.shape)
# Create a new figure
fig = plt.figure()

# Axis with 3D projection
ax = fig.add_subplot(projection='3d')

# Plot the voxels
cmap = plt.get_cmap("gray")
ax.voxels(gradient_test, edgecolor="k", facecolors=cmap(gradient_test))  # , cmap='gray'

# Display the plot
plt.show()


########################################################################################################################
def gradient_2D(array, filter="gradient"):
    plt.subplot(1, 4, 1), plt.imshow(array, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()

    roberts_cross_x = np.array([[1, 0], [0,
                                         -1]])  # Uses diagonally adjacent pixel to compute the gradient, so it will not necessarily come out correct
    roberts_cross_y = np.array([[0, 1], [-1, 0]])

    gradient_x = np.array([[1, -1], [1, -1]])
    gradient_y = np.array([[-1, -1], [1, 1]])

    perwitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    perwitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if filter == "roberts":
        filter_x = roberts_cross_x
        filter_y = roberts_cross_y
    elif filter == "perwitt":
        filter_x = perwitt_x
        filter_y = perwitt_y
    elif filter == "sobel":
        filter_x = sobel_x
        filter_y = sobel_y
    else:
        filter_x = gradient_x
        filter_y = gradient_y

    G_x = scipy.signal.convolve2d(array.copy(), filter_x, 'valid')
    G_y = scipy.signal.convolve2d(array.copy(), filter_y, 'valid')

    # G_x = ndimage.convolve(gradient_test, gradient_x) #Working, but produces image of same size
    # G_y = ndimage.convolve(gradient_test, gradient_y) #Working, but produces image of same size

    gradient_size = len(G_x)

    gradient_zeros = np.zeros((gradient_size, gradient_size))
    gradient_ones = np.ones((gradient_size, gradient_size))

    x = np.arange(0, gradient_size, 1)
    y = np.arange(0, gradient_size, 1)
    x, y = np.meshgrid(x, y)

    gradient_magnitude = np.sqrt(np.square(G_x) + np.square(G_y))

    # normx = np.linalg.norm(G_x.copy())  # Used to show an even distribution of arrows
    # normy = np.linalg.norm(G_y.copy()) # Should we normalize our gradients?
    # unit_x = G_x / normx
    # unit_y = G_y / normy

    plt.subplot(1, 4, 2), plt.quiver(x, y, G_x, gradient_zeros, units='xy', scale=1, color='red'), plt.imshow(
        gradient_ones, origin='upper', cmap='gray', vmin=0, vmax=1)
    plt.title("Gradient in x Direction")
    plt.colorbar()

    plt.subplot(1, 4, 3), plt.quiver(x, y, gradient_zeros, G_y, units='xy', scale=1, color='red'), plt.imshow(
        gradient_ones, origin='upper', cmap='gray', vmin=0, vmax=1)
    plt.title("Gradient in y Direction")
    plt.colorbar()

    plt.subplot(1, 4, 4), plt.quiver(x, y, G_x, G_y, units='xy', scale=1, color='red'), plt.imshow(gradient_ones,
                                                                                                   origin='upper',
                                                                                                   cmap='gray', vmin=0,
                                                                                                   vmax=1)
    plt.title("Gradients Calculated with " + filter + " filter")
    plt.gca().set_aspect('equal')
    plt.show()

    return G_x, G_y


gradient_2D(gradient_test[0])


def gradient_3D(array_1, array_2, array_3, filter="sobel"):
    array = [array_1, array_2, array_3]
    G_x = []
    G_y = []
    G_z = []
    for i in range(0, 3):
        plt.subplot(1, 3, i+1), plt.imshow(array[i], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        '''
        #USE WHEN MORE FILTERS AVAILABLE
        roberts_cross_x = np.array([[1, 0], [0,
                                             -1]])  # Uses diagonally adjacent pixel to compute the gradient, so it will not necessarily come out correct
        roberts_cross_y = np.array([[0, 1], [-1, 0]])
    
        gradient_x = np.array([[1, -1], [1, -1]])
        gradient_y = np.array([[-1, -1], [1, 1]])
    
        perwitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        perwitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        if filter == "roberts":
            filter_x = roberts_cross_x
            filter_y = roberts_cross_y
        elif filter == "perwitt":
            filter_x = perwitt_x
            filter_y = perwitt_y
        elif filter == "sobel":
            filter_x = sobel_x
            filter_y = sobel_y
        else:
            filter_x = gradient_x
            filter_y = gradient_y
        '''
        sobel_x = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
        sobel_y = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-2, -4, -2], [ 0, 0, 0], [2, 4, 2]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
        sobel_z = np.array([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])

        filter_x = sobel_x[i]
        filter_y = sobel_y[i]
        filter_z = sobel_z[i]

        G_x += scipy.signal.convolve2d(array[i].copy(), filter_x, 'valid')
        G_y += scipy.signal.convolve2d(array[i].copy(), filter_y, 'valid')
        G_z += scipy.signal.convolve2d(array[i].copy(), filter_z, 'valid')
    plt.show()
    G = np.sqrt(np.square(G_x) + np.square(G_x) + np.power(G_x, 2))  # Gradient magnitude calculation

    return G, G_x, G_y, G_z




'''
# DEBUG USE TO PLOT SMOOTHNESS TESTS
tuple_test = basic_box_array_step_gradient(image_size, pixel_step_change)  # The function used to create the steps
# tuple_test = hot_dog_array_step(image_size, pixel_step_change)
# print((tuple_test[0] == tuple_test[1]))
print(np.shape(tuple_test))
plt.matshow(tuple_test[0])
plt.show()


num_interp = np.shape(tuple_test)[0]
plot_rows = 1
plot_columns = num_interp


for i in range(1, num_interp+1):
    plt.subplot(plot_rows, plot_columns, i), plt.imshow(tuple_test[i-1], cmap='gray', vmin=0, vmax=1)
    plt.title(i)
    plt.axis('off')
plt.show()

RMSE_plot(tuple_test, num_interp)
euclidean_plot(tuple_test, num_interp)
'''


