import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import scipy
from scipy import signal
from matplotlib.patches import FancyArrowPatch
import tensorflow
import warnings
from tensorflow.python.framework.ops import disable_eager_execution
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox


image_size = 28
pixel_step_change = 1  # the number of pixels that will be added in between each step

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


########################################################################################################################
gradient_test = hot_dog_array_step_density(image_size, pixel_step_change)[-1]  # The function used to create the steps
plt.subplot(1, 4, 1), plt.imshow(gradient_test, cmap='gray', vmin=0, vmax=1)
plt.colorbar()

# roberts_cross_x = np.array([0, 1, -1, 0])
# roberts_cross_y = np.array([1, 0, 0, -1])
roberts_cross_x = np.array( [[ 0, 1 ], [ -1, 0 ]] )
roberts_cross_y = np.array( [[1, 0 ], [0,-1 ]] )

G_flat = gradient_test.flatten(order='C')
print(np.shape(G_flat))
# G_x = np.convolve(G_flat.copy(), roberts_cross_x, 'same')
# G_y = np.convolve(G_flat.copy(), roberts_cross_y, 'same')

# G_x = scipy.signal.convolve2d(gradient_test, roberts_cross_x, 'same')
# G_y = scipy.signal.convolve2d(gradient_test, roberts_cross_y, 'same')

G_x = ndimage.convolve(gradient_test, roberts_cross_x)
G_y = ndimage.convolve(gradient_test, roberts_cross_y)

print(G_x)
print(np.shape(G_x))

# G_x = np.reshape(G_x, (image_size, image_size))
# G_y = np.reshape(G_y, (image_size, image_size))
# print(G_x)
# print(np.shape(G_x))

# gradient = np.gradient(gradient_test)
plt.subplot(1, 4, 2), plt.imshow(G_x, cmap='gray') # , vmin=0, vmax=1
plt.title("Gradient in x Direction")
plt.colorbar()

plt.subplot(1, 4, 3), plt.imshow(G_y, cmap='gray') # , vmin=0, vmax=1
plt.title("Gradient in y Direction")
plt.colorbar()

print(G_x)
print(np.shape(G_x))

gradient_magnitude = np.sqrt(np.square(G_x) + np.square(G_x))

x = np.arange(0, 28, 1)
y = np.arange(0, 28, 1)
x, y = np.meshgrid(x, y)

norm = np.linalg.norm(np.array((G_x, G_y)), axis=0)
unit_x = G_x / norm
unit_y = G_y / norm

#
plt.subplot(1, 4, 4), plt.quiver(x, y, unit_x, unit_y, units='xy', scale=0.5, color='gray'), plt.imshow(gradient_magnitude, origin='upper', cmap='RdYlBu')
plt.gca().set_aspect('equal')
plt.show()





########################################################################################################################
feature_x = np.arange(-50, 50, 2)
feature_y = np.arange(-50, 50, 2)

x, y = np.meshgrid(feature_x, feature_y)
# z = 0.5*(y-x)**2 + 0.5*(1-x)**2
u = 2*x - y - 1
v = y - x

# Normalize all gradients to focus on the direction not the magnitude
norm = np.linalg.norm(np.array((u, v)), axis=0)
u = u / norm
v = v / norm

fig, ax = plt.subplots(1, 1)
ax.set_aspect(1)
# ax.plot(feature_x, feature_y, c='k')
ax.quiver(x, y, u, v, units='xy', scale=0.5, color='gray')
# ax.contour(x, y, z, 10, cmap='jet')

# arrow = FancyArrowPatch((35, 35), (35+34*0.2, 35+0), arrowstyle='simple',
#                         color='r', mutation_scale=10)
# ax.add_patch(arrow)  # NOTE: this gradient is scaled to make it better visible
plt.show()
# def gradient(matrix):
#
#
# def gradient_plot(gradient_array):


########################################################################################################################
'''
# USE TO PLOT SMOOTHNESS TESTS
tuple_test = forward_slash_step_density(image_size, pixel_step_change)  # The function used to create the steps
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
