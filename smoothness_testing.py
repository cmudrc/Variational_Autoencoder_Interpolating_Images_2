import random
import numpy as np
import matplotlib.pyplot as plt
import math
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