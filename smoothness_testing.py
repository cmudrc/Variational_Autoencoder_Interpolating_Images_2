import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import warnings
from tensorflow.python.framework.ops import disable_eager_execution
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pacmap

image_size = 28

zeros = np.zeros((image_size, image_size))
ones = np.ones((image_size, image_size))


def euclidean(matrix1, matrix2):
    matrix1 = np.reshape(matrix1, (1, np.shape(matrix1)[0] ** 2))
    matrix2 = np.reshape(matrix2, (1, np.shape(matrix2)[0] ** 2))
    euclidean_distance = np.linalg.norm(matrix1-matrix2)
    return euclidean_distance


def RMSE(matrix1, matrix2):
    matrix1 = np.reshape(matrix1, (1, np.shape(matrix1)[0] ** 2))
    matrix2 = np.reshape(matrix2, (1, np.shape(matrix2)[0] ** 2))
    RMSE = np.linalg.norm(matrix1 - matrix2) / np.sqrt(np.shape(matrix1)[1])
    return RMSE


print("Test Euclidean ")
print(euclidean(zeros, ones))
print(RMSE(zeros, ones))




# Euclidean Distance between two images in interpolation
# for i in range(num_interp-1):
#     diff = predicted_interps[i] - predicted_interps[i + 1]
#     diff_2 = np.power(diff, 2)
#     sqr_diff_2 = pow(np.sum(diff_2), 1/2)
#     plt.scatter(i, sqr_diff_2)
'''
plt.xlabel("Set of Interpolation")
plt.ylabel("Euclidean Distance between Images")
plt.title("Euclidean Distance to Evaluate Smoothness of Interpolations")
plt.ylim(0,)
plt.show()

# RMSE
# for i in range(num_interp-1):
#     diff = predicted_interps[i] - predicted_interps[i + 1]
#     diff_2 = np.power(diff, 2)
#     sqr_diff_2 = pow(np.sum(diff_2)/len(predicted_interps[0]), 1/2)
#     plt.scatter(i, sqr_diff_2)


plt.xlabel("Set of Interpolation")
plt.ylabel("RMSE between Images")
plt.title("RMSE to Evaluate Smoothness of Interpolations")
plt.ylim(0, 1.1)
plt.show()

'''
