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


print(euclidean(zeros, ones))
print(RMSE(zeros, zeros))


def forward_slash_step(image_size):
    B = []
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == (image_size-1)-j:
                A[i][j] = 1
                # plt.matshow(A)
                # plt.show()
                # the_tuple = tuple(A)
                # B.append(the_tuple)
                B.append(A.copy())

    return B


tuple_test = forward_slash_step(image_size)
# print((tuple_test[0] == tuple_test[1]))
print(np.shape(tuple_test))
plt.matshow(tuple_test[-1])
plt.show()
'''
predicted_interps = []  # Used to store the predicted interpolations
# Interpolate the Images and Print out to User
for latent_point in range(2, num_interp + 2):  # cycles the latent points through the decoder model to create images
    # generated_image.append((decoder_model_boxes.predict(np.array([latent_matrix[latent_point]]))[0]))
    generated_image = decoder_model_boxes.predict(np.array([latent_matrix[latent_point - 2]]))[0]  # generates an interpolated image based on the latent point
    predicted_interps.append(generated_image[:, :, -1])
    plt.subplot(plot_rows, plot_columns, latent_point), plt.imshow(generated_image[:, :, -1], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
'''

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
