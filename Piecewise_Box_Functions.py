import numpy as np
import matplotlib.pyplot as plt
import math


def basic_box_array(image_size):
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    # Creates the outside edges of the box
    for i in range(image_size):
        for j in range(image_size):
            if i == 0 or j == 0 or i == image_size - 1 or j == image_size - 1:
                A[i][j] = 1
    return A


def back_slash_array(image_size):
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == j:
                A[i][j] = 1
    return A


def forward_slash_array(image_size):
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == (image_size-1)-j:
                A[i][j] = 1
    return A


def hot_dog_array(image_size):
    # Places pixels across the vertical axis to split the box
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if j == math.floor((image_size - 1) / 2) or j == math.ceil((image_size - 1) / 2):
                A[i][j] = 1
    return A


def hamburger_array(image_size):
    # Places pixels across the horizontal axis to split the box
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == math.floor((image_size - 1) / 2) or i == math.ceil((image_size - 1) / 2):
                A[i][j] = 1
    return A


def center_array(image_size):
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for i in range(image_size):
        for j in range(image_size):
            if i == math.floor((image_size-1)/2) and j == math.ceil((image_size-1)/2):
                A[i][j] = 1
            if i == math.floor((image_size-1)/2) and j == math.floor((image_size-1)/2):
                A[i][j] = 1
            if j == math.ceil((image_size-1)/2) and i == math.ceil((image_size-1)/2):
                A[i][j] = 1
            if j == math.floor((image_size-1)/2) and i == math.ceil((image_size-1)/2):
                A[i][j] = 1
    return A


def update_array(array_original, array_new, image_size):
    A = array_original
    for i in range(image_size):
        for j in range(image_size):
            if array_new[i][j] == 1:
                A[i][j] = 1
    return A


def add_pixels(array_original, additional_pixels, image_size):
    # Adds pixels to the thickness of each component of the box
    A = array_original
    A_updated = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    for dens in range(additional_pixels):
        for i in range(1, image_size - 1):
            for j in range(1, image_size - 1):
                if A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1] > 0:
                    A_updated[i][j] = 1
        A = update_array(A, A_updated,image_size)
    return A


'''
image_size = 28  # The desired total shape size
# and 1 adds a pixel on either side of a shape
desired_density = 0.2 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
B = center_array(image_size)
print("Figure")
plt.matshow(B, cmap='gray')
plt.colorbar()
plt.show()
'''
