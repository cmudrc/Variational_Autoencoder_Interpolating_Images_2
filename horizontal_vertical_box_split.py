import numpy as np
import matplotlib.pyplot as plt
import math


def horizontal_vertical_box_split(additional_pixels, density, n):
    A = np.zeros((int(n), int(n)))  # Initializes A matrix with 0 values
    A_updated = np.zeros((int(n), int(n)))  # Initializes A matrix with 0 values
    # Creates the outside edges of the box
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == n-1 or j == n-1:
                A[i][j] = 1
    # Places pixels across the horizontal and vertical axes to split the box
    for i in range(n):
        for j in range(n):
            if i == math.floor((n-1)/2) or i == math.ceil((n-1)/2):
                A[i][j] = 1
            if j == math.floor((n-1)/2) or j == math.ceil((n-1)/2):
                A[i][j] = 1
    # Adds pixels to the thickness of each component of the box
    for dens in range(additional_pixels):
        for i in range(1, n-1):
            for j in range(1, n-1):
                # print(str(i) + "," + str(j))
                if A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] > 0:
                    A_updated[i][j] = 1
        for i in range(n):
            for j in range(n):
                if A_updated[i][j] == 1:
                    A[i][j] = 1  # Replace the A with the new updated A terms, and then add additional pixels by repeating the loop again
    return A*density


# n = 28  # The desired total shape size
# desired_number_of_additional_pixels = 5
# desired_density = 100 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
# B = horizontal_vertical_box_split(desired_number_of_additional_pixels, desired_density, n)
# print("Figure")
# plt.matshow(B, cmap='gray')
# plt.title("Shape with " + str(desired_number_of_additional_pixels) + " Additonal Pixel(s)")
# plt.colorbar()
# plt.show()
