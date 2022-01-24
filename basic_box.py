import numpy as np


def basic_box(additional_pixels, density, image_size):
    A = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    A_updated = np.zeros((int(image_size), int(image_size)))  # Initializes A matrix with 0 values
    # Creates the outside edges of the box
    for i in range(image_size):
        for j in range(image_size):
            if i == 0 or j == 0 or i == image_size-1 or j == image_size-1:
                A[i][j] = 1
    # Increase the thickness of each part of the box
    for dens in range(additional_pixels):
        for i in range(1, image_size-1):
            for j in range(1, image_size-1):
                if A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] > 0:
                    A_updated[i][j] = 1
        for i in range(image_size):
            for j in range(image_size):
                if A_updated[i][j] == 1:
                    A[i][j] = 1  # Replace the A with the new updated A terms, and then perform the density increase again
    return A*density


# n = 32  # The desired total shape size
# desired_number_of_additional_pixels = 14 # Insert the desired number of additional pixels here, where 0 corresponds to shape thickness of 1
# # and 1 adds a pixel on either side of a shape
# desired_density = 1 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
# B = basic_box(desired_number_of_additional_pixels, desired_density, n)
# print("Figure")
# plt.matshow(B, cmap='gray')
# plt.title("Shape with " + str(desired_number_of_additional_pixels) + " Additonal Pixel(s)")
# plt.colorbar()
# plt.show()
