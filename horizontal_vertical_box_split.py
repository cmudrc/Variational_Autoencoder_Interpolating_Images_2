import numpy as np
import matplotlib.pyplot as plt
import math
n = 28  # The desired total shape size
# additional_pixels - Insert the desired number of additional pixels here, where 0 corresponds to shape thickness of 1
# and 1 adds a pixel on either side of a shape
def horizontal_vertical_box_split(additional_pixels, percent_density):
    A = np.zeros((int(n), int(n)))  # Initializes A matrix with 0 values
    A_updated = np.zeros((int(n), int(n)))  # Initializes A matrix with 0 values
    # Creates the outside edges of the box
    print(A[0, 1])
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
        print("dens" + str(dens))
        for i in range(1, n-1):
            for j in range(1, n-1):
                # print(str(i) + "," + str(j))
                if A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] > 0:
                    A_updated[i][j] = 1
        for i in range(n):
            for j in range(n):
                if A_updated[i][j] == 1:
                    A[i][j] = 1  # Replace the A with the new updated A terms, and then add additional pixels by repeating the loop again
    return A*(percent_density/100)


desired_number_of_additional_pixels = 1
desired_density = 100 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
B = horizontal_vertical_box_split(desired_number_of_additional_pixels, desired_density)
print("Figure 2")
plt.matshow(B, cmap='gray')
plt.title("Shape with " + str(desired_number_of_additional_pixels) + " Additonal Pixel(s)")
plt.colorbar()
plt.show()
