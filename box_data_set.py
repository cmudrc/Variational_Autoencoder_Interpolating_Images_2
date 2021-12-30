import numpy as np
from basic_box import basic_box
from diagonal_box_split import diagonal_box_split
from horizontal_vertical_box_split import horizontal_vertical_box_split
import matplotlib.pyplot as plt  # Use to verify various box shapes and densities

image_size = 28  # number of pixels on 2D plane. Must be divisible by 4 to compute diagonal and horizontal split shapes
maximum_additional_pixels_basic = int((image_size/2)-1)  # this will calculate the amount of additional pixels needed
# to completely fill the box, if pixels are added symmetrically. This equation is only valid for the basic box shape.
maximum_additional_pixels_split = int((image_size/4)-1)  # this equation is only valid for boxes split into 4 sections

step_number = 5  # The amount of densities that will be tested
min_density = 0
max_density = 1
density_increment = (max_density-min_density)/step_number  # amount that the density will increase between two points

# Basic Box data generator

def make_basic_boxes(i, j):
    density = i*density_increment  # causing some sort of decimal float issue
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    number_of_additional_pixels = j
    A = basic_box(number_of_additional_pixels, density, image_size)
    print("Basic Box: " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(round(density, 5)) + " Pixel Density")
    # Use to verify various box shapes and densities
    # plt.matshow(A, cmap='gray')
    # plt.title("Basic Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
    # plt.colorbar()
    # plt.show()
    new = tuple((A, density, number_of_additional_pixels, "Basic_Box"))
    return new


print(maximum_additional_pixels_basic)

matrix = np.dtype([])
print(matrix)
for i in range(1, step_number+1):
    for j in range(maximum_additional_pixels_basic):
        matrix = np.append(matrix, [make_basic_boxes(i, j)])


# plt.matshow(matrix[0][0], cmap='gray')
# plt.title("Basic Boxes")
# plt.colorbar()
# plt.show()
print(matrix[1])
print(np.shape(matrix))
'''
# Diagonal Split Box data generator
for i in range(1, step_number+1):
    density = i*density_increment  # causing some sort of decimal float issue
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    print("\nDENSITY: " + str(round(density, 5)))
    for j in range(maximum_additional_pixels_split):
        number_of_additional_pixels = j
        A = diagonal_box_split(number_of_additional_pixels, density, image_size)
        print("Diagonal Box Split: " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(round(density, 5)) + " Pixel Density")
        # Use to verify various box shapes and densities
        # plt.matshow(A, cmap='gray')
        # plt.title("Diagonal Split Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
        # plt.colorbar()
        # plt.show()

# Horizontal Vertical Split Box data generator
for i in range(1, step_number+1):
    density = i*density_increment  # causing some sort of decimal float issue
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    print("\nDENSITY: " + str(round(density, 5)))
    for j in range(maximum_additional_pixels_split):
        number_of_additional_pixels = j
        A = horizontal_vertical_box_split(number_of_additional_pixels, density, image_size)
        print("Horizontal Vertical Split: " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(round(density, 5)) + " Pixel Density")
        # Use to verify various box shapes and densities
        # plt.matshow(A, cmap='gray')
        # plt.title("Diagonal Split Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
        # plt.colorbar()
        # plt.show()



# want to save so it would be referenced as: (box_test, box_density, additional_pixels, box_shape) = data.load()
'''
