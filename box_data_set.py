import numpy as np
from basic_box import basic_box
from diagonal_box_split import diagonal_box_split
from horizontal_vertical_box_split import horizontal_vertical_box_split
import matplotlib.pyplot as plt  # Use to verify various box shapes and densities
########################################################################################################################
# Basic Box data generator


def make_basic_boxes(i: int, j: int, density_increment, image_size) -> tuple:
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
    new = (A, round(density, 5), number_of_additional_pixels, "Basic_Box")
    return new


########################################################################################################################
# Diagonal Split Box data generator
def make_diagonal_split_boxes(i: int, j: int, density_increment, image_size) -> tuple:
    density = i*density_increment  # causing some sort of decimal float issue
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    print("\nDENSITY: " + str(round(density, 5)))
    number_of_additional_pixels = j
    A = diagonal_box_split(number_of_additional_pixels, density, image_size)
    print("Diagonal Box Split: " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(round(density, 5)) + " Pixel Density")
    # Use to verify various box shapes and densities
    # plt.matshow(A, cmap='gray')
    # plt.title("Diagonal Split Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
    # plt.colorbar()
    # plt.show()
    new = (A, round(density, 5), number_of_additional_pixels, "Diagonal_Box_Split")
    return new


########################################################################################################################
# Horizontal Vertical Split Box data generator
def make_horizontal_vertical_split_boxes(i: int, j: int, density_increment, image_size) -> tuple:
    density = i * density_increment  # causing some sort of decimal float issue
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    print("\nDENSITY: " + str(round(density, 5)))
    number_of_additional_pixels = j
    A = horizontal_vertical_box_split(number_of_additional_pixels, density, image_size)
    print("Horizontal Vertical Split: " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(
        round(density, 5)) + " Pixel Density")
    new = (A, round(density, 5), number_of_additional_pixels, "Horizontal_Box_Split")
    return new


########################################################################################################################
# Make the data using all the box code
def make_boxes(image_size):
    maximum_additional_pixels_basic = int((image_size / 2) - 1)  # this will calculate the amount of additional pixels
    # needed to completely fill the box, if pixels are added symmetrically. This equation is only valid for the basic box shape
    maximum_additional_pixels_split = int((image_size / 4) - 1)  # this equation is only valid for boxes split into 4 sections

    number_of_densities = 5  # The amount of densities that will be tested
    min_density = 0  # The minimum density IS NOT included in the data created, it only serves as a placeholder
    max_density = 1  # The maximum density IS included in the data created
    density_increment = (max_density - min_density) / number_of_densities  # amount that the density will increase between two points

    matrix = []
    # Creates basic boxes which consist of pixels surrounding the border of the image size
    for i in range(1, number_of_densities + 1):
        for j in range(maximum_additional_pixels_basic):
            matrix.append(make_basic_boxes(i, j, density_increment, image_size))

    # Creates diagonal split boxes which consist of pixels surrounding the border of the image as well as across the
    # diagonals, thereby splitting the box into 4 parts
    for i in range(1, number_of_densities + 1):
        for j in range(maximum_additional_pixels_split):
            matrix.append(make_diagonal_split_boxes(i, j, density_increment, image_size))

    # Creates horizontal split boxes which consist of pixels surrounding the border of the image as well as across the
    # center horizontally and vertically, thereby splitting the box into 4 parts
    for i in range(1, number_of_densities+1):
        for j in range(maximum_additional_pixels_split):
            matrix.append(make_horizontal_vertical_split_boxes(i, j, density_increment, image_size))
    return matrix


########################################################################################################################
# Test the boxes!
'''
box_data = make_boxes(28)
desired_density = 0.2
desired_additional_pixels = 0
box_type = "Basic_Box"
print(len(box_data))

for j in range(len(box_data)):
    if desired_density == box_data[j][1] and desired_additional_pixels == int(box_data[j][2]):
        plt.matshow(box_data[j][0],
                    cmap='gray')  # Where index1 is the row, and index2 is the column(which should remain 0)
        plt.title(str(box_data[j][3]) + "\nPixel Density: " + str(
            box_data[j][1]) + "\nAdditional Pixels: " + str(box_data[j][2]))
        plt.colorbar()
        plt.show()

# The data is made such that it can be unpacked: (box_matrix, box_density, additional_pixels, box_shape) = data.load()
'''
