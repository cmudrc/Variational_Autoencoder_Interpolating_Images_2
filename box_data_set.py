import numpy as np
from basic_box import basic_box
from diagonal_box_split import diagonal_box_split
from horizontal_vertical_box_split import horizontal_vertical_box_split
import matplotlib.pyplot as plt  # Use to verify various box shapes and densities
# The data is made such that it can be unpacked from the following tuple:
# (box_matrix, box_density, additional_pixels, box_shape)


########################################################################################################################
# Basic Box data generator
def make_basic_box(density_level: int, number_of_additional_pixels: int, density_increment, image_size) -> tuple:
    density = density_level * density_increment  # The value that will be the non-zero terms in the matrix
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    A = basic_box(number_of_additional_pixels, density, image_size)
    '''
    # Use to verify various box shapes and densities
    plt.matshow(A, cmap='gray')
    plt.title("Basic Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
    plt.colorbar()
    plt.show()
    '''
    new = (A, round(density, 5), number_of_additional_pixels, "Basic_Box")
    return new


########################################################################################################################
# Diagonal Split Box data generator
def make_diagonal_split_box(density_level: int, number_of_additional_pixels: int, density_increment, image_size) -> tuple:
    density = density_level * density_increment  # The value that will be the non-zero terms in the matrix
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    A = diagonal_box_split(number_of_additional_pixels, density, image_size)
    '''
    # Use to verify various box shapes and densities
    plt.matshow(A, cmap='gray')
    plt.title("Diagonal Split Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
    plt.colorbar()
    plt.show()
    '''
    new = (A, round(density, 5), number_of_additional_pixels, "Diagonal_Box_Split")
    return new


########################################################################################################################
# Horizontal Vertical Split Box data generator
def make_horizontal_vertical_split_box(density_level: int, number_of_additional_pixels: int, density_increment, image_size) -> tuple:
    density = density_level * density_increment  # The value that will be the non-zero terms in the matrix
    # Will be used to create matrices with various densities, ranges from greater than 0 and equal to 1,
    # representing null space and fully solid space respectively
    A = horizontal_vertical_box_split(number_of_additional_pixels, density, image_size)
    '''
    # Use to verify various box shapes and densities
    plt.matshow(A, cmap='gray')
    plt.title("Horizontal Vertical Split Box with " + str(number_of_additional_pixels) + " Additonal Pixel(s) and " + str(density) + " Pixel Density")
    plt.colorbar()
    plt.show()
    '''
    new = (A, round(density, 5), number_of_additional_pixels, "Horizontal_Box_Split")
    return new


########################################################################################################################
# Make the data using all the box code
def make_boxes(image_size, number_of_densities, min_density, max_density):
    # full_box = np.ones((image_size, image_size))  # May be useful to have data with a full box, however, this is not
    # necessary for our work
    # Number of Densities # The amount of densities that will be tested
    # The minimum density IS NOT included in the data created, it only serves as a placeholder (Recommend: 0)
    # The maximum density IS included in the data created (Recommend: 1)
    # Choosing different mins and maxes will require the data to be normalized

    density_increment = (max_density - min_density) / number_of_densities  # amount that the density will increase between two points

    matrix = []
    # Creates basic boxes which consist of pixels surrounding the border of the image size
    for i in range(1, number_of_densities + 1):
        for j in range(image_size):
            # If image size is extremely large, it may be beneficial to divide by 2 and round, as the system should
            # never add more than half of the pixels in width
            if (np.where((make_basic_box(i, j, density_increment, image_size)[0] == float(0)))[0] > 0).any():
                # Want to index the matrix such that we search only through the box-array,
                # then the array that lists the location of the 0's must be indexed, then if that is > 0,
                # the object will be appended to the matrix
                matrix.append(make_basic_box(i, j, density_increment, image_size))
                print("Basic Box: " + str(j) + " Additonal Pixel(s) and " + str(
                    round(i*density_increment, 5)) + " Pixel Density")

    # Creates diagonal split boxes which consist of pixels surrounding the border of the image as well as across the
    # diagonals, thereby splitting the box into 4 parts
    for i in range(1, number_of_densities + 1):
        for j in range(image_size):
            if (np.where((make_diagonal_split_box(i, j, density_increment, image_size)[0] == float(0)))[0] > 0).any():
                matrix.append(make_diagonal_split_box(i, j, density_increment, image_size))
                print("Diagonal Box Split: " + str(j) + " Additonal Pixel(s) and " + str(
                    round(i*density_increment, 5)) + " Pixel Density")

    # Creates horizontal split boxes which consist of pixels surrounding the border of the image as well as across the
    # center horizontally and vertically, thereby splitting the box into 4 parts
    for i in range(1, number_of_densities+1):
        for j in range(image_size):
            if (np.where((make_horizontal_vertical_split_box(i, j, density_increment, image_size)[0] == float(0)))[
                    0] > 0).any():
                matrix.append(make_horizontal_vertical_split_box(i, j, density_increment, image_size))
                print("Horizontal Vertical Split: " + str(j) + " Additonal Pixel(s) and " + str(
                    round(i * density_increment, 5)) + " Pixel Density")
    return matrix


########################################################################################################################
# Test the boxes!
'''
box_data = make_boxes(28, 5, 0, 1)
desired_density = 0.2
desired_additional_pixels = 5
box_type = "Basic_Box"
print(len(box_data))
print(box_data[0])
for j in range(len(box_data)):
    if desired_density == box_data[j][1] and desired_additional_pixels == int(box_data[j][2]):
        plt.matshow(box_data[j][0],
                    cmap='gray')  # Where index1 is the row, and index2 is the column(which should remain 0)
        plt.title(str(box_data[j][3]) + "\nPixel Density: " + str(
            box_data[j][1]) + "\nAdditional Pixels: " + str(box_data[j][2]))
        plt.colorbar()
        plt.show()
'''


