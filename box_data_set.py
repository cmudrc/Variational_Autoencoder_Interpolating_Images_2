import numpy as np
from basic_box_shapes import basic_box, diagonal_box_split, horizontal_vertical_box_split
from complex_box_shapes import back_slash_box, forward_slash_box, back_slash_plus_box, forward_slash_plus_box, hot_dog_box, hamburger_box, x_hamburger_box, x_hot_dog_box, x_plus_box, center_box
import matplotlib.pyplot as plt  # Use to verify various box shapes and densities
# The data is made such that it can be unpacked from the following tuple:
# (box_matrix, box_density, additional_pixels, box_shape)

box_functions = [basic_box, diagonal_box_split, horizontal_vertical_box_split, back_slash_box, forward_slash_box,
                 back_slash_plus_box, forward_slash_plus_box, hot_dog_box, hamburger_box, x_hamburger_box,
                 x_hot_dog_box, x_plus_box]  # center_box was removed for the purpose of our use case

# box_functions = [basic_box, diagonal_box_split, horizontal_vertical_box_split]


########################################################################################################################
# Make the data using all the box code
def make_boxes(image_size, number_of_densities, min_density, max_density):
    # Number of Densities # The amount of densities that will be tested
    # The minimum density IS NOT included in the data created, it only serves as a placeholder (Recommend: 0)
    # The maximum density IS included in the data created (Recommend: 1)
    # Choosing different mins and maxes will require the data to be normalized

    density_increment = (max_density - min_density) / number_of_densities  # amount that the density will increase between two points

    matrix = []
    # Creates basic boxes which consist of pixels surrounding the border of the image size
    for function in box_functions:
        for i in range(1, number_of_densities + 1):
            for j in range(image_size):
                density = i * density_increment
                number_of_additional_pixels = j
                A = (function(number_of_additional_pixels, density, image_size))
                if (np.where((A == float(0)))[0] > 0).any():
                    # Want to index the matrix such that we search only through the box-array,
                    # then the array that lists the location of the 0's must be indexed, then if that is > 0,
                    # the object will be appended to the matrix

                    the_tuple = (A, round(density, 5), number_of_additional_pixels, str(function.__name__), "ground_truth")
                    matrix.append(the_tuple)
                    print(str(function.__name__) + ": " + str(j) + " Additonal Pixel(s) and " + str(
                        round(density, 5)) + " Pixel Density")
                else:
                    break  # ends the looping through additional pixels once there are no 0's present in the array
    return matrix


########################################################################################################################
'''
# Test the boxes!
box_data = make_boxes(28, 5, 0, 1)
desired_density = 1
desired_additional_pixels = 4
desired_box_type = "horizontal_vertical_box_split"
print(len(box_data))
print(box_data[0])
for j in range(len(box_data)):
    if desired_additional_pixels == int(box_data[j][2]) and desired_box_type == box_data[j][3]:
        plt.matshow(box_data[j][0],
                    cmap='gray', vmin=0, vmax=1)  # Where index1 is the row, and index2 is the column(which should remain 0)
        plt.title(str(box_data[j][3]) + "\nPixel Density: " + str(
            box_data[j][1]) + "\nAdditional Pixels: " + str(box_data[j][2]))
        plt.colorbar()
        plt.show()

# The Plot of each of the boxes
image_size = 28  # The desired total shape size
desired_additional_pixels = 0 # Will add pixels next to each square in the box
desired_density = 1  # Determines the grayscale value

plot_rows = 2
plot_columns = 6
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(basic_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("basic_box")

plt.subplot(plot_rows, plot_columns, 2), plt.imshow(diagonal_box_split(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("diagonal_box_split")

plt.subplot(plot_rows, plot_columns, 3), plt.imshow(horizontal_vertical_box_split(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title(" horizontal_vertical_box_split")

plt.subplot(plot_rows, plot_columns, 4), plt.imshow(back_slash_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("back_slash_box")

plt.subplot(plot_rows, plot_columns, 5), plt.imshow(forward_slash_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("forward_slash_box")

plt.subplot(plot_rows, plot_columns, 6), plt.imshow(hot_dog_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("hot_dog_box")

plt.subplot(plot_rows, plot_columns, 7), plt.imshow(hamburger_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("hamburger_box")

plt.subplot(plot_rows, plot_columns, 8), plt.imshow(x_plus_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("x_plus_box")

plt.subplot(plot_rows, plot_columns, 9), plt.imshow(forward_slash_plus_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("forward_slash_plus_box")

plt.subplot(plot_rows, plot_columns, 10), plt.imshow(back_slash_plus_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("back_slash_plus_box")

plt.subplot(plot_rows, plot_columns, 11), plt.imshow(x_hot_dog_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("x_hot_dog_box")

plt.subplot(plot_rows, plot_columns, 12), plt.imshow(x_hamburger_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("x_hamburger_box")

plt.show()
'''
